import argparse
import os
import time
import warnings

from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F

from data_generator.utils import build_geometry_context
from kirigami_training.data import KirigamiDataModule
from kirigami_training.data import model_to_x_space
from kirigami_training.data import prepare_training_config
from kirigami_training.metrics import compute_shape_metrics_batch
from kirigami_training.model import build_model
from kirigami_training.sampling import sample_with_solver
from kirigami_training.utils import (
    configure_adamw_cosine,
    load_config,
    precision_from_config,
    prepare_epoch_dirs,
    resolve_checkpoint_path,
    resolve_run_dir,
    save_epoch_meta,
    select_training_config,
    save_validation_artifacts,
    TrainingTQDMProgressBar,
)


def _geometry_context_from_config(config: dict) -> dict:
    data_cfg = config["data"]
    rows = int(data_cfg["grid_rows"])
    cols = int(data_cfg["grid_cols"])
    return build_geometry_context(rows, cols)


class FlowMatchModule(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.context = _geometry_context_from_config(config)
        self.x_min = config["data"].get("x_min")
        self.x_max = config["data"].get("x_max")
        self.metric_threshold = float(config["training"].get("mask_threshold", 0.5))
        self.source_noise_std = float(config["training"].get("source_noise_std", 0.5))

        self.model = build_model(config, device=torch.device("cpu"))
        self.path = AffineProbPath(scheduler=CondOTScheduler())

        tr = config["training"]
        self.val_freq = int(tr["val_freq"])
        self.num_val_samples = int(tr["num_val_samples"])
        self.solver_config = {
            "method": tr["method"],
            "step_size": tr["step_size"],
            "time_points": tr["time_points"],
            "source_noise_std": self.source_noise_std,
        }

        if tr.get("allow_tf32", False) and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        channels = int(config["model_config"]["in_channels"])
        in_h, in_w = tuple(config["model_config"]["input_size"])
        mask_h, mask_w = tuple(config["model_config"]["mask_size"])
        self.example_input_array = {
            "x": torch.randn(2, channels, in_h, in_w),
            "t": torch.rand(2),
            "masks": torch.randn(2, 1, mask_h, mask_w),
        }

    def _maybe_ot_couple(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        x0_cpu = x0.detach().to("cpu", dtype=torch.float32)
        x1_cpu = x1.detach().to("cpu", dtype=torch.float32)
        bsz = x0_cpu.shape[0]
        x0_flat = x0_cpu.view(bsz, -1)
        x1_flat = x1_cpu.view(bsz, -1)
        x0_sq = (x0_flat ** 2).sum(dim=1, keepdim=True)
        x1_sq = (x1_flat ** 2).sum(dim=1, keepdim=True).t()
        cost = (x0_sq + x1_sq - 2.0 * (x0_flat @ x1_flat.t())).clamp_min(0.0)
        row_ind, col_ind = linear_sum_assignment(cost.numpy())
        x0_perm = torch.empty_like(x0_cpu)
        x0_perm[col_ind] = x0_cpu[row_ind]
        return x0_perm.to(device=x0.device, dtype=x0.dtype)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        masks: torch.Tensor,
    ):
        return self.model(x, t, masks)

    def _flow_matching_loss(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        x0 = self.source_noise_std * torch.randn_like(images)
        x0 = self._maybe_ot_couple(x0, images)
        t = torch.rand(images.shape[0], device=images.device)
        sample_info = self.path.sample(t=t, x_0=x0, x_1=images)
        pred = self.model(sample_info.x_t, sample_info.t, masks)
        return F.mse_loss(pred, sample_info.dx_t)

    def training_step(self, batch, batch_idx):
        images = batch["images"]
        masks = batch["masks"]
        loss = self._flow_matching_loss(images, masks)
        self.log("train/loss_step", loss, on_step=True, on_epoch=False, prog_bar=True, logger=False, sync_dist=True)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["images"]
        masks = batch["masks"]
        metric_masks = batch["metric_masks"]
        loss = self._flow_matching_loss(images, masks)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        with torch.no_grad():
            pred_z = sample_with_solver(
                self.model,
                self.source_noise_std * torch.randn_like(images),
                self.solver_config,
                masks=masks,
                return_intermediates=False,
            )
            pred_x = model_to_x_space(pred_z, x_min=self.x_min, x_max=self.x_max)

        metrics = compute_shape_metrics_batch(
            pred_x,
            metric_masks,
            self.context,
            x_min=self.x_min,
            x_max=self.x_max,
            threshold=self.metric_threshold,
            device=self.device,
        )

        self.log("val/IoU", metrics["iou"].mean(), on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val/SIoU", metrics["siou"].mean(), on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(
            "val/fill_error_mean",
            metrics["fill_error"].mean(),
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "val/invalid_any_rate",
            (metrics["invalid_quad_count"] > 0).float().mean(),
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "val/overlap_ratio_mean",
            metrics["overlap_ratio"].mean(),
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "val/build_fail_rate",
            (1.0 - metrics["build_ok"]).mean(),
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "val/range_violation_mean",
            metrics["range_violation_l1"].mean(),
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "val/clipped_fraction_mean",
            metrics["clipped_fraction"].mean(),
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

    def on_validation_epoch_end(self) -> None:
        if (self.current_epoch + 1) % self.val_freq != 0:
            return
        if not self.trainer.is_global_zero:
            return

        run_dir = resolve_run_dir(self.trainer)
        epoch = self.current_epoch + 1
        epoch_dir, outdir = prepare_epoch_dirs(run_dir, epoch)

        save_validation_artifacts(
            model=self.model,
            dataloader=self.trainer.datamodule.val_dataloader(),
            solver_config=self.solver_config,
            device=self.device,
            outdir=outdir,
            num_samples=self.num_val_samples,
            context=self.context,
            x_min=self.x_min,
            x_max=self.x_max,
            save_triplets=True,
            plot_steps=True,
        )
        save_epoch_meta(epoch_dir, epoch, self.config)

    def configure_optimizers(self):
        return configure_adamw_cosine(self.model, self.trainer, self.config["training"])


def run_flow_training(config: dict, *, config_path: str) -> None:
    tr = config["training"]
    seed_everything(int(tr["seed"]), workers=True)

    precision = precision_from_config(tr["mixed_precision"])
    root_ckpt_dir = os.path.expanduser(tr["checkpoint_dir"])
    run_name = config.get("run_name", os.path.splitext(os.path.basename(config_path))[0])
    tb_root = os.path.join(root_ckpt_dir, "tb")
    tb_version = time.strftime("%Y%m%d-%H%M%S")

    logger = TensorBoardLogger(save_dir=tb_root, name=run_name, version=tb_version)

    ckpt_monitor = str(tr.get("ckpt_monitor", "") or "").strip() or "val/SIoU"
    ckpt_mode = str(tr.get("ckpt_mode", "") or "").strip() or "max"
    ckpt_filename = str(tr.get("ckpt_filename", "") or "").strip()
    if not ckpt_filename:
        metric_slug = ckpt_monitor.replace("/", "")
        ckpt_filename = f"epoch{{epoch:03d}}-{metric_slug}{{{ckpt_monitor}:.6f}}"

    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(root_ckpt_dir, run_name),
        filename=ckpt_filename,
        monitor=ckpt_monitor,
        mode=ckpt_mode,
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
        every_n_epochs=max(1, int(tr["val_freq"])),
        save_on_train_epoch_end=True,
    )
    callbacks = [ckpt_cb, LearningRateMonitor(logging_interval="step"), TrainingTQDMProgressBar()]
    if tr.get("swa", False):
        callbacks.append(StochasticWeightAveraging(swa_lrs=float(tr["swa_lr"])))

    trainer = Trainer(
        default_root_dir=root_ckpt_dir,
        max_epochs=int(tr["num_epochs"]),
        max_steps=int(tr.get("max_steps", -1)),
        precision=precision,
        accumulate_grad_batches=int(tr.get("gradient_accumulation_steps", 1)),
        gradient_clip_val=(float(tr["grad_clip_norm"]) if tr.get("grad_clip_norm") else None),
        check_val_every_n_epoch=max(1, int(tr.get("val_freq", 1))),
        val_check_interval=tr.get("val_check_interval"),
        enable_progress_bar=True,
        logger=logger,
        callbacks=callbacks,
        accelerator=tr["accelerator"],
        devices=tr["devices"],
        deterministic=bool(tr.get("deterministic", False)),
        log_every_n_steps=int(tr.get("log_every_n_steps", 50)),
        num_sanity_val_steps=0,
    )

    datamodule = KirigamiDataModule(config)
    module = FlowMatchModule(config)

    ckpt_path = resolve_checkpoint_path(root_ckpt_dir, run_name, "last")
    if ckpt_path is not None:
        print(f"Resuming FM from {ckpt_path}")

    trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)


def main() -> None:
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="Flow Matching training for the current kirigami x-matrix generator.")
    parser.add_argument("--config_path", type=str, default="configs/training.yaml")
    args = parser.parse_args()

    config = load_config(args.config_path)
    config = select_training_config(config, "fm_training")
    config = prepare_training_config(config)
    run_flow_training(config, config_path=args.config_path)


if __name__ == "__main__":
    main()
