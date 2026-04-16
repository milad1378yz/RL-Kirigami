import argparse
import copy
import glob
import os
import time
import warnings
from typing import Optional

from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torch.nn.functional as F

from data_generator.utils import build_context
from kirigami_training.data import KirigamiDataModule
from kirigami_training.data import prepare_training_config
from kirigami_training.metrics import compute_shape_metrics_batch
from kirigami_training.model import build_model
from kirigami_training.sampling import sample_with_solver
from kirigami_training.utils import (
    configure_adamw_cosine,
    load_config,
    precision_from_config,
    prepare_epoch_dirs,
    resolve_run_dir,
    save_epoch_meta,
    save_validation_artifacts,
)


def _context_from_config(config: dict) -> dict:
    data_cfg = config["data"]
    rows = int(data_cfg["grid_rows"])
    cols = int(data_cfg["grid_cols"])
    return build_context(rows, cols)


def _repeat_batch(batch: dict[str, torch.Tensor], repeats: int) -> dict[str, torch.Tensor]:
    if repeats == 1:
        return batch
    return {
        key: value.repeat_interleave(repeats, dim=0)
        for key, value in batch.items()
        if torch.is_tensor(value)
    }


def _group_advantages(rewards: torch.Tensor, group_size: int, eps: float = 1e-6) -> torch.Tensor:
    grouped = rewards.view(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True).clamp_min(eps)
    return ((grouped - mean) / std).view(-1)


def _group_softmax_weights(
    rewards: torch.Tensor,
    *,
    group_size: int,
    temperature: float,
    adv_alpha: Optional[float],
    weight_clip: Optional[float],
) -> torch.Tensor:
    advantages = _group_advantages(rewards, group_size=group_size)
    alpha = float(adv_alpha) if adv_alpha is not None else (1.0 / max(1e-6, float(temperature)))
    weights = torch.exp(alpha * advantages)
    weights = weights.view(-1, group_size)
    weights = weights / (weights.mean(dim=1, keepdim=True) + 1e-8)

    if weight_clip is not None and weight_clip > 1.0:
        weights = weights.clamp(min=1.0 / float(weight_clip), max=float(weight_clip))
        weights = weights / (weights.mean(dim=1, keepdim=True) + 1e-8)
    return weights.view(-1).detach()


def _merge_training_config(config: dict) -> dict:
    base = dict(config.get("training", {}) or {})
    overrides = dict(config.get("rl_training", {}) or {})
    merged = copy.deepcopy(config)
    if overrides:
        base.update(overrides)
        merged["training"] = base
    return merged


def _adv_weighted_flow_matching_loss(
    model: torch.nn.Module,
    path: AffineProbPath,
    batch_rep: dict[str, torch.Tensor],
    x0s: torch.Tensor,
    weights: torch.Tensor,
    *,
    ref_model: Optional[torch.nn.Module] = None,
    ref_reg_weight: float = 0.0,
) -> torch.Tensor:
    images = batch_rep["images"]
    t = torch.rand(images.shape[0], device=images.device)
    sample_info = path.sample(t=t, x_0=x0s, x_1=images)
    masks = batch_rep["masks"]

    pred = model(sample_info.x_t, sample_info.t, masks)
    per_sample = F.mse_loss(pred, sample_info.dx_t, reduction="none").flatten(1).mean(dim=1)
    loss = (weights * per_sample).mean()

    if ref_model is not None and ref_reg_weight > 0.0:
        with torch.inference_mode():
            ref_pred = ref_model(sample_info.x_t, sample_info.t, masks)
        loss = loss + float(ref_reg_weight) * F.mse_loss(pred, ref_pred, reduction="mean")
    return loss


class RLFlowMatchModule(pl.LightningModule):
    def __init__(
        self,
        config: dict,
        *,
        group_size: int,
        reward_temperature: float,
        adv_alpha: Optional[float],
        weight_clip: Optional[float],
        ref_reg_weight: float,
        init_from_ckpt: Optional[str],
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            {
                "config": config,
                "group_size": group_size,
                "reward_temperature": reward_temperature,
                "adv_alpha": adv_alpha,
                "weight_clip": weight_clip,
                "ref_reg_weight": ref_reg_weight,
                "init_from_ckpt": init_from_ckpt,
            }
        )

        self.config = config
        self.context = _context_from_config(config)
        self.x_min = config["data"].get("x_min")
        self.x_max = config["data"].get("x_max")
        self.metric_threshold = float(config["training"].get("mask_threshold", 0.5))
        self.reward_workers = int(config["training"].get("reward_workers", 1))
        self.val_metric_workers = int(config["training"].get("val_metric_workers", 1))

        self.model = build_model(config, device=torch.device("cpu"))
        if init_from_ckpt is not None and str(init_from_ckpt).lower() not in {"", "none"}:
            self._load_model_weights(init_from_ckpt)

        self.path = AffineProbPath(scheduler=CondOTScheduler())
        tr = config["training"]
        self.solver_config = {
            "method": tr["method"],
            "step_size": tr["step_size"],
            "time_points": tr["time_points"],
        }
        self.reward_metric = str(tr.get("reward_metric", "siou") or "siou").lower()
        if self.reward_metric not in {"iou", "siou"}:
            raise ValueError("reward_metric must be 'iou' or 'siou' for the current shape pipeline.")

        self.reward_cfg = {
            "transform": str(tr.get("reward_transform", "logit") or "logit"),
            "power": float(tr.get("reward_power", 1.0)),
            "logit_eps": float(tr.get("reward_logit_eps", 1e-4)),
            "scale": float(tr.get("reward_scale", 1.0)),
            "shift": float(tr.get("reward_shift", 0.0)),
            "penalty_scale": float(tr.get("reward_penalty_scale", 1.0)),
        }
        self.shape_penalty_cfg = dict(tr.get("shape_penalty", {}) or {})

        self.ref_model = None
        if ref_reg_weight > 0.0:
            self.ref_model = copy.deepcopy(self.model)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad_(False)

        channels = int(config["model_config"]["in_channels"])
        in_h, in_w = tuple(config["model_config"]["input_size"])
        mask_h, mask_w = tuple(config["model_config"]["mask_size"])
        self.example_input_array = {
            "images": torch.randn(2, channels, in_h, in_w),
            "masks": torch.randn(2, 1, mask_h, mask_w),
        }

    def _load_model_weights(self, ckpt_path: str) -> None:
        ckpt = torch.load(os.path.expanduser(ckpt_path), map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        model_state = {k.split("model.", 1)[1]: v for k, v in state.items() if k.startswith("model.")}
        missing, unexpected = self.model.load_state_dict(model_state, strict=False)
        print(
            f"Loaded model weights from {ckpt_path}. Missing: {len(missing)} Unexpected: {len(unexpected)}"
        )

    def configure_optimizers(self):
        return configure_adamw_cosine(self.model, self.trainer, self.config["training"])

    def _plain_fm_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        images = batch["images"]
        masks = batch["masks"]
        x0 = torch.randn_like(images)
        t = torch.rand(images.shape[0], device=images.device)
        sample_info = self.path.sample(t=t, x_0=x0, x_1=images)
        pred = self.model(sample_info.x_t, sample_info.t, masks)
        return F.mse_loss(pred, sample_info.dx_t)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        group_size = int(self.hparams["group_size"]) or 1
        batch_rep = _repeat_batch(batch, repeats=group_size)
        x0s = torch.randn_like(batch_rep["images"])
        masks = batch_rep["masks"]
        metric_masks = batch_rep.get("metric_masks", masks)

        was_training = self.model.training
        self.model.eval()
        with torch.inference_mode():
            sol = sample_with_solver(
                model=self.model,
                x_init=x0s,
                solver_config=self.solver_config,
                masks=masks,
                return_intermediates=False,
            )
        self.model.train(was_training)

        pred_x = sol[-1] if sol.dim() == 5 else sol
        metrics = compute_shape_metrics_batch(
            pred_x,
            metric_masks,
            self.context,
            x_min=self.x_min,
            x_max=self.x_max,
            threshold=self.metric_threshold,
            shape_penalty_cfg=self.shape_penalty_cfg,
            reward_cfg=self.reward_cfg,
            reward_metric=self.reward_metric,
            num_workers=self.reward_workers,
            device=self.device,
        )
        rewards = metrics["reward"]
        if rewards is None:
            raise RuntimeError("RL rewards were not computed.")
        reward_raw = metrics["reward_raw"]
        metric_vals = metrics[self.reward_metric]

        weights = _group_softmax_weights(
            rewards,
            group_size=group_size,
            temperature=float(self.hparams["reward_temperature"]),
            adv_alpha=self.hparams["adv_alpha"],
            weight_clip=self.hparams["weight_clip"],
        )

        loss = _adv_weighted_flow_matching_loss(
            self.model,
            self.path,
            batch_rep,
            x0s,
            weights,
            ref_model=self.ref_model,
            ref_reg_weight=float(self.hparams["ref_reg_weight"] or 0.0),
        )

        batch_size = batch["images"].shape[0]
        grouped_rewards = reward_raw.view(batch_size, group_size)
        grouped_shaped_rewards = rewards.view(batch_size, group_size)
        grouped_metrics = metric_vals.view(batch_size, group_size)
        grouped_penalty = metrics["penalty"].view(batch_size, group_size)
        grouped_weights = weights.view(batch_size, group_size)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/reward_mean", grouped_rewards.mean(dim=1).mean(), on_step=True, on_epoch=True)
        self.log("train/reward_max", grouped_rewards.max(dim=1).values.mean(), on_step=True, on_epoch=True)
        self.log(
            "train/reward_shaped_mean",
            grouped_shaped_rewards.mean(dim=1).mean(),
            on_step=True,
            on_epoch=True,
        )
        self.log(
            f"train/{self.reward_metric}_mean",
            grouped_metrics.mean(dim=1).mean(),
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train/shape_penalty_mean",
            grouped_penalty.mean(dim=1).mean(),
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train/invalid_any_rate",
            (metrics["invalid_quad_count"].view(batch_size, group_size) > 0).float().mean(),
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train/overlap_ratio_mean",
            metrics["overlap_ratio"].view(batch_size, group_size).mean(dim=1).mean(),
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train/build_fail_rate",
            (1.0 - metrics["build_ok"]).view(batch_size, group_size).mean(dim=1).mean(),
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train/fill_error_mean",
            metrics["fill_error"].view(batch_size, group_size).mean(dim=1).mean(),
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train/range_violation_mean",
            metrics["range_violation_l1"].view(batch_size, group_size).mean(dim=1).mean(),
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train/clipped_fraction_mean",
            metrics["clipped_fraction"].view(batch_size, group_size).mean(dim=1).mean(),
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train/weight_mean",
            grouped_weights.mean(dim=1).mean(),
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train/weight_max",
            grouped_weights.max(dim=1).values.mean(),
            on_step=True,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        loss = self._plain_fm_loss(batch)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        metric_masks = batch.get("metric_masks", batch["masks"])

        with torch.inference_mode():
            sol = sample_with_solver(
                self.model,
                torch.randn_like(batch["images"]),
                self.solver_config,
                masks=batch["masks"],
                return_intermediates=False,
            )
            pred_x = sol[-1] if sol.dim() == 5 else sol

        metrics = compute_shape_metrics_batch(
            pred_x,
            metric_masks,
            self.context,
            x_min=self.x_min,
            x_max=self.x_max,
            threshold=self.metric_threshold,
            num_workers=self.val_metric_workers,
            device=self.device,
        )

        self.log("val/IoU", metrics["iou"].mean(), on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/SIoU", metrics["siou"].mean(), on_epoch=True, prog_bar=True, sync_dist=True)
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
            "val/fill_error_mean",
            metrics["fill_error"].mean(),
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
        if (self.current_epoch + 1) % int(self.config["training"]["val_freq"]) != 0:
            return
        if hasattr(self.trainer, "is_global_zero") and not self.trainer.is_global_zero:
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
            num_samples=int(self.config["training"]["num_val_samples"]),
            context=self.context,
            x_min=self.x_min,
            x_max=self.x_max,
            save_triplets=False,
            plot_steps=True,
        )
        save_epoch_meta(epoch_dir, epoch, self.config)


def _resolve_last_checkpoint(root_ckpt_dir: str, run_name: str) -> Optional[str]:
    run_dir = os.path.join(root_ckpt_dir, run_name)
    last_ckpt = os.path.join(run_dir, "last.ckpt")
    if os.path.isfile(last_ckpt):
        return last_ckpt
    candidates = sorted(glob.glob(os.path.join(run_dir, "*.ckpt")), key=os.path.getmtime)
    return candidates[-1] if candidates else None


def run_rl_training(config: dict, *, config_path: str, init_from: str, resume: str = "last") -> None:
    config = _merge_training_config(config)
    tr = config["training"]
    seed_everything(int(tr["seed"]), workers=True)

    precision = precision_from_config(tr["mixed_precision"])
    root_ckpt_dir = os.path.expanduser(tr["checkpoint_dir"])
    base_run = config.get("run_name", os.path.splitext(os.path.basename(config_path))[0])
    run_name = f"{base_run}_RL"
    tb_root = os.path.expanduser(tr["log_dir"])
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
        every_n_train_steps=(int(tr["ckpt_every_n_steps"]) if tr.get("ckpt_every_n_steps") else None),
        every_n_epochs=(None if tr.get("ckpt_every_n_steps") else max(1, int(tr["val_freq"]))),
        save_on_train_epoch_end=(False if tr.get("ckpt_every_n_steps") else True),
    )
    callbacks = [ckpt_cb, LearningRateMonitor(logging_interval="step")]
    if tr.get("swa", False):
        callbacks.append(StochasticWeightAveraging(swa_lrs=float(tr["swa_lr"])))

    trainer = Trainer(
        default_root_dir=root_ckpt_dir,
        max_epochs=int(tr["num_epochs"]),
        max_steps=int(tr.get("max_steps", -1)),
        precision=precision,
        accumulate_grad_batches=int(tr.get("gradient_accumulation_steps", 1)),
        gradient_clip_val=(float(tr["grad_clip_norm"]) if tr.get("grad_clip_norm") else None),
        check_val_every_n_epoch=1,
        val_check_interval=tr.get("val_check_interval"),
        enable_progress_bar=True,
        logger=logger,
        callbacks=callbacks,
        accelerator=tr["accelerator"],
        devices=tr["devices"],
        strategy="auto",
        deterministic=bool(tr.get("deterministic", False)),
        log_every_n_steps=int(tr.get("log_every_n_steps", 50)),
        num_sanity_val_steps=int(tr.get("num_sanity_val_steps", 0)),
    )

    resolved_init = init_from
    if str(init_from).lower() == "last":
        resolved_init = _resolve_last_checkpoint(root_ckpt_dir, base_run)
    elif str(init_from).lower() in {"", "none"}:
        resolved_init = None
    else:
        resolved_init = os.path.expanduser(init_from)
    if resolved_init is not None and not os.path.isfile(resolved_init):
        raise FileNotFoundError(f"Could not resolve RL init checkpoint '{init_from}'.")

    datamodule = KirigamiDataModule(config)
    module = RLFlowMatchModule(
        config,
        group_size=int(tr["group_size"]),
        reward_temperature=float(tr["reward_temperature"]),
        adv_alpha=tr.get("adv_alpha"),
        weight_clip=tr.get("weight_clip"),
        ref_reg_weight=float(tr.get("ref_reg_weight", 0.0) or 0.0),
        init_from_ckpt=resolved_init,
    )

    ckpt_path = None
    if resume and str(resume).lower() not in {"", "none"}:
        if str(resume).lower() == "last":
            ckpt_path = _resolve_last_checkpoint(root_ckpt_dir, run_name)
        else:
            resume_path = os.path.expanduser(resume)
            ckpt_path = resume_path if os.path.isfile(resume_path) else None
        if ckpt_path is None:
            print(f"[WARN] Could not resolve --resume '{resume}'. Starting RL without resume.")

    trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)


def main() -> None:
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="RL fine-tuning for the current kirigami Flow Matching model.")
    parser.add_argument("--config_path", type=str, default="configs/training.yaml")
    parser.add_argument("--init_from", type=str, default="last")
    parser.add_argument("--resume", type=str, default="last")
    args = parser.parse_args()

    config = load_config(args.config_path)
    config = prepare_training_config(config)
    run_rl_training(config, config_path=args.config_path, init_from=args.init_from, resume=args.resume)


if __name__ == "__main__":
    main()
