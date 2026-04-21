"""Distillation fine-tuning toward search-found z-targets.

Separate pipeline from `rl_training.py` (GRPO). Stages:
  1. Load a pretrained FM checkpoint.
  2. Build per-mask targets: best-of-N rejection seed + (1+lambda) evolution
     strategy on rendered SIoU in z = log10(x) space.
  3. Fine-tune by differentiable Euler sampling + MSE to the target z.

Targets are cached to `checkpoints/{run}_distill/distill_targets.pt` so stage 2
only runs once per mask set.
"""
import argparse
import os
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from data_generator.utils import build_geometry_context
from kirigami_training.data import (
    KirigamiDataModule,
    model_to_x_space,
    prepare_training_config,
)
from kirigami_training.metrics import compute_shape_metrics_batch
from kirigami_training.model import build_model
from kirigami_training.sampling import sample_with_solver
from kirigami_training.targets import compute_distillation_targets, euler_sample
from kirigami_training.utils import (
    load_config,
    resolve_checkpoint_path,
    select_training_config,
)


def _load_model_weights(model: torch.nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(os.path.expanduser(ckpt_path), map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model_state = {k.split("model.", 1)[1]: v for k, v in state.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    print(f"Loaded {ckpt_path} (missing={len(missing)} unexpected={len(unexpected)})")


def _save_ckpt(path: str, model: torch.nn.Module, config: dict) -> None:
    state = {"state_dict": {f"model.{k}": v for k, v in model.state_dict().items()}, "config": config}
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def run_distill_training(config: dict, *, config_path: str, init_from: str) -> None:
    tr = config["training"]
    torch.manual_seed(int(tr["seed"]))
    np.random.seed(int(tr["seed"]))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    root_ckpt_dir = os.path.expanduser(tr["checkpoint_dir"])
    base_run = config.get("run_name", os.path.splitext(os.path.basename(config_path))[0])
    run_name = f"{base_run}_distill"
    out_dir = os.path.join(root_ckpt_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    init_ckpt = resolve_checkpoint_path(root_ckpt_dir, base_run, init_from)
    if init_ckpt is None and init_from not in {None, "", "none"}:
        raise FileNotFoundError(f"Could not resolve init checkpoint '{init_from}'.")

    datamodule = KirigamiDataModule(config)
    datamodule.setup()
    train_ds = datamodule.train_loader.dataset
    val_loader = datamodule.val_dataloader()

    model = build_model(config, device=device).to(device)
    if init_ckpt:
        _load_model_weights(model, init_ckpt)

    context = build_geometry_context(
        int(config["data"]["grid_rows"]),
        int(config["data"]["grid_cols"]),
    )
    source_std = float(tr["source_noise_std"])
    x_min = config["data"]["x_min"]
    x_max = config["data"]["x_max"]
    solver_config = {
        "method": tr["method"],
        "step_size": tr["step_size"],
        "time_points": tr["time_points"],
        "source_noise_std": source_std,
    }

    images = train_ds.images.to(device)
    masks = train_ds.masks.to(device)
    cache_path = os.path.join(out_dir, "distill_targets.pt")
    targets_z, targets_siou = compute_distillation_targets(
        model, images, masks, masks,
        solver_config=solver_config, context=context,
        x_min=x_min, x_max=x_max, source_std=source_std,
        n_candidates=int(tr["n_candidates"]),
        es_iters=int(tr["es_iters"]),
        es_pop_size=int(tr["es_pop_size"]),
        es_sigma_init=float(tr["es_sigma_init"]),
        es_sigma_min=float(tr["es_sigma_min"]),
        device=device,
        cache_path=cache_path,
    )
    targets_siou_list = targets_siou.cpu().numpy().tolist()
    print(f"Targets per-mask SIoU: {targets_siou_list} mean={float(np.mean(targets_siou_list)):.4f}")

    # Dropout off during distillation: keeps the Euler trajectory deterministic
    # given x0 so MSE to a fixed target is well-defined.
    model.eval()

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(tr["lr"]),
        weight_decay=float(tr.get("weight_decay", 0.0)),
        betas=(0.9, 0.95),
    )
    tb_dir = os.path.join(root_ckpt_dir, "tb", run_name, time.strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(tb_dir)

    time_points_train = int(tr.get("time_points_train", tr["time_points"]))
    repeat = int(tr.get("target_repeat", 1))

    def tile(x: torch.Tensor, r: int) -> torch.Tensor:
        return x if r == 1 else x.repeat_interleave(r, dim=0)
    targets_z_t = tile(targets_z, repeat)
    masks_t = tile(masks, repeat)

    val_batch = next(iter(val_loader))
    v_images = val_batch["images"].to(device)
    v_masks = val_batch["masks"].to(device)
    v_metric_masks = val_batch["metric_masks"].to(device)

    num_eval = int(tr.get("num_eval_draws", 10))

    def eval_siou() -> tuple[float, float, np.ndarray]:
        vals, per = [], []
        with torch.no_grad():
            for _ in range(num_eval):
                x0 = source_std * torch.randn_like(v_images)
                pred_z = sample_with_solver(
                    model, x0, solver_config, masks=v_masks, return_intermediates=False,
                )
                pred_x = model_to_x_space(pred_z, x_min=x_min, x_max=x_max)
                m = compute_shape_metrics_batch(
                    pred_x, v_metric_masks, context,
                    x_min=x_min, x_max=x_max, device=device,
                )
                vals.append(m["siou"].mean().item())
                per.append(m["siou"].detach().cpu().numpy())
        return float(np.mean(vals)), float(np.max(vals)), np.stack(per, 0).mean(0)

    m0, mx0, ps0 = eval_siou()
    print(f"[pre] SIoU mean={m0:.4f} max={mx0:.4f} per={ps0.tolist()}")
    writer.add_scalar("val/SIoU", m0, 0)

    steps = int(tr["steps"])
    eval_every = int(tr["eval_every"])
    grad_clip = float(tr.get("grad_clip_norm", 1.0))
    best = m0

    t0 = time.time()
    loss_ema = None
    for step in range(1, steps + 1):
        x0 = source_std * torch.randn_like(targets_z_t)
        pred_z = euler_sample(model, x0, masks_t, time_points=time_points_train)
        loss = F.mse_loss(pred_z, targets_z_t)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()

        lv = float(loss.detach().cpu())
        loss_ema = lv if loss_ema is None else 0.99 * loss_ema + 0.01 * lv
        writer.add_scalar("train/loss", lv, step)
        if step % 20 == 0:
            print(f"step={step} loss={lv:.4f} ema={loss_ema:.4f} t={time.time()-t0:.0f}s")

        if step % eval_every == 0 or step == steps:
            mean_s, max_s, ps = eval_siou()
            writer.add_scalar("val/SIoU", mean_s, step)
            writer.add_scalar("val/SIoU_max", max_s, step)
            print(
                f"[eval step={step}] SIoU mean={mean_s:.4f} max={max_s:.4f} "
                f"min_per={ps.min():.4f} per={ps.tolist()}"
            )
            _save_ckpt(os.path.join(out_dir, "last.ckpt"), model, config)
            if mean_s > best:
                best = mean_s
                _save_ckpt(os.path.join(out_dir, f"distill-best-SIoU{mean_s:.4f}.ckpt"), model, config)

    print(f"[done] best SIoU={best:.4f}")
    writer.close()


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Distillation fine-tuning to search-found targets.")
    parser.add_argument("--config_path", type=str, default="configs/training.yaml")
    parser.add_argument("--init_from", type=str, default="last")
    args = parser.parse_args()

    config = load_config(args.config_path)
    config = select_training_config(config, "distill_training")
    config = prepare_training_config(config)
    run_distill_training(config, config_path=args.config_path, init_from=args.init_from)


if __name__ == "__main__":
    main()
