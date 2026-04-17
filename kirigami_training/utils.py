import glob
import json
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from .sampling import plot_solver_steps, sample_with_solver
from data_generator.utils import mask_overlay_rgb, render_structure_mask_and_metrics
from data_generator.visualization import plot_x_matrix_structure


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def precision_from_config(mixed_precision: str):
    mode = str(mixed_precision).lower()
    return {"no": 32, "fp16": "16-mixed", "bf16": "bf16-mixed"}.get(mode, 32)


def configure_adamw_cosine(model: torch.nn.Module, trainer, training_cfg: dict):
    lr = float(training_cfg["lr"])
    weight_decay = float(training_cfg["weight_decay"])
    warmup_epochs = int(training_cfg["warmup_epochs"])
    min_lr_factor = float(training_cfg["min_lr_factor"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    cosine_epochs = max(1, trainer.max_epochs - max(0, warmup_epochs))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=lr * min_lr_factor,
    )
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = cosine

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "epoch",
        },
    }


def resolve_run_dir(trainer) -> str:
    run_dir = None
    ckpt_cb = getattr(trainer, "checkpoint_callback", None)
    if ckpt_cb is not None and getattr(ckpt_cb, "dirpath", None):
        run_dir = ckpt_cb.dirpath
    if not run_dir:
        run_dir = trainer.default_root_dir
    return run_dir


def resolve_last_checkpoint(root_ckpt_dir: str, run_name: str) -> Optional[str]:
    run_dir = os.path.join(os.path.expanduser(root_ckpt_dir), run_name)
    last_candidates = sorted(glob.glob(os.path.join(run_dir, "last*.ckpt")), key=os.path.getmtime)
    if last_candidates:
        return last_candidates[-1]

    candidates = sorted(glob.glob(os.path.join(run_dir, "*.ckpt")), key=os.path.getmtime)
    return candidates[-1] if candidates else None


def resolve_checkpoint_path(root_ckpt_dir: str, run_name: str, checkpoint: Optional[str]) -> Optional[str]:
    if checkpoint is None:
        return None

    checkpoint = str(checkpoint).strip()
    if checkpoint.lower() in {"", "none"}:
        return None
    if checkpoint.lower() == "last":
        return resolve_last_checkpoint(root_ckpt_dir, run_name)

    checkpoint_path = os.path.expanduser(checkpoint)
    return checkpoint_path if os.path.isfile(checkpoint_path) else None


def prepare_epoch_dirs(run_dir: str, epoch: int) -> tuple[str, str]:
    epoch_dir = os.path.join(run_dir, f"epoch_{epoch}")
    outdir = os.path.join(epoch_dir, f"val_samples_epoch_{epoch}")
    os.makedirs(outdir, exist_ok=True)
    return epoch_dir, outdir


def save_epoch_meta(epoch_dir: str, epoch: int, config: dict) -> None:
    meta_path = os.path.join(epoch_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump({"epoch": int(epoch), "config": config}, handle, indent=2)


def _save_grayscale(path: str, image: np.ndarray) -> None:
    plt.imsave(path, np.asarray(image, dtype=np.float32), cmap="gray", vmin=0.0, vmax=1.0)


def _save_rgb(path: str, image: np.ndarray) -> None:
    plt.imsave(path, np.asarray(image, dtype=np.float32))


def _save_structure_image(
    path: str,
    x_matrix: np.ndarray,
    context: dict,
    *,
    mask_2d: Optional[np.ndarray] = None,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    try:
        plot_x_matrix_structure(
            ax,
            x_matrix,
            context,
            mask_2d=mask_2d,
            x_min=x_min,
            x_max=x_max,
            normalize_phi=None,
        )
    except Exception as exc:
        ax.axis("off")
        ax.text(0.5, 0.5, f"invalid\n{exc}", ha="center", va="center")
    fig.tight_layout(pad=0.05)
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_validation_artifacts(
    *,
    model: torch.nn.Module,
    dataloader,
    solver_config: dict,
    device: torch.device,
    outdir: str,
    num_samples: int,
    context: dict,
    x_min: Optional[float],
    x_max: Optional[float],
    save_triplets: bool,
    plot_steps: bool,
    max_plot: Optional[int] = None,
) -> None:
    if not plot_steps and not save_triplets:
        return

    max_samples = int(num_samples or 0)
    if max_samples <= 0:
        return

    count = 0
    did_plot = False

    with torch.inference_mode():
        for batch in dataloader:
            images = batch["images"].to(device)
            masks = batch["masks"].to(device)

            need_plot = plot_steps and not did_plot
            sol = sample_with_solver(
                model=model,
                x_init=torch.randn_like(images),
                solver_config=solver_config,
                masks=masks,
                return_intermediates=need_plot,
            )
            final_images = sol[-1] if sol.dim() == 5 else sol

            if save_triplets:
                for i in range(final_images.size(0)):
                    if count >= max_samples:
                        break
                    sample_dir = os.path.join(outdir, f"sample_{count + 1:03d}")
                    os.makedirs(sample_dir, exist_ok=True)

                    pred_x = final_images[i].detach().cpu().numpy().squeeze()
                    real_x = images[i].detach().cpu().numpy().squeeze()
                    gt_mask = masks[i].detach().cpu().numpy().squeeze()

                    pred_mask, _, _, _ = render_structure_mask_and_metrics(
                        context["rows"],
                        context["cols"],
                        pred_x,
                        context,
                        gt_mask.shape[-2],
                        gt_mask.shape[-1],
                        x_min=x_min,
                        x_max=x_max,
                    )
                    _save_grayscale(os.path.join(sample_dir, "mask.png"), gt_mask)
                    _save_rgb(
                        os.path.join(sample_dir, "overlay.png"),
                        mask_overlay_rgb(pred_mask, gt_mask),
                    )

                    _save_structure_image(
                        os.path.join(sample_dir, "gen.png"),
                        pred_x,
                        context,
                        mask_2d=gt_mask,
                        x_min=x_min,
                        x_max=x_max,
                    )
                    _save_structure_image(
                        os.path.join(sample_dir, "real.png"),
                        real_x,
                        context,
                        mask_2d=gt_mask,
                        x_min=x_min,
                        x_max=x_max,
                    )
                    _save_grayscale(os.path.join(sample_dir, "gen_mask.png"), pred_mask)
                    count += 1
            else:
                count += int(final_images.size(0))

            if need_plot:
                plot_solver_steps(
                    sol,
                    images,
                    masks,
                    context,
                    outdir,
                    x_min=x_min,
                    x_max=x_max,
                    max_plot=(4 if max_plot is None else int(max_plot)),
                )
                did_plot = True

            if count >= max_samples:
                break
            if not save_triplets and did_plot:
                break
