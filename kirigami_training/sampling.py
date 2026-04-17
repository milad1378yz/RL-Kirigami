import os
from typing import Any, Optional

from flow_matching.solver import ODESolver
import matplotlib.pyplot as plt
import numpy as np
import torch

from data_generator.utils import (
    mask_iou,
    mask_overlay_rgb,
    mask_siou,
    render_structure_mask_and_metrics,
)
from data_generator.visualization import plot_x_matrix_structure


def sample_with_solver(
    model: torch.nn.Module,
    x_init: torch.Tensor,
    solver_config: dict[str, Any],
    masks: torch.Tensor,
    *,
    return_intermediates: bool = True,
) -> torch.Tensor:
    solver = ODESolver(velocity_model=model)
    time_points = int(solver_config.get("time_points", 10))
    time_grid = torch.linspace(0, 1, time_points, device=x_init.device, dtype=x_init.dtype)

    return solver.sample(
        time_grid=time_grid,
        x_init=x_init,
        method=solver_config.get("method", "midpoint"),
        step_size=solver_config.get("step_size", 0.02),
        return_intermediates=bool(return_intermediates),
        masks=masks,
    )


def _plot_invalid(ax, message: str) -> None:
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center")


def plot_solver_steps(
    sol: torch.Tensor,
    x_batch: torch.Tensor,
    mask_batch: torch.Tensor,
    context: dict,
    outdir: str,
    *,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    max_plot: int = 4,
) -> None:
    if sol.dim() != 5:
        return

    n_samples = min(int(sol.shape[1]), int(max_plot))
    total_cols = 7

    fig, axes = plt.subplots(
        n_samples,
        total_cols,
        figsize=(2.7 * total_cols, max(4, 3 * n_samples)),
        squeeze=False,
    )

    for i in range(n_samples):
        gt_x = x_batch[i].detach().cpu().numpy().squeeze()
        gt_mask = mask_batch[i].detach().cpu().numpy().squeeze()
        pred_final = sol[-1, i].detach().cpu().numpy().squeeze()

        pred_mask, _, _, _ = render_structure_mask_and_metrics(
            context["rows"],
            context["cols"],
            pred_final,
            context,
            gt_mask.shape[-2],
            gt_mask.shape[-1],
            x_min=x_min,
            x_max=x_max,
        )
        iou = mask_iou(pred_mask, gt_mask)
        siou, aligned_gt_mask, _ = mask_siou(pred_mask, gt_mask, return_alignment=True)
        overlay = mask_overlay_rgb(pred_mask, aligned_gt_mask.astype(np.float32))

        col = 0
        try:
            plot_x_matrix_structure(
                axes[i, col],
                pred_final,
                context,
                mask_2d=pred_mask,
                x_min=x_min,
                x_max=x_max,
                normalize_phi=None,
            )
        except Exception as exc:
            _plot_invalid(axes[i, col], f"invalid\n{exc}")
        if i == 0:
            axes[i, col].set_title("Final Step + Mask")
        col += 1

        heatmap_artist = axes[i, col].imshow(
            pred_final,
            cmap="viridis",
            vmin=x_min,
            vmax=x_max,
            interpolation="nearest",
        )
        axes[i, col].set_xticks(np.arange(pred_final.shape[1]))
        axes[i, col].set_yticks(np.arange(pred_final.shape[0]))
        axes[i, col].set_xticks(np.arange(-0.5, pred_final.shape[1], 1.0), minor=True)
        axes[i, col].set_yticks(np.arange(-0.5, pred_final.shape[0], 1.0), minor=True)
        axes[i, col].grid(which="minor", color="white", linewidth=0.3, alpha=0.35)
        axes[i, col].tick_params(which="minor", bottom=False, left=False)
        axes[i, col].tick_params(labelsize=6)
        for row in range(pred_final.shape[0]):
            for col_idx in range(pred_final.shape[1]):
                value = pred_final[row, col_idx]
                if not np.isfinite(value):
                    continue
                color = "white" if float(heatmap_artist.norm(value)) < 0.55 else "black"
                axes[i, col].text(
                    col_idx,
                    row,
                    f"{value:.2g}",
                    ha="center",
                    va="center",
                    fontsize=4.0,
                    color=color,
                )
        if i == 0:
            axes[i, col].set_title("Gen x_ij")
        col += 1

        axes[i, col].imshow(pred_mask, cmap="gray", vmin=0.0, vmax=1.0)
        axes[i, col].axis("off")
        if i == 0:
            axes[i, col].set_title("Gen Mask")
        col += 1

        axes[i, col].imshow(gt_mask, cmap="gray", vmin=0.0, vmax=1.0)
        axes[i, col].axis("off")
        if i == 0:
            axes[i, col].set_title("GT Mask")
        col += 1

        axes[i, col].imshow(overlay)
        axes[i, col].axis("off")
        axes[i, col].set_title(f"IoU {iou:.3f}\nSIoU {siou:.3f}")
        col += 1

        try:
            plot_x_matrix_structure(
                axes[i, col],
                pred_final,
                context,
                mask_2d=gt_mask,
                x_min=x_min,
                x_max=x_max,
                normalize_phi=None,
            )
        except Exception as exc:
            _plot_invalid(axes[i, col], f"invalid\n{exc}")
        if i == 0:
            axes[i, col].set_title("Gen + GT Mask")
        col += 1

        try:
            plot_x_matrix_structure(
                axes[i, col],
                gt_x,
                context,
                mask_2d=gt_mask,
                x_min=x_min,
                x_max=x_max,
                normalize_phi=None,
            )
        except Exception as exc:
            _plot_invalid(axes[i, col], f"invalid\n{exc}")
        if i == 0:
            axes[i, col].set_title("Real")

    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, "solver_steps.png"), bbox_inches="tight", pad_inches=0)
    plt.close(fig)
