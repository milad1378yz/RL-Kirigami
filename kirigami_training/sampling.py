import os
from typing import Any, Optional

from flow_matching.solver import ODESolver
import matplotlib.pyplot as plt
import numpy as np
import torch

from data_generator.utils import mask_dice, mask_iou, mask_overlay_rgb, x_matrix_to_mask_and_metrics
from data_generator.visualization import plot_x_matrix_structure


def sample_with_solver(
    model: torch.nn.Module,
    x_init: torch.Tensor,
    solver_config: dict[str, Any],
    masks: Optional[torch.Tensor] = None,
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


def _step_indices(n_steps: int) -> list[int]:
    if n_steps <= 5:
        return list(range(n_steps))
    picks = np.linspace(0, n_steps - 1, num=5, dtype=int).tolist()
    picks[-1] = n_steps - 1
    picks = sorted(set(picks))
    while len(picks) < 5:
        for idx in range(n_steps - 1):
            if idx not in picks:
                picks.insert(-1, idx)
            if len(picks) >= 5:
                break
    return picks


def _plot_invalid(ax, message: str) -> None:
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center")


def plot_solver_steps(
    sol: torch.Tensor,
    x_batch: torch.Tensor,
    mask_batch: Optional[torch.Tensor],
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
    n_steps = int(sol.shape[0])
    step_idx = _step_indices(n_steps)
    extra_cols = 1 if mask_batch is None else 5

    fig, axes = plt.subplots(
        n_samples,
        len(step_idx) + extra_cols,
        figsize=(2.5 * (len(step_idx) + extra_cols), max(4, 3 * n_samples)),
        squeeze=False,
    )

    for i in range(n_samples):
        gt_x = x_batch[i].detach().cpu().numpy().squeeze()
        gt_mask = mask_batch[i].detach().cpu().numpy().squeeze() if mask_batch is not None else None
        pred_final = sol[-1, i].detach().cpu().numpy().squeeze()

        pred_mask = None
        overlay = None
        iou = None
        dice = None
        if gt_mask is not None:
            pred_mask, _, _, _ = x_matrix_to_mask_and_metrics(
                context["rows"],
                context["cols"],
                pred_final,
                context,
                gt_mask.shape[-2],
                gt_mask.shape[-1],
                x_min=x_min,
                x_max=x_max,
            )
            overlay = mask_overlay_rgb(pred_mask, gt_mask)
            iou = mask_iou(pred_mask, gt_mask)
            dice = mask_dice(pred_mask, gt_mask)

        for col, step in enumerate(step_idx):
            ax = axes[i, col]
            step_x = sol[step, i].detach().cpu().numpy().squeeze()
            step_mask = pred_mask if (gt_mask is not None and step == n_steps - 1) else None
            try:
                plot_x_matrix_structure(
                    ax,
                    step_x,
                    context,
                    mask_2d=step_mask,
                    x_min=x_min,
                    x_max=x_max,
                    normalize_phi=None,
                )
            except Exception as exc:
                _plot_invalid(ax, f"invalid\n{exc}")
            if i == 0:
                title = f"Step {step}"
                if step == n_steps - 1 and step_mask is not None:
                    title += " + mask"
                ax.set_title(title)

        col = len(step_idx)
        if gt_mask is not None:
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
            axes[i, col].set_title(f"IoU {iou:.3f}\nDice {dice:.3f}")
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

