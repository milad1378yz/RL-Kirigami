import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/kirigami_x_mplconfig")

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from data_generator.utils import (
    center_points,
    compute_pose_points,
    compute_structure_points,
    solve_flat_points,
)


def draw_structure(ax, points, quads):
    for quad in quads:
        poly = points[np.asarray(quad, dtype=int)]
        ax.fill(poly[:, 0], poly[:, 1], facecolor="#d7e0e8", edgecolor="#22313a", linewidth=0.6)
    ax.set_aspect("equal")
    ax.axis("off")


def mask_extent(points, out_h, out_w):
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return [0.0, float(out_w), 0.0, float(out_h)]

    x = pts[:, 0]
    y = pts[:, 1]
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return [0.0, float(out_w), 0.0, float(out_h)]

    x = x[finite]
    y = y[finite]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    sx = (out_w - 1) / (xmax - xmin) if xmax > xmin else 1.0
    sy = (out_h - 1) / (ymax - ymin) if ymax > ymin else 1.0
    scale = max(1e-8, min(sx, sy))
    return [xmin, xmin + out_w / scale, ymax - out_h / scale, ymax]


def draw_structure_with_mask(
    ax,
    points,
    quads,
    mask_2d=None,
    overlay_alpha=0.35,
    dark_alpha=0.6,
):
    draw_structure(ax, points, quads)
    if mask_2d is None:
        return

    mask_2d = np.asarray(mask_2d, dtype=np.float32)
    extent = mask_extent(points, mask_2d.shape[-2], mask_2d.shape[-1])

    dark = np.zeros(mask_2d.shape + (4,), dtype=np.float32)
    dark[..., 3] = (1.0 - mask_2d) * float(dark_alpha)
    ax.imshow(dark, extent=extent, origin="upper", zorder=5)

    overlay = np.zeros(mask_2d.shape + (4,), dtype=np.float32)
    overlay[..., 1] = 1.0
    overlay[..., 3] = mask_2d * float(overlay_alpha)
    ax.imshow(overlay, extent=extent, origin="upper", zorder=6)


def plot_x_matrix_structure(
    ax,
    x_matrix,
    context,
    *,
    phi=0.0,
    mask_2d=None,
    x_min=None,
    x_max=None,
    normalize_phi=None,
):
    rows = context["rows"]
    cols = context["cols"]
    points, _, _ = compute_structure_points(
        rows,
        cols,
        x_matrix,
        context,
        phi=phi,
        x_min=x_min,
        x_max=x_max,
        normalize_phi=normalize_phi,
    )
    draw_structure_with_mask(ax, points, context["quads"], mask_2d=mask_2d)
    return points


def save_preview(path, samples, context, n_show):
    n_show = min(n_show, len(samples))
    if n_show == 0:
        return

    rows = context["rows"]
    cols = context["cols"]
    fig, axes = plt.subplots(n_show, 3, figsize=(8.6, 2.6 * n_show), squeeze=False)
    for i in range(n_show):
        sample = samples[i]
        x_matrix = sample["metadata"]["x_matrix"]
        flat_points = solve_flat_points(rows, cols, x_matrix, context["corners"], context["boundary_points"])
        rectangle_points = compute_pose_points(
            flat_points,
            context["linkages"],
            context["quads"],
            context["linkage_to_quads"],
            rows,
            cols,
            phi=np.pi,
        )
        deployed_points = compute_pose_points(
            flat_points,
            context["linkages"],
            context["quads"],
            context["linkage_to_quads"],
            rows,
            cols,
            phi=0.0,
        )
        rectangle_points = center_points(rectangle_points, phi=np.pi)
        deployed_points = center_points(deployed_points)

        all_points = np.vstack([rectangle_points, deployed_points])
        pad = 0.05 * max(np.ptp(all_points[:, 0]), np.ptp(all_points[:, 1]))
        xlim = (all_points[:, 0].min() - pad, all_points[:, 0].max() + pad)
        ylim = (all_points[:, 1].min() - pad, all_points[:, 1].max() + pad)

        draw_structure(axes[i, 0], rectangle_points, context["quads"])
        axes[i, 0].set_xlim(*xlim)
        axes[i, 0].set_ylim(*ylim)
        axes[i, 0].set_title("rectangle", fontsize=9)

        draw_structure(axes[i, 1], deployed_points, context["quads"])
        axes[i, 1].set_xlim(*xlim)
        axes[i, 1].set_ylim(*ylim)
        axes[i, 1].set_title("deployed", fontsize=9)

        axes[i, 2].imshow(sample["mask"][0], cmap="gray")
        axes[i, 2].set_title("mask", fontsize=9)
        axes[i, 2].axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def render_frame(points, quads, phi, xlim, ylim):
    fig, ax = plt.subplots(figsize=(5, 5))
    draw_structure(ax, points, quads)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(f"phi = {phi:.2f}", fontsize=10)
    fig.tight_layout(pad=0.05)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return image


def save_gif(path, x_matrix, context, n_frames, duration):
    rows = context["rows"]
    cols = context["cols"]
    flat_points = solve_flat_points(rows, cols, x_matrix, context["corners"], context["boundary_points"])
    phis = np.linspace(np.pi, 0.0, n_frames)
    frames_points = []
    for phi in phis:
        points = compute_pose_points(
            flat_points,
            context["linkages"],
            context["quads"],
            context["linkage_to_quads"],
            rows,
            cols,
            phi=phi,
        )
        frames_points.append(center_points(points, phi=phi))

    all_points = np.vstack(frames_points)
    pad = 0.05 * max(np.ptp(all_points[:, 0]), np.ptp(all_points[:, 1]))
    xlim = (all_points[:, 0].min() - pad, all_points[:, 0].max() + pad)
    ylim = (all_points[:, 1].min() - pad, all_points[:, 1].max() + pad)

    frames = [
        render_frame(points, context["quads"], phi, xlim, ylim)
        for points, phi in zip(frames_points, phis)
    ]
    imageio.mimsave(path, frames, duration=duration, loop=0)


def save_gifs(out_dir, samples, context, n_gifs, n_frames, duration):
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for i, sample in enumerate(samples[: min(n_gifs, len(samples))]):
        gif_path = os.path.join(out_dir, f"sample_{i:02d}.gif")
        save_gif(gif_path, sample["metadata"]["x_matrix"], context, n_frames, duration)
        saved.append(gif_path)
    return saved
