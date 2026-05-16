"""Checkpoint evaluation for target silhouette masks.

Example:
    python -m kirigami_training.evaluation \
        --config_path configs/training.yaml \
        --training_key fm_training \
        --checkpoint last \
        --masks path/to/masks \
        --outdir eval/fm_last \
        --num_candidates 16
"""
import argparse
from collections import Counter
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/kirigami_x_mplconfig")

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from data_generator.utils import (
    build_geometry_context,
    dataset_entry_filter_reason,
    mask_hole_metrics,
    mask_iou,
    mask_overlay_rgb,
    mask_siou,
    render_structure_mask_and_metrics,
)
from data_generator.visualization import plot_x_matrix_structure

from .data import model_to_x_space, prepare_training_config
from .model import build_model
from .sampling import sample_with_solver
from .utils import load_config, resolve_checkpoint_path, select_training_config


MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy", ".npz"}
TRAINING_KEYS = ("fm_training", "rl_training", "distill_training")

CANDIDATE_FIELDS = [
    "target_index",
    "target_name",
    "target_path",
    "candidate_index",
    "selected",
    "siou",
    "iou",
    "build_success",
    "build_failure",
    "invalid_quad_count",
    "overlap_ratio",
    "fill_error",
    "target_fill_ratio",
    "pred_fill_ratio",
    "clipped_fraction",
    "range_violation_l1",
    "range_violation_max",
    "out_of_range_count",
    "ratio_min",
    "ratio_max",
    "pred_hole_count",
    "pred_hole_fraction",
    "target_hole_count",
    "target_hole_fraction",
    "failure_category",
    "error",
]

BEST_FIELDS = [
    "target_index",
    "target_name",
    "target_path",
    "best_candidate_index",
    "num_candidates",
    "selection_metric",
    "sample_dir",
] + CANDIDATE_FIELDS[5:]


@dataclass(frozen=True)
class TargetMask:
    index: int
    name: str
    path: str
    mask: np.ndarray


def _safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return name or "mask"


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def _write_records_csv(path: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _normalize_mask_array(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = arr[..., :3].mean(axis=-1)

    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        raise ValueError("empty mask array")

    finite = np.isfinite(arr)
    if not bool(finite.all()):
        arr = np.where(finite, arr, 0.0)

    max_value = float(arr.max()) if arr.size else 0.0
    if max_value > 1.0:
        arr = arr / (255.0 if max_value <= 255.0 else max_value)
    return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)


def _split_mask_array(array: np.ndarray) -> list[np.ndarray]:
    arr = np.asarray(array)
    if arr.ndim == 2:
        return [_normalize_mask_array(arr)]
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        return [_normalize_mask_array(arr)]
    if arr.ndim == 3 and arr.shape[0] == 1:
        return [_normalize_mask_array(arr[0])]
    if arr.ndim == 3:
        return [_normalize_mask_array(arr[i]) for i in range(arr.shape[0])]
    if arr.ndim == 4 and arr.shape[1] == 1:
        return [_normalize_mask_array(arr[i, 0]) for i in range(arr.shape[0])]
    if arr.ndim == 4 and arr.shape[-1] in (3, 4):
        return [_normalize_mask_array(arr[i]) for i in range(arr.shape[0])]
    raise ValueError(f"unsupported mask array shape {arr.shape}")


def _read_mask_file(path: str) -> list[np.ndarray]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return _split_mask_array(np.load(path))
    if ext == ".npz":
        package = np.load(path)
        if not package.files:
            raise ValueError(f"no arrays found in {path}")
        return _split_mask_array(package[package.files[0]])
    return _split_mask_array(iio.imread(path))


def _resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if tuple(mask.shape[-2:]) == tuple(size):
        return mask.astype(np.float32, copy=False)
    tensor = torch.as_tensor(mask, dtype=torch.float32).view(1, 1, *mask.shape[-2:])
    resized = F.interpolate(tensor, size=tuple(size), mode="nearest")
    return resized[0, 0].cpu().numpy().astype(np.float32, copy=False)


def load_target_masks(
    mask_path: str,
    *,
    mask_size: tuple[int, int],
    limit: Optional[int] = None,
    invert: bool = False,
) -> list[TargetMask]:
    """Load masks from an image/array file or every supported file in a directory."""
    root = os.path.expanduser(mask_path)
    if os.path.isdir(root):
        files = [
            os.path.join(root, name)
            for name in sorted(os.listdir(root))
            if os.path.splitext(name)[1].lower() in MASK_EXTENSIONS
        ]
    elif os.path.isfile(root):
        files = [root]
    else:
        raise FileNotFoundError(f"Mask path does not exist: {mask_path}")

    targets: list[TargetMask] = []
    for file_path in files:
        arrays = _read_mask_file(file_path)
        for arr_index, array in enumerate(arrays):
            mask = _resize_mask(array, mask_size)
            if invert:
                mask = 1.0 - mask
            stem = os.path.splitext(os.path.basename(file_path))[0]
            name = stem if len(arrays) == 1 else f"{stem}_{arr_index:04d}"
            targets.append(
                TargetMask(
                    index=len(targets),
                    name=name,
                    path=file_path,
                    mask=np.clip(mask, 0.0, 1.0).astype(np.float32, copy=False),
                )
            )
            if limit is not None and len(targets) >= int(limit):
                return targets

    if not targets:
        raise ValueError(f"No masks found in {mask_path}")
    return targets


def _default_run_name(config: dict, config_path: str, training_key: str) -> str:
    base = config.get("run_name", os.path.splitext(os.path.basename(config_path))[0])
    if training_key == "rl_training":
        return f"{base}_RL"
    if training_key == "distill_training":
        return f"{base}_distill"
    return str(base)


def _resolve_eval_checkpoint(
    *,
    root_ckpt_dir: str,
    run_name: str,
    checkpoint: str,
) -> str:
    checkpoint = str(checkpoint or "last").strip()
    lowered = checkpoint.lower()
    if lowered in {"", "none"}:
        raise ValueError("Evaluation requires a checkpoint path or checkpoint name.")

    direct = os.path.expanduser(checkpoint)
    if os.path.isfile(direct):
        return direct

    if lowered == "last":
        resolved = resolve_checkpoint_path(root_ckpt_dir, run_name, "last")
        if resolved is not None:
            return resolved
        raise FileNotFoundError(
            f"Could not find a last checkpoint under {root_ckpt_dir}/{run_name}."
        )

    run_dir = os.path.join(os.path.expanduser(root_ckpt_dir), run_name)
    named = os.path.join(run_dir, checkpoint)
    if os.path.isfile(named):
        return named
    if not named.endswith(".ckpt") and os.path.isfile(named + ".ckpt"):
        return named + ".ckpt"

    if lowered == "best":
        candidates = [
            os.path.join(run_dir, name)
            for name in os.listdir(run_dir)
            if name.endswith(".ckpt") and not name.startswith("last")
        ] if os.path.isdir(run_dir) else []
        if candidates:
            return max(candidates, key=os.path.getmtime)

    raise FileNotFoundError(
        f"Could not resolve checkpoint '{checkpoint}' as a file or under {run_dir}."
    )


def load_checkpoint_weights(model: torch.nn.Module, checkpoint_path: str) -> None:
    ckpt = torch.load(os.path.expanduser(checkpoint_path), map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint has no state dict: {checkpoint_path}")

    model_state = model.state_dict()
    model_keys = set(model_state)
    if any(str(key).startswith("model.") for key in state):
        candidate = {
            str(key).split("model.", 1)[1]: value
            for key, value in state.items()
            if str(key).startswith("model.")
        }
    else:
        candidate = {str(key).removeprefix("module."): value for key, value in state.items()}

    matched = sum(
        key in model_keys and tuple(value.shape) == tuple(model_state[key].shape)
        for key, value in candidate.items()
        if torch.is_tensor(value)
    )
    if matched == 0:
        raise ValueError(
            f"No model weights in checkpoint matched the current model: {checkpoint_path}"
        )

    missing, unexpected = model.load_state_dict(candidate, strict=False)
    print(
        f"Loaded checkpoint {checkpoint_path} "
        f"(matched={matched}, missing={len(missing)}, unexpected={len(unexpected)})."
    )


def _failure_category(
    metrics: dict[str, Any],
    *,
    filters: Optional[dict],
    out_of_range_count: int,
) -> str:
    reject_reason = dataset_entry_filter_reason(metrics, filters=filters)
    if reject_reason is not None:
        return str(reject_reason)
    if int(out_of_range_count) > 0:
        return "out_of_range"
    return "ok"


def _candidate_metrics(
    *,
    target: TargetMask,
    target_holes: dict[str, Any],
    x_matrix: np.ndarray,
    context: dict,
    x_min: float,
    x_max: float,
    threshold: float,
    filters: Optional[dict],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    rows = int(context["rows"])
    cols = int(context["cols"])
    target_mask = target.mask

    pred_mask, geom, _, clipped = render_structure_mask_and_metrics(
        rows,
        cols,
        x_matrix,
        context,
        int(target_mask.shape[0]),
        int(target_mask.shape[1]),
        x_min=x_min,
        x_max=x_max,
    )

    out_of_range = np.logical_or(x_matrix < float(x_min), x_matrix > float(x_max))
    out_of_range_count = int(out_of_range.sum())
    fill_error = abs(float(geom.get("fill_ratio", 0.0)) - float(target_mask.mean()))

    row = {
        "siou": float(mask_siou(pred_mask, target_mask, threshold=threshold)),
        "iou": float(mask_iou(pred_mask, target_mask, threshold=threshold)),
        "build_success": bool(geom.get("ok", False)),
        "build_failure": not bool(geom.get("ok", False)),
        "invalid_quad_count": int(geom.get("invalid_quad_count", 0) or 0),
        "overlap_ratio": float(geom.get("overlap_ratio", 0.0) or 0.0),
        "fill_error": float(fill_error),
        "target_fill_ratio": float(target_mask.mean()),
        "pred_fill_ratio": float(geom.get("fill_ratio", 0.0) or 0.0),
        "clipped_fraction": float(geom.get("clipped_fraction", 0.0) or 0.0),
        "range_violation_l1": float(geom.get("range_violation_l1", 0.0) or 0.0),
        "range_violation_max": float(geom.get("range_violation_max", 0.0) or 0.0),
        "out_of_range_count": out_of_range_count,
        "ratio_min": float(np.min(x_matrix)),
        "ratio_max": float(np.max(x_matrix)),
        "pred_hole_count": int(geom.get("hole_count", 0) or 0),
        "pred_hole_fraction": float(geom.get("hole_fraction", 0.0) or 0.0),
        "target_hole_count": int(target_holes.get("hole_count", 0) or 0),
        "target_hole_fraction": float(target_holes.get("hole_fraction", 0.0) or 0.0),
        "error": geom.get("error") or "",
    }
    row["failure_category"] = _failure_category(
        geom,
        filters=filters,
        out_of_range_count=out_of_range_count,
    )
    return row, pred_mask, clipped


def _save_grayscale(path: str, image: np.ndarray) -> None:
    plt.imsave(path, np.asarray(image, dtype=np.float32), cmap="gray", vmin=0.0, vmax=1.0)


def _save_rgb(path: str, image: np.ndarray) -> None:
    plt.imsave(path, np.asarray(image, dtype=np.float32))


def _save_ratio_heatmap(path: str, x_matrix: np.ndarray, *, x_min: float, x_max: float) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(x_matrix, cmap="viridis", vmin=x_min, vmax=x_max, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(pad=0.05)
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _save_structure_image(
    path: str,
    x_matrix: np.ndarray,
    context: dict,
    *,
    target_mask: np.ndarray,
    x_min: float,
    x_max: float,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    try:
        plot_x_matrix_structure(
            ax,
            x_matrix,
            context,
            mask_2d=target_mask,
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


def _save_sample_visuals(
    *,
    outdir: str,
    target: TargetMask,
    best_x: np.ndarray,
    best_clipped_x: np.ndarray,
    best_pred_mask: np.ndarray,
    context: dict,
    x_min: float,
    x_max: float,
    threshold: float,
) -> str:
    sample_dir = os.path.join(outdir, "samples", f"{target.index:04d}_{_safe_name(target.name)}")
    os.makedirs(sample_dir, exist_ok=True)

    _save_grayscale(os.path.join(sample_dir, "target_mask.png"), target.mask)
    _save_grayscale(os.path.join(sample_dir, "pred_mask.png"), best_pred_mask)
    _save_rgb(
        os.path.join(sample_dir, "overlay.png"),
        mask_overlay_rgb(best_pred_mask, target.mask, threshold=threshold),
    )
    _save_ratio_heatmap(
        os.path.join(sample_dir, "ratio_field.png"),
        best_x,
        x_min=x_min,
        x_max=x_max,
    )
    _save_structure_image(
        os.path.join(sample_dir, "structure.png"),
        best_clipped_x,
        context,
        target_mask=target.mask,
        x_min=x_min,
        x_max=x_max,
    )
    np.save(os.path.join(sample_dir, "ratio_field.npy"), best_x.astype(np.float32))
    np.save(os.path.join(sample_dir, "ratio_field_clipped.npy"), best_clipped_x.astype(np.float32))
    np.save(os.path.join(sample_dir, "pred_mask.npy"), best_pred_mask.astype(np.float32))
    return sample_dir


def _save_candidate_visuals(
    *,
    sample_dir: str,
    candidate_index: int,
    target_mask: np.ndarray,
    x_matrix: np.ndarray,
    pred_mask: np.ndarray,
    x_min: float,
    x_max: float,
    threshold: float,
) -> None:
    candidate_dir = os.path.join(sample_dir, "candidates", f"candidate_{candidate_index:03d}")
    os.makedirs(candidate_dir, exist_ok=True)
    _save_grayscale(os.path.join(candidate_dir, "pred_mask.png"), pred_mask)
    _save_rgb(
        os.path.join(candidate_dir, "overlay.png"),
        mask_overlay_rgb(pred_mask, target_mask, threshold=threshold),
    )
    _save_ratio_heatmap(
        os.path.join(candidate_dir, "ratio_field.png"),
        x_matrix,
        x_min=x_min,
        x_max=x_max,
    )
    np.save(os.path.join(candidate_dir, "ratio_field.npy"), x_matrix.astype(np.float32))


def _aggregate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    row_list = list(rows)
    if not row_list:
        return {}

    numeric_fields = [
        "siou",
        "iou",
        "build_success",
        "build_failure",
        "invalid_quad_count",
        "overlap_ratio",
        "fill_error",
        "clipped_fraction",
        "range_violation_l1",
        "range_violation_max",
        "out_of_range_count",
        "pred_hole_count",
        "target_hole_count",
    ]
    out: dict[str, Any] = {"count": len(row_list)}
    for field in numeric_fields:
        values = np.asarray(
            [float(row.get(field, 0.0) or 0.0) for row in row_list],
            dtype=np.float64,
        )
        out[f"{field}_mean"] = float(values.mean())
        out[f"{field}_min"] = float(values.min())
        out[f"{field}_max"] = float(values.max())
    out["failure_category_counts"] = dict(
        Counter(str(row.get("failure_category", "")) for row in row_list)
    )
    return out


def evaluate_checkpoint(
    *,
    config: dict,
    checkpoint_path: str,
    mask_path: str,
    outdir: str,
    num_candidates: int,
    batch_size: int,
    device: torch.device,
    threshold: float,
    selection_metric: str,
    source_noise_std: float,
    limit: Optional[int],
    invert_masks: bool,
    save_visuals: bool,
    save_all_candidates: bool,
) -> dict[str, Any]:
    data_cfg = config["data"]
    model_cfg = config["model_config"]
    tr = config["training"]

    os.makedirs(outdir, exist_ok=True)
    mask_size = tuple(int(v) for v in model_cfg["mask_size"])
    targets = load_target_masks(mask_path, mask_size=mask_size, limit=limit, invert=invert_masks)
    print(f"Loaded {len(targets)} target masks from {mask_path}.")

    context = build_geometry_context(int(data_cfg["grid_rows"]), int(data_cfg["grid_cols"]))
    x_min = float(data_cfg["x_min"])
    x_max = float(data_cfg["x_max"])
    filters = data_cfg.get("filters")

    model = build_model(config, device=device).to(device)
    load_checkpoint_weights(model, checkpoint_path)
    model.eval()

    solver_config = {
        "method": tr.get("method", "midpoint"),
        "step_size": tr.get("step_size", 0.02),
        "time_points": tr.get("time_points", 10),
        "source_noise_std": source_noise_std,
    }

    channels = int(model_cfg.get("in_channels", 1))
    input_size = tuple(int(v) for v in model_cfg["input_size"])
    candidate_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    selection_metric = selection_metric.lower()
    if selection_metric not in {"siou", "iou"}:
        raise ValueError("selection_metric must be 'siou' or 'iou'.")

    for start in range(0, len(targets), int(batch_size)):
        batch_targets = targets[start : start + int(batch_size)]
        batch_masks_np = np.stack([target.mask[None, :, :] for target in batch_targets], axis=0)
        batch_masks = torch.as_tensor(batch_masks_np, dtype=torch.float32, device=device)
        repeated_masks = batch_masks.repeat_interleave(int(num_candidates), dim=0)
        x_init = source_noise_std * torch.randn(
            repeated_masks.shape[0],
            channels,
            int(input_size[0]),
            int(input_size[1]),
            device=device,
            dtype=torch.float32,
        )

        with torch.inference_mode():
            pred_z = sample_with_solver(
                model=model,
                x_init=x_init,
                solver_config=solver_config,
                masks=repeated_masks,
                return_intermediates=False,
            )
            pred_x = model_to_x_space(pred_z, x_min=x_min, x_max=x_max, clip=False)

        pred_x_np = pred_x.detach().cpu().numpy()
        if pred_x_np.ndim != 4 or pred_x_np.shape[1] != 1:
            raise ValueError(
                f"Expected generated ratios with shape [N,1,H,W], got {pred_x_np.shape}."
            )

        for local_idx, target in enumerate(batch_targets):
            target_holes = mask_hole_metrics(target.mask, threshold=threshold)
            local_records: list[dict[str, Any]] = []
            local_masks: list[np.ndarray] = []
            local_clipped: list[np.ndarray] = []
            local_x: list[np.ndarray] = []

            for candidate_idx in range(int(num_candidates)):
                flat_idx = local_idx * int(num_candidates) + candidate_idx
                x_matrix = pred_x_np[flat_idx, 0].astype(np.float32, copy=False)
                metrics, pred_mask, clipped = _candidate_metrics(
                    target=target,
                    target_holes=target_holes,
                    x_matrix=x_matrix,
                    context=context,
                    x_min=x_min,
                    x_max=x_max,
                    threshold=threshold,
                    filters=filters,
                )
                record = {
                    "target_index": int(target.index),
                    "target_name": target.name,
                    "target_path": target.path,
                    "candidate_index": int(candidate_idx),
                    "selected": False,
                    **metrics,
                }
                local_records.append(record)
                local_masks.append(pred_mask)
                local_clipped.append(clipped)
                local_x.append(x_matrix)

            scores = [float(row[selection_metric]) for row in local_records]
            best_local_idx = int(np.argmax(scores))
            local_records[best_local_idx]["selected"] = True
            candidate_rows.extend(local_records)

            best_record = local_records[best_local_idx]
            sample_dir = ""
            if save_visuals:
                sample_dir = _save_sample_visuals(
                    outdir=outdir,
                    target=target,
                    best_x=local_x[best_local_idx],
                    best_clipped_x=local_clipped[best_local_idx],
                    best_pred_mask=local_masks[best_local_idx],
                    context=context,
                    x_min=x_min,
                    x_max=x_max,
                    threshold=threshold,
                )
                if save_all_candidates:
                    for candidate_idx, (x_matrix, pred_mask) in enumerate(
                        zip(local_x, local_masks)
                    ):
                        _save_candidate_visuals(
                            sample_dir=sample_dir,
                            candidate_index=candidate_idx,
                            target_mask=target.mask,
                            x_matrix=x_matrix,
                            pred_mask=pred_mask,
                            x_min=x_min,
                            x_max=x_max,
                            threshold=threshold,
                        )

            best_rows.append(
                {
                    "target_index": int(target.index),
                    "target_name": target.name,
                    "target_path": target.path,
                    "best_candidate_index": int(best_record["candidate_index"]),
                    "num_candidates": int(num_candidates),
                    "selection_metric": selection_metric,
                    "sample_dir": sample_dir,
                    **{field: best_record.get(field, "") for field in CANDIDATE_FIELDS[5:]},
                }
            )

        print(
            f"Evaluated targets {start + 1}-{start + len(batch_targets)} "
            f"of {len(targets)} with K={int(num_candidates)}."
        )

    per_candidate_csv = os.path.join(outdir, "per_candidate_results.csv")
    best_csv = os.path.join(outdir, "best_of_k_results.csv")
    per_candidate_json = os.path.join(outdir, "per_candidate_results.json")
    best_json = os.path.join(outdir, "best_of_k_results.json")

    _write_records_csv(per_candidate_csv, candidate_rows, CANDIDATE_FIELDS)
    _write_records_csv(best_csv, best_rows, BEST_FIELDS)
    _write_json(per_candidate_json, candidate_rows)
    _write_json(best_json, best_rows)

    summary = {
        "checkpoint_path": checkpoint_path,
        "mask_path": mask_path,
        "output_dir": outdir,
        "num_targets": len(targets),
        "num_candidates_per_target": int(num_candidates),
        "selection_metric": selection_metric,
        "threshold": float(threshold),
        "source_noise_std": float(source_noise_std),
        "solver_config": solver_config,
        "all_candidates": _aggregate(candidate_rows),
        "best_of_k": _aggregate(best_rows),
        "files": {
            "per_candidate_csv": per_candidate_csv,
            "per_candidate_json": per_candidate_json,
            "best_of_k_csv": best_csv,
            "best_of_k_json": best_json,
        },
    }
    _write_json(os.path.join(outdir, "summary.json"), summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a kirigami checkpoint on a folder of masks."
    )
    parser.add_argument("--config_path", type=str, default="configs/training.yaml")
    parser.add_argument("--training_key", type=str, default="fm_training", choices=TRAINING_KEYS)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path, 'last', 'best', or name under the run directory.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Override the run directory name used for named checkpoints.",
    )
    parser.add_argument(
        "--masks",
        "--mask_dir",
        dest="masks",
        type=str,
        default=None,
        help="Mask image/array file or directory.",
    )
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--num_candidates", "-K", type=int, default=None)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Number of target masks per sampling batch.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--selection_metric", type=str, default=None, choices=("siou", "iou"))
    parser.add_argument("--source_noise_std", type=float, default=None)
    parser.add_argument("--invert_masks", action="store_true")
    parser.add_argument("--no_visuals", action="store_true")
    parser.add_argument("--save_all_candidates", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config_path)
    config = select_training_config(config, args.training_key)
    config = prepare_training_config(config)

    eval_cfg = config.get("evaluation", {}) or {}
    tr = config["training"]
    root_ckpt_dir = os.path.expanduser(str(tr.get("checkpoint_dir", "checkpoints")))
    run_name = args.run_name or str(
        eval_cfg.get("run_name")
        or _default_run_name(config, args.config_path, args.training_key)
    )
    checkpoint_name = (
        args.checkpoint
        or eval_cfg.get("checkpoint")
        or eval_cfg.get("checkpoint_path")
        or "last"
    )
    checkpoint_path = _resolve_eval_checkpoint(
        root_ckpt_dir=root_ckpt_dir,
        run_name=run_name,
        checkpoint=checkpoint_name,
    )

    mask_path = (
        args.masks
        or eval_cfg.get("masks")
        or eval_cfg.get("mask_dir")
        or eval_cfg.get("mask_path")
    )
    if not mask_path:
        raise ValueError("Pass --masks or set evaluation.masks in the config.")

    num_candidates = int(
        args.num_candidates
        or eval_cfg.get("num_candidates")
        or eval_cfg.get("k")
        or 1
    )
    if num_candidates <= 0:
        raise ValueError("--num_candidates must be positive.")

    batch_size = int(args.batch_size or eval_cfg.get("batch_size") or 1)
    if batch_size <= 0:
        raise ValueError("--batch_size must be positive.")

    seed = int(args.seed if args.seed is not None else eval_cfg.get("seed", tr.get("seed", 0)))
    torch.manual_seed(seed)
    np.random.seed(seed)

    device_arg = args.device or eval_cfg.get("device") or "auto"
    if str(device_arg).lower() == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(str(device_arg))

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    checkpoint_slug = _safe_name(os.path.splitext(os.path.basename(checkpoint_path))[0])
    outdir = args.outdir or eval_cfg.get("outdir") or eval_cfg.get("output_dir")
    if not outdir:
        outdir = os.path.join("evaluation_results", f"{run_name}_{checkpoint_slug}_{timestamp}")
    outdir = os.path.expanduser(str(outdir))

    threshold = float(
        args.threshold
        if args.threshold is not None
        else eval_cfg.get("threshold", tr.get("mask_threshold", 0.5))
    )
    selection_metric = str(
        args.selection_metric or eval_cfg.get("selection_metric", "siou")
    ).lower()
    source_noise_std = float(
        args.source_noise_std
        if args.source_noise_std is not None
        else eval_cfg.get("source_noise_std", tr.get("source_noise_std", 0.5))
    )
    invert_masks = bool(args.invert_masks or _as_bool(eval_cfg.get("invert_masks"), default=False))
    save_visuals = not bool(args.no_visuals or _as_bool(eval_cfg.get("no_visuals"), default=False))
    save_all_candidates = bool(
        args.save_all_candidates
        or _as_bool(eval_cfg.get("save_all_candidates"), default=False)
    )

    summary = evaluate_checkpoint(
        config=config,
        checkpoint_path=checkpoint_path,
        mask_path=str(mask_path),
        outdir=outdir,
        num_candidates=num_candidates,
        batch_size=batch_size,
        device=device,
        threshold=threshold,
        selection_metric=selection_metric,
        source_noise_std=source_noise_std,
        limit=args.limit if args.limit is not None else eval_cfg.get("limit"),
        invert_masks=invert_masks,
        save_visuals=save_visuals,
        save_all_candidates=save_all_candidates,
    )

    best = summary.get("best_of_k", {})
    print(
        "Evaluation complete: "
        f"mean best SIoU={best.get('siou_mean', float('nan')):.4f}, "
        f"mean best IoU={best.get('iou_mean', float('nan')):.4f}. "
        f"Results: {summary['output_dir']}"
    )


if __name__ == "__main__":
    main()
