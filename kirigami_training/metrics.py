import math
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import torch

from data_generator.utils import mask_iou, mask_siou, x_matrix_to_mask_and_metrics


def shape_penalty_from_metrics(metrics: dict, cfg: Optional[dict]) -> float:
    if not cfg or not bool(cfg.get("enabled", False)):
        return 0.0

    penalty = 0.0
    if not bool(metrics.get("ok", False)):
        penalty += float(cfg.get("build_fail", 0.0))

    invalid_count = int(metrics.get("invalid_quad_count", 0) or 0)
    if invalid_count > 0:
        penalty += float(cfg.get("invalid_any", 0.0))
        penalty += float(cfg.get("invalid_per_quad", 0.0)) * float(invalid_count)

    overlap_ratio = float(metrics.get("overlap_ratio", 0.0) or 0.0)
    overlap_thr = float(cfg.get("overlap_ratio_threshold", 0.0) or 0.0)
    if overlap_thr > 0.0:
        overlap_excess = max(0.0, overlap_ratio - overlap_thr) / overlap_thr
    else:
        overlap_excess = max(0.0, overlap_ratio)
    penalty += float(cfg.get("overlap_ratio", 0.0)) * float(overlap_excess)

    penalty += float(cfg.get("fill_error", 0.0)) * float(metrics.get("fill_error", 0.0) or 0.0)
    penalty += float(cfg.get("range_violation", 0.0)) * float(
        metrics.get("range_violation_l1", 0.0) or 0.0
    )
    penalty += float(cfg.get("clipped_fraction", 0.0)) * float(
        metrics.get("clipped_fraction", 0.0) or 0.0
    )
    return float(penalty)


def shape_reward(metric: float, penalty: float, cfg: dict) -> float:
    transform = str(cfg.get("transform", "none") or "none").lower()
    reward = float(metric)
    if transform == "logit":
        eps = max(1e-6, min(0.49, float(cfg.get("logit_eps", 1e-6))))
        reward = min(max(reward, eps), 1.0 - eps)
        reward = math.log(reward / (1.0 - reward))
    elif transform == "power":
        reward = max(0.0, reward) ** float(cfg.get("power", 1.0))
    elif transform == "sqrt":
        reward = math.sqrt(max(0.0, reward))
    elif transform == "log1p":
        reward = math.log1p(max(0.0, reward))

    reward = float(cfg.get("scale", 1.0)) * reward + float(cfg.get("shift", 0.0))
    reward -= float(cfg.get("penalty_scale", 1.0)) * float(penalty)
    return float(reward)


def compute_shape_metrics_batch(
    pred_x: torch.Tensor,
    masks: torch.Tensor,
    context: dict,
    *,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    threshold: float = 0.5,
    shape_penalty_cfg: Optional[dict] = None,
    reward_cfg: Optional[dict] = None,
    reward_metric: str = "siou",
    num_workers: int = 1,
    device: Optional[torch.device] = None,
) -> dict[str, Optional[torch.Tensor]]:
    if device is None:
        device = torch.device("cpu")

    pred_cpu = pred_x.detach().to("cpu", dtype=torch.float32)
    masks_cpu = masks.detach().to("cpu", dtype=torch.float32)

    if pred_cpu.dim() == 4:
        pred_np = pred_cpu[:, 0].numpy()
    elif pred_cpu.dim() == 3:
        pred_np = pred_cpu.numpy()
    else:
        raise ValueError(f"Expected pred_x with shape [B,1,H,W] or [B,H,W], got {pred_cpu.shape}.")

    if masks_cpu.dim() != 4 or masks_cpu.shape[1] != 1:
        raise ValueError(f"Expected masks with shape [B,1,H,W], got {masks_cpu.shape}.")
    mask_np = masks_cpu[:, 0].numpy()
    out_h, out_w = mask_np.shape[-2:]
    rows = int(context["rows"])
    cols = int(context["cols"])
    metric_name = str(reward_metric).lower()
    if metric_name not in {"iou", "siou"}:
        raise ValueError(f"Unsupported reward_metric '{reward_metric}'. Expected 'iou' or 'siou'.")

    def _metrics_for_index(i: int):
        pred_mask, geom_metrics, _, _ = x_matrix_to_mask_and_metrics(
            rows,
            cols,
            pred_np[i],
            context,
            out_h,
            out_w,
            x_min=x_min,
            x_max=x_max,
        )
        gt_mask = mask_np[i]
        iou = mask_iou(pred_mask, gt_mask, threshold=threshold)
        siou = mask_siou(pred_mask, gt_mask, threshold=threshold)
        fill_error = abs(float(geom_metrics["fill_ratio"]) - float(gt_mask.mean()))

        full_metrics = dict(geom_metrics)
        full_metrics["fill_error"] = fill_error
        penalty = shape_penalty_from_metrics(full_metrics, shape_penalty_cfg)

        reward_raw = None
        reward = None
        if reward_cfg is not None:
            metric_value = iou if metric_name == "iou" else siou
            reward_raw = metric_value - penalty
            reward = shape_reward(metric_value, penalty, reward_cfg)

        return {
            "iou": iou,
            "siou": siou,
            "reward_raw": reward_raw,
            "reward": reward,
            "penalty": penalty,
            "invalid_quad_count": int(full_metrics["invalid_quad_count"]),
            "overlap_ratio": float(full_metrics["overlap_ratio"]),
            "build_ok": bool(full_metrics["ok"]),
            "fill_error": fill_error,
            "range_violation_l1": float(full_metrics["range_violation_l1"]),
            "clipped_fraction": float(full_metrics["clipped_fraction"]),
        }

    n_items = int(pred_np.shape[0])
    workers = min(max(1, int(num_workers or 1)), max(1, n_items))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_metrics_for_index, range(n_items)))
    else:
        results = [_metrics_for_index(i) for i in range(n_items)]

    def _tensor(key: str) -> torch.Tensor:
        return torch.tensor([item[key] for item in results], device=device, dtype=torch.float32)

    reward_raw = _tensor("reward_raw") if reward_cfg is not None else None
    reward = _tensor("reward") if reward_cfg is not None else None

    return {
        "iou": _tensor("iou"),
        "siou": _tensor("siou"),
        "reward_raw": reward_raw,
        "reward": reward,
        "penalty": _tensor("penalty"),
        "invalid_quad_count": _tensor("invalid_quad_count"),
        "overlap_ratio": _tensor("overlap_ratio"),
        "build_ok": _tensor("build_ok"),
        "fill_error": _tensor("fill_error"),
        "range_violation_l1": _tensor("range_violation_l1"),
        "clipped_fraction": _tensor("clipped_fraction"),
    }
