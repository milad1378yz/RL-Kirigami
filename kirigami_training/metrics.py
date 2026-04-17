from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import torch

from data_generator.utils import mask_iou, mask_siou, render_structure_mask_and_metrics
from kirigami_training.rewards import compute_shape_reward, shape_penalty_from_metrics


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

    if pred_cpu.dim() != 4 or pred_cpu.shape[1] != 1:
        raise ValueError(f"Expected pred_x with shape [B,1,H,W], got {pred_cpu.shape}.")
    pred_np = pred_cpu[:, 0].numpy()

    if masks_cpu.dim() != 4 or masks_cpu.shape[1] != 1:
        raise ValueError(f"Expected masks with shape [B,1,H,W], got {masks_cpu.shape}.")
    mask_np = masks_cpu[:, 0].numpy()
    out_h, out_w = mask_np.shape[-2:]
    rows = int(context["rows"])
    cols = int(context["cols"])
    use_iou_reward = reward_metric == "iou"

    def _metrics_for_index(i: int):
        pred_mask, geom_metrics, _, _ = render_structure_mask_and_metrics(
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
            metric_value = iou if use_iou_reward else siou
            reward_raw, reward = compute_shape_reward(metric_value, penalty, reward_cfg)

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
