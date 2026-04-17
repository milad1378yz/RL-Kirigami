import math
from typing import Optional


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


def compute_shape_reward(metric: float, penalty: float, cfg: dict) -> tuple[float, float]:
    metric_value = float(metric)
    penalty_value = float(penalty)
    raw_reward = metric_value - penalty_value

    transform = cfg.get("transform", "none")
    transformed = metric_value
    if transform == "logit":
        eps = max(1e-6, min(0.49, float(cfg.get("logit_eps", 1e-6))))
        clipped = min(max(metric_value, eps), 1.0 - eps)
        transformed = math.log(clipped / (1.0 - clipped))
    elif transform == "power":
        transformed = max(0.0, metric_value) ** float(cfg.get("power", 1.0))
    elif transform == "sqrt":
        transformed = math.sqrt(max(0.0, metric_value))
    elif transform == "log1p":
        transformed = math.log1p(max(0.0, metric_value))

    shaped_reward = float(cfg.get("scale", 1.0)) * transformed + float(cfg.get("shift", 0.0))
    shaped_reward -= float(cfg.get("penalty_scale", 1.0)) * penalty_value
    return float(raw_reward), float(shaped_reward)
