"""Distillation targets for RL fine-tuning.

Two-stage pipeline:
1. Rejection sampling — draw n_candidates per mask from the pretrained model,
   keep the highest-SIoU z per mask as an on-distribution seed.
2. (1+lambda) evolution strategy in z-space with adaptive sigma, using SIoU as
   a non-differentiable fitness. Pushes each seed past what the model can reach
   on its own so distillation has headroom.
"""
import os
import time
from typing import Optional

import numpy as np
import torch

from data_generator.utils import mask_siou, render_structure_mask_and_metrics

from .data import model_to_x_space
from .metrics import compute_shape_metrics_batch
from .sampling import sample_with_solver


def euler_sample(
    model: torch.nn.Module,
    x0: torch.Tensor,
    masks: torch.Tensor,
    time_points: int,
) -> torch.Tensor:
    """Differentiable fixed-step Euler sampler. Used for distillation backprop."""
    x = x0
    times = torch.linspace(0.0, 1.0, time_points, device=x0.device, dtype=x0.dtype)
    for i in range(time_points - 1):
        t = times[i].expand(x.shape[0])
        v = model(x, t, masks)
        x = x + (times[i + 1] - times[i]) * v
    return x


@torch.no_grad()
def gather_best_targets(
    model: torch.nn.Module,
    images: torch.Tensor,
    masks: torch.Tensor,
    metric_masks: torch.Tensor,
    *,
    solver_config: dict,
    context: dict,
    x_min: float,
    x_max: float,
    source_std: float,
    n_candidates: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample n_candidates per mask, keep highest-SIoU z per mask."""
    was_training = model.training
    model.eval()
    n = masks.shape[0]
    best_siou = torch.full((n,), -1.0, device=device)
    best_z: Optional[torch.Tensor] = None
    for i in range(n_candidates):
        x0 = source_std * torch.randn_like(images)
        pred_z = sample_with_solver(model, x0, solver_config, masks=masks, return_intermediates=False)
        pred_x = model_to_x_space(pred_z, x_min=x_min, x_max=x_max)
        m = compute_shape_metrics_batch(
            pred_x, metric_masks, context, x_min=x_min, x_max=x_max, device=device,
        )
        siou = m["siou"]
        if best_z is None:
            best_z = pred_z.clone()
            best_siou = siou.clone()
        else:
            better = siou > best_siou
            if bool(better.any()):
                idx = better.nonzero(as_tuple=True)[0]
                best_z[idx] = pred_z[idx]
                best_siou[idx] = siou[idx]
        if (i + 1) % 20 == 0 or i == n_candidates - 1:
            print(f"  rejection {i+1}/{n_candidates} per-mask={best_siou.cpu().numpy().tolist()}")
    model.train(was_training)
    assert best_z is not None
    return best_z.detach(), best_siou.detach()


def _siou_of_x(
    x_np: np.ndarray,
    gt_mask_np: np.ndarray,
    *,
    context: dict,
    rows: int,
    cols: int,
    out_h: int,
    out_w: int,
    x_min: float,
    x_max: float,
) -> float:
    pred_mask, _, _, _ = render_structure_mask_and_metrics(
        rows, cols, x_np, context, out_h, out_w, x_min=x_min, x_max=x_max,
    )
    return float(mask_siou(pred_mask, gt_mask_np))


def _es_search_one(
    z_init: np.ndarray,
    gt_mask_np: np.ndarray,
    *,
    context: dict,
    rows: int,
    cols: int,
    out_h: int,
    out_w: int,
    x_min: float,
    x_max: float,
    iters: int,
    pop_size: int,
    sigma_init: float,
    sigma_min: float,
) -> tuple[np.ndarray, float]:
    z_best = z_init.copy()
    s_best = _siou_of_x(
        np.power(10.0, z_best)[0], gt_mask_np,
        context=context, rows=rows, cols=cols, out_h=out_h, out_w=out_w,
        x_min=x_min, x_max=x_max,
    )
    z_min_clip = float(np.log10(x_min))
    z_max_clip = float(np.log10(x_max))
    sigma = float(sigma_init)
    stagnation = 0
    for it in range(iters):
        noise = np.random.randn(pop_size, *z_best.shape).astype(np.float32) * sigma
        cand_z = np.clip(z_best[None] + noise, z_min_clip, z_max_clip)
        cand_x = np.power(10.0, cand_z)
        scores = np.array(
            [
                _siou_of_x(
                    cand_x[k, 0], gt_mask_np,
                    context=context, rows=rows, cols=cols, out_h=out_h, out_w=out_w,
                    x_min=x_min, x_max=x_max,
                )
                for k in range(pop_size)
            ]
        )
        k_best = int(np.argmax(scores))
        if scores[k_best] > s_best:
            s_best = float(scores[k_best])
            z_best = cand_z[k_best]
            stagnation = 0
        else:
            stagnation += 1
        if stagnation >= 5:
            sigma = max(sigma * 0.7, float(sigma_min))
            stagnation = 0
        if it % 50 == 0 or it == iters - 1:
            print(f"    iter={it:4d} sigma={sigma:.3f} best_siou={s_best:.4f}")
    return z_best, s_best


def es_refine_targets(
    seed_z: torch.Tensor,
    seed_siou: torch.Tensor,
    metric_masks: torch.Tensor,
    *,
    context: dict,
    x_min: float,
    x_max: float,
    iters: int,
    pop_size: int,
    sigma_init: float,
    sigma_min: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = int(context["rows"])
    cols = int(context["cols"])
    out_h = int(metric_masks.shape[-2])
    out_w = int(metric_masks.shape[-1])
    device = seed_z.device
    dtype = seed_z.dtype
    improved_z = seed_z.clone()
    improved_siou = seed_siou.clone().to(dtype=torch.float32)
    for i in range(seed_z.shape[0]):
        gt_np = metric_masks[i, 0].detach().cpu().numpy().astype(np.float32)
        z0 = seed_z[i].detach().cpu().numpy().astype(np.float32)
        print(f"[mask {i}] ES from SIoU={float(seed_siou[i]):.4f}")
        t0 = time.time()
        z_ev, s_ev = _es_search_one(
            z0, gt_np,
            context=context, rows=rows, cols=cols, out_h=out_h, out_w=out_w,
            x_min=x_min, x_max=x_max,
            iters=iters, pop_size=pop_size,
            sigma_init=sigma_init, sigma_min=sigma_min,
        )
        print(f"  done in {time.time()-t0:.0f}s, SIoU={s_ev:.4f}")
        improved_z[i] = torch.from_numpy(z_ev).to(device=device, dtype=dtype)
        improved_siou[i] = float(s_ev)
    return improved_z, improved_siou


def compute_distillation_targets(
    model: torch.nn.Module,
    images: torch.Tensor,
    masks: torch.Tensor,
    metric_masks: torch.Tensor,
    *,
    solver_config: dict,
    context: dict,
    x_min: float,
    x_max: float,
    source_std: float,
    n_candidates: int,
    es_iters: int,
    es_pop_size: int,
    es_sigma_init: float,
    es_sigma_min: float,
    device: torch.device,
    cache_path: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rejection-seed then ES-refine. Caches targets to `cache_path` if set."""
    if cache_path and os.path.exists(cache_path):
        pkg = torch.load(cache_path, map_location=device)
        targets_z = pkg["targets_z"].to(device)
        targets_siou = pkg["targets_siou"].to(device)
        print(
            f"Loaded cached targets from {cache_path} "
            f"(mean SIoU={targets_siou.mean().item():.4f}, "
            f"per-mask={targets_siou.cpu().numpy().tolist()})"
        )
        return targets_z, targets_siou

    print(f"Stage 1: rejection sampling {n_candidates} candidates per mask...")
    seed_z, seed_siou = gather_best_targets(
        model, images, masks, metric_masks,
        solver_config=solver_config, context=context,
        x_min=x_min, x_max=x_max, source_std=source_std,
        n_candidates=n_candidates, device=device,
    )
    print(f"  rejection seeds mean SIoU={seed_siou.mean().item():.4f}")

    print(f"Stage 2: ES refinement ({es_iters} iters x {es_pop_size} pop per mask)...")
    targets_z, targets_siou = es_refine_targets(
        seed_z, seed_siou, metric_masks,
        context=context, x_min=x_min, x_max=x_max,
        iters=es_iters, pop_size=es_pop_size,
        sigma_init=es_sigma_init, sigma_min=es_sigma_min,
    )
    print(
        f"  final targets mean SIoU={targets_siou.mean().item():.4f} "
        f"per-mask={targets_siou.cpu().numpy().tolist()}"
    )

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(
            {
                "targets_z": targets_z.detach().cpu(),
                "targets_siou": targets_siou.detach().cpu(),
            },
            cache_path,
        )
        print(f"Saved targets to {cache_path}")
    return targets_z, targets_siou
