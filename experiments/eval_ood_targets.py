"""Evaluate the trained generator on the OOD target silhouettes.

Loads the paper's checkpoint, samples ``K`` ratio fields per OOD target
(reporting both the K=1 result and the best-of-K result), decodes each through
the existing geometry simulator, and scores silhouette match + feasibility.
Produces a per-target CSV, a per-bucket summary, the sIoU-vs-solidity figure
(the "where it works / where it breaks" plot), and a success/failure panel.

Run from the repo root, after ``experiments.make_ood_targets``:
    python -m experiments.eval_ood_targets --ckpt <path> --k 8
"""

import argparse
import csv
import math
import os
import re
import sys
import time

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data_generator.utils import (  # noqa: E402
    build_geometry_context,
    mask_overlay_rgb,
    render_structure_mask_and_metrics,
)
from kirigami_training.data import model_to_x_space, prepare_training_config  # noqa: E402
from kirigami_training.metrics import compute_shape_metrics_batch  # noqa: E402
from kirigami_training.model import build_model  # noqa: E402
from kirigami_training.sampling import sample_with_solver  # noqa: E402
from kirigami_training.utils import load_config, select_training_config  # noqa: E402
from data_generator.visualization import plot_x_matrix_structure  # noqa: E402

BUCKET_ORDER = ["convex", "concave", "topological_limit", "literal"]
BUCKET_COLORS = {
    "convex": "#2ca02c",
    "concave": "#1f77b4",
    "topological_limit": "#d62728",
    "literal": "#9467bd",
}

def _centroid_area(mask_bool):
    ys, xs = np.nonzero(mask_bool)
    if xs.size == 0:
        return 0.0, 0.0, 0
    return float(xs.mean()), float(ys.mean()), int(xs.size)


def best_alignment(pred_mask, gt_mask, *, refine: bool = True):
    """Maximize IoU of the prediction over similarity transforms (rotation +
    scale + translation) and return (best_sIoU, aligned_pred_in_target_frame).

    sIoU is meant to be invariant to pose/size, so we search the full group:
    a dense rotation x coarse-scale grid (global, avoids the single-scale
    greedy trap), then a local rotation/scale/translation polish.
    """
    P = np.asarray(pred_mask, dtype=np.float32) >= 0.5
    G = np.asarray(gt_mask, dtype=np.float32) >= 0.5
    h, w = G.shape
    pcx, pcy, pa = _centroid_area(P)
    gcx, gcy, ga = _centroid_area(G)
    if pa == 0 or ga == 0:
        return 0.0, np.zeros_like(G)

    gxx, gyy = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    Gsum = float(ga)
    base = math.sqrt(ga / pa)

    def warp(scale, angle, tx, ty):
        c, s = math.cos(angle), math.sin(angle)
        xr = gxx - (gcx + tx)
        yr = gyy - (gcy + ty)
        xin = (c * xr + s * yr) / scale + pcx
        yin = (-s * xr + c * yr) / scale + pcy
        xi = np.rint(xin).astype(np.int64)
        yi = np.rint(yin).astype(np.int64)
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        out = np.zeros((h, w), dtype=bool)
        out[ok] = P[yi[ok], xi[ok]]
        return out

    def iou(m):
        inter = float(np.count_nonzero(m & G))
        if inter == 0.0:
            return 0.0
        return inter / (float(np.count_nonzero(m)) + Gsum - inter)

    best = (-1.0, base, 0.0, 0.0, 0.0)
    scales1 = base * np.array([0.8, 0.9, 1.0, 1.1, 1.25])
    angles1 = np.linspace(0.0, 2.0 * math.pi, 180 if not refine else 360, endpoint=False)
    for sc in scales1:
        for an in angles1:
            v = iou(warp(sc, an, 0.0, 0.0))
            if v > best[0]:
                best = (v, sc, an, 0.0, 0.0)

    if refine:
        _, sc0, an0, _, _ = best
        for an in an0 + np.deg2rad(np.linspace(-2.5, 2.5, 9)):
            for sc in sc0 * np.linspace(0.85, 1.18, 9):
                for tx in np.linspace(-6.0, 6.0, 3):
                    for ty in np.linspace(-6.0, 6.0, 3):
                        v = iou(warp(sc, an, tx, ty))
                        if v > best[0]:
                            best = (v, sc, an, tx, ty)

    v, sc, an, tx, ty = best
    return float(v), warp(sc, an, tx, ty)


def pretty_name(name: str) -> str:
    """Human-readable label for an internal target id like 'concave/star5_in0.50'."""
    base = name.split("/", 1)[1] if "/" in name else name

    def num(tok: str) -> str:
        return tok.rstrip("0").rstrip(".") if "." in tok else tok

    m = re.match(r"ellipse_ar([0-9.]+)", base)
    if m:
        return f"Ellipse, AR {num(m.group(1))}"
    m = re.match(r"regular_polygon_n([0-9]+)", base)
    if m:
        return {3: "Triangle", 4: "Square", 5: "Pentagon", 6: "Hexagon", 8: "Octagon"}.get(
            int(m.group(1)), f"{m.group(1)}-gon"
        )
    m = re.match(r"rounded_rect_a([0-9.]+)_r([0-9.]+)", base)
    if m:
        return f"Rounded rect, AR {num(m.group(1))}"
    m = re.match(r"superellipse_e([0-9.]+)", base)
    if m:
        return f"Superellipse, n={num(m.group(1))}"
    m = re.match(r"star([0-9]+)_in([0-9.]+)", base)
    if m:
        return f"{m.group(1)}-point star, depth {num(m.group(2))}"
    m = re.match(r"crescent_off([0-9.]+)", base)
    if m:
        return f"Crescent, offset {num(m.group(1))}"
    m = re.match(r"dumbbell_w([0-9.]+)", base)
    if m:
        return f"Dumbbell, waist {num(m.group(1))}"
    m = re.match(r"plus_arm([0-9.]+)", base)
    if m:
        return f"Plus, arm {num(m.group(1))}"
    m = re.match(r"L_notch([0-9.]+)", base)
    if m:
        return f"L-shape, notch {num(m.group(1))}"
    m = re.match(r"annulus_in([0-9.]+)", base)
    if m:
        return f"Annulus, inner {num(m.group(1))}"
    m = re.match(r"spiral_t([0-9.]+)", base)
    if m:
        return f"Spiral, {num(m.group(1))} turns"
    m = re.match(r"letter_([A-Z])", base)
    if m:
        return f"Letter {m.group(1)}"
    return {
        "T_shape": "T-shape",
        "astroid": "Astroid",
        "ring_O": "Ring (O)",
        "doodle": "Free-form doodle",
    }.get(base, base.replace("_", " "))


def resolve_ckpt(path_or_dir: str) -> str:
    """Accept an explicit .ckpt or a dir; pick the highest-valSIoU epoch ckpt."""
    if os.path.isfile(path_or_dir):
        return path_or_dir
    if not os.path.isdir(path_or_dir):
        raise FileNotFoundError(path_or_dir)
    best, best_siou = None, -1.0
    for fn in os.listdir(path_or_dir):
        m = re.search(r"valSIoU([0-9.]+)\.ckpt$", fn)
        if m:
            s = float(m.group(1))
            if s > best_siou:
                best, best_siou = fn, s
    if best is None:
        last = os.path.join(path_or_dir, "last.ckpt")
        if os.path.isfile(last):
            return last
        raise FileNotFoundError(f"No epoch/last checkpoint found in {path_or_dir}")
    print(f"Resolved checkpoint: {best} (val SIoU {best_siou:.4f})")
    return os.path.join(path_or_dir, best)


def load_model_weights(model: torch.nn.Module, ckpt_path: str) -> None:
    """Same extraction rule as rl_training.RLFlowMatchModule._load_model_weights."""
    ckpt = torch.load(os.path.expanduser(ckpt_path), map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model_state = {k.split("model.", 1)[1]: v for k, v in state.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys (e.g. {missing[:2]})")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys (e.g. {unexpected[:2]})")


def failure_category(row: dict, success_threshold: float) -> str:
    if not row["build_ok"]:
        return "build_failed"
    if row["invalid_quad_count"] > 0:
        return "invalid_quads"
    if row["overlap_ratio"] > 0.02:
        return "overlap"
    if row["target_hole_count"] > 0 and row["siou_bestk"] < success_threshold:
        return "hole_unrepresentable"
    if row["siou_bestk"] < success_threshold:
        return "poor_match"
    return "ok"


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate the generator on OOD targets.")
    p.add_argument("--config", default="configs/training.yaml")
    p.add_argument(
        "--training-key", default="fm_training", help="fm_training (OT-CFM prior) or rl_training."
    )
    p.add_argument("--ckpt", default="checkpoints/training", help=".ckpt file or a checkpoint dir.")
    p.add_argument("--targets", default="outputs/ood/ood_targets.npz")
    p.add_argument("--out-dir", default="outputs/ood")
    p.add_argument("--k", type=int, default=8, help="Samples per target (best-of-K).")
    p.add_argument("--success-threshold", type=float, default=0.5, help="sIoU success cutoff.")
    p.add_argument(
        "--limit", type=int, default=0, help="Evaluate only the first N targets (smoke test)."
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--figures-only",
        action="store_true",
        help="Skip sampling; rebuild figures from a previous run's CSV + saved fields.",
    )
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = prepare_training_config(
        select_training_config(load_config(args.config), args.training_key)
    )
    data_cfg = config["data"]
    tr = config["training"]
    rows, cols = int(data_cfg["grid_rows"]), int(data_cfg["grid_cols"])
    x_min, x_max = float(data_cfg["x_min"]), float(data_cfg["x_max"])
    context = build_geometry_context(rows, cols)
    solver_config = {
        "method": tr["method"],
        "step_size": tr["step_size"],
        "time_points": tr["time_points"],
    }
    source_std = float(tr.get("source_noise_std", 0.5))

    if args.figures_only:
        blob = np.load(args.targets, allow_pickle=True)
        mask_by_name = dict(zip(list(blob["names"]), blob["masks"].astype(np.float32)))
        with open(os.path.join(args.out_dir, "ood_results.csv"), encoding="utf-8") as fh:
            rows_out = list(csv.DictReader(fh))
        for r in rows_out:  # csv reads everything as str; restore the types figures use
            r["solidity"] = float(r["solidity"])
            r["siou_bestk"] = float(r["siou_bestk"])
            r["target_hole_count"] = int(r["target_hole_count"])
        fields = np.load(os.path.join(args.out_dir, "ood_pred_fields.npz"), allow_pickle=True)
        field_by_name = dict(zip(list(fields["names"]), fields["pred_x"]))
        masks_for_fig = np.stack([mask_by_name[r["name"]] for r in rows_out])
        best_pred_x = {k: field_by_name[r["name"]] for k, r in enumerate(rows_out)}
        _make_figures(
            rows_out, best_pred_x, masks_for_fig, context, rows, cols, x_min, x_max, args.out_dir
        )
        print("Rebuilt figures from cached results (no sampling).")
        return

    model = build_model(config, device=device)
    load_model_weights(model, resolve_ckpt(args.ckpt))
    model.eval()
    torch.set_grad_enabled(False)

    blob = np.load(args.targets, allow_pickle=True)
    masks_np = blob["masks"].astype(np.float32)
    names = list(blob["names"])
    buckets = list(blob["buckets"])
    families = list(blob["families"])
    solidity = blob["solidity"].astype(np.float32)
    hole_count = blob["hole_count"].astype(np.int32)
    n = masks_np.shape[0] if not args.limit else min(args.limit, masks_np.shape[0])
    print(f"Evaluating {n} targets, K={args.k}, on {device} ({args.training_key}).")

    rows_out: list[dict] = []
    best_pred_x: dict[int, np.ndarray] = {}
    t_start = time.time()
    for i in range(n):
        gt = torch.from_numpy(masks_np[i])[None, None]  # [1,1,H,W]
        masks_k = gt.repeat(args.k, 1, 1, 1).to(device)
        x0 = source_std * torch.randn(args.k, 1, rows, cols, device=device)
        pred_z = sample_with_solver(
            model, x0, solver_config, masks=masks_k, return_intermediates=False
        )
        pred_x = model_to_x_space(pred_z, x_min=x_min, x_max=x_max)
        m = compute_shape_metrics_batch(
            pred_x, masks_k, context, x_min=x_min, x_max=x_max, device=device
        )

        # Max-coverage sIoU per candidate (coarse rank, then exact refine on
        # the chosen best and the K=1 candidate for the reported numbers).
        pred_x_np = pred_x[:, 0].detach().cpu().numpy()
        h_i, w_i = masks_np[i].shape
        pred_masks = [
            render_structure_mask_and_metrics(
                rows, cols, pred_x_np[k], context, h_i, w_i, x_min=x_min, x_max=x_max
            )[0]
            for k in range(args.k)
        ]
        rank = np.array(
            [best_alignment(pm, masks_np[i], refine=False)[0] for pm in pred_masks],
            dtype=np.float32,
        )
        best = int(np.argmax(rank))
        siou = rank.copy()
        siou[best] = best_alignment(pred_masks[best], masks_np[i], refine=True)[0]
        siou[0] = best_alignment(pred_masks[0], masks_np[i], refine=True)[0]
        best_pred_x[i] = pred_x[best, 0].cpu().numpy()
        row = {
            "name": names[i],
            "bucket": buckets[i],
            "family": families[i],
            "solidity": round(float(solidity[i]), 4),
            "target_hole_count": int(hole_count[i]),
            "siou_k1": round(float(siou[0]), 4),
            "siou_bestk": round(float(siou[best]), 4),
            "siou_mean": round(float(siou.mean()), 4),
            "iou_bestk": round(float(m["iou"].cpu().numpy()[best]), 4),
            "build_ok": bool(m["build_ok"].cpu().numpy()[best] > 0.5),
            "invalid_quad_count": int(m["invalid_quad_count"].cpu().numpy()[best]),
            "overlap_ratio": round(float(m["overlap_ratio"].cpu().numpy()[best]), 4),
            "fill_error": round(float(m["fill_error"].cpu().numpy()[best]), 4),
            "clipped_fraction": round(float(m["clipped_fraction"].cpu().numpy()[best]), 4),
        }
        row["failure_category"] = failure_category(row, args.success_threshold)
        row["success"] = (
            row["siou_bestk"] >= args.success_threshold and row["failure_category"] == "ok"
        )
        rows_out.append(row)
        print(
            f"  [{i+1:2d}/{n}] {row['name']:<34} sol={row['solidity']:.2f} "
            f"sIoU(K1)={row['siou_k1']:.3f} sIoU(K{args.k})={row['siou_bestk']:.3f} "
            f"{row['failure_category']}"
        )

    dt = time.time() - t_start
    os.makedirs(args.out_dir, exist_ok=True)

    csv_path = os.path.join(args.out_dir, "ood_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    summary_path = os.path.join(args.out_dir, "ood_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["bucket", "n", "siou_k1_mean", "siou_bestk_mean", "success_rate", "build_fail_rate"]
        )
        for b in BUCKET_ORDER + ["ALL"]:
            grp = rows_out if b == "ALL" else [r for r in rows_out if r["bucket"] == b]
            if not grp:
                continue
            w.writerow(
                [
                    b,
                    len(grp),
                    round(float(np.mean([r["siou_k1"] for r in grp])), 4),
                    round(float(np.mean([r["siou_bestk"] for r in grp])), 4),
                    round(float(np.mean([r["success"] for r in grp])), 4),
                    round(float(np.mean([not r["build_ok"] for r in grp])), 4),
                ]
            )

    fields_path = os.path.join(args.out_dir, "ood_pred_fields.npz")
    np.savez_compressed(
        fields_path,
        names=np.array([r["name"] for r in rows_out], dtype=object),
        pred_x=np.stack([best_pred_x[i] for i in range(len(rows_out))]).astype(np.float32),
    )

    fig_path = _make_figures(
        rows_out, best_pred_x, masks_np, context, rows, cols, x_min, x_max, args.out_dir
    )

    print(f"\nDone in {dt:.0f}s ({dt/max(1,n):.1f}s/target).")
    print(f"  per-target : {csv_path}")
    print(f"  summary    : {summary_path}")
    print(f"  figure     : {fig_path}")
    with open(summary_path, encoding="utf-8") as fh:
        print("\n" + fh.read())


# Representative (deliberately not pathological) cases spanning the story.
CURATED_PANEL = [
    ("convex/superellipse_e4.0", "convex - works"),
    ("concave/L_notch0.40", "concave - works"),
    ("concave/plus_arm0.55", "concave - partial"),
    ("literal/doodle", "hand-drawn - works"),
    ("topological_limit/annulus_in0.50", "hole - decoder limit"),
]

# A few scatter points to call out with a small (target | overlay) image.
# Offsets (in points) are hand-tuned to land in empty regions with short
# leaders and no box-box overlap.
CURATED_CALLOUTS = {
    "concave/L_notch0.40": (34.0, -78.0),
    "concave/plus_arm0.40": (-66.0, 26.0),
    "literal/letter_S": (-64.0, -40.0),
    "topological_limit/annulus_in0.70": (60.0, -34.0),
}


def _overlay_target_frame(pred_x, gt_mask, context, rows, cols, x_min, x_max):
    """Decoded mask + overlay with the generated shape warped onto the
    undistorted target at the pose/size that maximizes coverage (sIoU)."""
    pred_mask, _, _, _ = render_structure_mask_and_metrics(
        rows,
        cols,
        pred_x,
        context,
        gt_mask.shape[0],
        gt_mask.shape[1],
        x_min=x_min,
        x_max=x_max,
    )
    siou, aligned_pred = best_alignment(pred_mask, gt_mask, refine=True)
    overlay = mask_overlay_rgb(aligned_pred.astype(np.float32), gt_mask).astype(np.float32)
    overlay[~np.any(overlay > 0, axis=2)] = 1.0  # black background -> white (print-clean)
    return pred_mask, overlay, float(siou)


def _make_figures(rows_out, best_pred_x, masks_np, context, rows, cols, x_min, x_max, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from matplotlib.patches import Patch

    plt.rcParams.update(
        {
            "font.size": 12,
            "font.weight": "bold",
            "axes.labelsize": 14,
            "axes.labelweight": "bold",
            "axes.linewidth": 1.1,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )
    colors = BUCKET_COLORS
    overlay_key = [
        Patch(facecolor=(0.0, 0.8, 0.0), edgecolor="k", label="match"),
        Patch(facecolor=(1.0, 0.0, 0.0), edgecolor="k", label="generated only"),
        Patch(facecolor=(0.0, 0.0, 1.0), edgecolor="k", label="target only"),
    ]
    name_to_idx = {r["name"]: i for i, r in enumerate(rows_out)}

    fig = plt.figure(figsize=(13.0, 11.2))
    # Top band: (a) vertical preview | (b) representative cases. Bottom: (c) scatter.
    top_top, top_bot = 0.955, 0.470
    sc_top, sc_bot = 0.405, 0.085

    # ---- (a) OOD target set: vertical thumbnail strip, border = category ------
    order = sorted(
        range(len(rows_out)),
        key=lambda i: (BUCKET_ORDER.index(rows_out[i]["bucket"]), -rows_out[i]["solidity"]),
    )
    pcols = 5
    prows = int(np.ceil(len(order) / pcols))
    gsa = fig.add_gridspec(
        prows, pcols, left=0.045, right=0.315, top=top_top, bottom=top_bot, wspace=0.06, hspace=0.06
    )
    for slot, idx in enumerate(order):
        axc = fig.add_subplot(gsa[slot // pcols, slot % pcols])
        axc.imshow(masks_np[idx], cmap="gray_r", vmin=0.0, vmax=1.0)
        axc.set_xticks([])
        axc.set_yticks([])
        for sp in axc.spines.values():
            sp.set_edgecolor(colors[rows_out[idx]["bucket"]])
            sp.set_linewidth(2.0)
    for slot in range(len(order), prows * pcols):
        fig.add_subplot(gsa[slot // pcols, slot % pcols]).axis("off")
    bucket_handles = [
        Line2D([0], [0], marker="s", ls="", mec="k", mfc=colors[b], ms=11, label=b.replace("_", " "))
        for b in BUCKET_ORDER
        if any(r["bucket"] == b for r in rows_out)
    ]

    # ---- (b) representative results -------------------------------------------
    picks = [(nm, name_to_idx[nm]) for nm, _ in CURATED_PANEL if nm in name_to_idx]
    if len(picks) < 3:  # smoke-test fallback
        ranked = sorted(range(len(rows_out)), key=lambda i: rows_out[i]["siou_bestk"])
        picks = [
            (rows_out[ranked[int(q * (len(ranked) - 1))]]["name"], ranked[int(q * (len(ranked) - 1))])
            for q in (0.9, 0.55, 0.2)
        ]
    headers = ["target", "structure", "generated", "overlay"]
    b_left = 0.515
    gsb = fig.add_gridspec(
        len(picks), 4, left=b_left, right=0.99, top=top_top, bottom=top_bot, wspace=0.04, hspace=0.10
    )
    for r_idx, (_, i) in enumerate(picks):
        gt_mask = masks_np[i]
        pred_x = best_pred_x[i]
        pred_mask, overlay, siou = _overlay_target_frame(
            pred_x, gt_mask, context, rows, cols, x_min, x_max
        )
        cells = []
        for c in range(4):
            axc = fig.add_subplot(gsb[r_idx, c])
            cells.append(axc)
            axc.set_xticks([])
            axc.set_yticks([])
        cells[0].imshow(gt_mask, cmap="gray_r", vmin=0.0, vmax=1.0)
        try:
            plot_x_matrix_structure(
                cells[1], pred_x, context, x_min=x_min, x_max=x_max, normalize_phi=None
            )
            cells[1].set_xticks([])
            cells[1].set_yticks([])
        except Exception:
            cells[1].text(0.5, 0.5, "invalid", ha="center", va="center", fontsize=9)
        cells[2].imshow(pred_mask, cmap="gray_r", vmin=0.0, vmax=1.0)
        cells[3].imshow(overlay)
        box = cells[0].get_position(fig)
        fig.text(
            box.x0 - 0.010,
            0.5 * (box.y0 + box.y1),
            f"{pretty_name(rows_out[i]['name'])}\nsIoU = {siou:.2f}",
            fontsize=12,
            fontweight="bold",
            ha="right",
            va="center",
        )
        if r_idx == 0:
            for c in range(4):
                cells[c].set_title(headers[c], fontsize=13, fontweight="bold")
    fig.legend(
        handles=overlay_key,
        loc="center",
        bbox_to_anchor=(0.5 * (b_left + 0.99), top_bot - 0.028),
        ncol=3,
        fontsize=12,
        frameon=False,
        handlelength=1.3,
        columnspacing=1.6,
    )

    # ---- (c) sIoU vs. solidity (full width) -----------------------------------
    gsc = fig.add_gridspec(1, 1, left=0.065, right=0.99, top=sc_top, bottom=sc_bot)
    axc = fig.add_subplot(gsc[0])
    sols = [r["solidity"] for r in rows_out]
    for b in BUCKET_ORDER:
        grp = [r for r in rows_out if r["bucket"] == b]
        if not grp:
            continue
        axc.scatter(
            [r["solidity"] for r in grp],
            [r["siou_bestk"] for r in grp],
            c=colors[b],
            s=[95 if r["target_hole_count"] else 58 for r in grp],
            marker="o",
            edgecolors="k",
            linewidths=[1.4 if r["target_hole_count"] else 0.5 for r in grp],
            zorder=3,
        )
    for nm, (dx, dy) in CURATED_CALLOUTS.items():
        if nm not in name_to_idx:
            continue
        i = name_to_idx[nm]
        r = rows_out[i]
        gt = masks_np[i]
        _, overlay, _ = _overlay_target_frame(best_pred_x[i], gt, context, rows, cols, x_min, x_max)
        tgt_rgb = np.repeat(1.0 - gt[..., None], 3, axis=2)
        sep = np.full((gt.shape[0], 4, 3), 0.6, dtype=np.float32)
        composite = np.concatenate([tgt_rgb, sep, overlay], axis=1)
        axc.add_artist(
            AnnotationBbox(
                OffsetImage(composite, zoom=0.30),
                (r["solidity"], r["siou_bestk"]),
                xybox=(dx, dy),
                xycoords="data",
                boxcoords="offset points",
                frameon=True,
                pad=0.18,
                bboxprops=dict(edgecolor=colors[r["bucket"]], linewidth=1.4),
                arrowprops=dict(arrowstyle="-", color="0.4", lw=1.0),
                zorder=5,
            )
        )
    hole_handle = Line2D(
        [0], [0], marker="o", ls="", mec="k", mfc="0.7", mew=1.5, ms=12, label="has hole"
    )
    axc.legend(
        handles=bucket_handles + [hole_handle],
        loc="lower right",
        fontsize=11,
        framealpha=0.95,
        ncol=1,
    )
    axc.set_xlabel("solidity")
    axc.set_ylabel("sIoU")
    axc.set_xlim(min(sols) - 0.05, 1.05)
    axc.set_ylim(0.0, 1.02)
    axc.grid(alpha=0.3)

    fig.text(0.010, 0.962, "(a)", fontsize=18, fontweight="bold", ha="left", va="bottom")
    fig.text(b_left - 0.085, 0.962, "(b)", fontsize=18, fontweight="bold", ha="left", va="bottom")
    fig.text(0.010, sc_top + 0.012, "(c)", fontsize=18, fontweight="bold", ha="left", va="bottom")

    out = os.path.join(out_dir, "ood_overview.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


if __name__ == "__main__":
    main()
