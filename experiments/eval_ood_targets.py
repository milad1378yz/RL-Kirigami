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
import pickle
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
from kirigami_training.model import build_model  # noqa: E402
from kirigami_training.sampling import sample_with_solver  # noqa: E402
from kirigami_training.utils import load_config, select_training_config  # noqa: E402
from data_generator.visualization import plot_x_matrix_structure  # noqa: E402

BUCKET_ORDER = ["convex", "concave", "with_hole", "literal"]
# Deliberately avoid the overlay RGB (green=match, red=generated-only,
# blue=target-only) so category color never reads as an overlay channel.
BUCKET_COLORS = {
    "convex": "#ff7f0e",  # orange
    "concave": "#9467bd",  # purple
    "with_hole": "#8c564b",  # brown
    "literal": "#e377c2",  # pink
}


def _centroid_area(mask_bool):
    ys, xs = np.nonzero(mask_bool)
    if xs.size == 0:
        return 0.0, 0.0, 0
    return float(xs.mean()), float(ys.mean()), int(xs.size)


def best_alignment(pred_mask, gt_mask, *, refine: bool = True):
    """Best-coverage sIoU: align the prediction to the target over rotation and
    translation at the *area-matched* scale, return (sIoU, aligned_pred).

    Scale is fixed to sqrt(area_gt / area_pred). At that scale the aligned
    prediction and the target have equal area, so the overlay's symmetric
    difference is balanced -- |generated-only| == |target-only| -- which is
    what a good, pose/size-invariant match should look like. Rotation is
    searched globally (dense grid) then polished together with translation.
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
    scale = math.sqrt(ga / pa)  # area match -> balanced XOR

    def warp(angle, tx, ty):
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

    best = (-1.0, 0.0, 0.0, 0.0)
    for an in np.linspace(0.0, 2.0 * math.pi, 240 if not refine else 720, endpoint=False):
        v = iou(warp(an, 0.0, 0.0))
        if v > best[0]:
            best = (v, an, 0.0, 0.0)

    if refine:
        _, an0, _, _ = best
        for an in an0 + np.deg2rad(np.linspace(-2.0, 2.0, 9)):
            for tx in np.linspace(-7.0, 7.0, 5):
                for ty in np.linspace(-7.0, 7.0, 5):
                    v = iou(warp(an, tx, ty))
                    if v > best[0]:
                        best = (v, an, tx, ty)

    v, an, tx, ty = best
    return float(v), warp(an, tx, ty)


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


def _write_csvs(rows_out: list[dict], out_dir: str) -> tuple[str, str]:
    """Write the per-target and per-bucket CSVs and return their paths.

    The summary aggregates only mask/sampling-determined metrics (sIoU,
    build_ok), so it is correct to regenerate from cached rows after a
    re-bucketing -- the bucket label changes, the numbers do not.
    """
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "ood_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    summary_path = os.path.join(out_dir, "ood_summary.csv")
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
    return csv_path, summary_path


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate the generator on OOD targets.")
    p.add_argument("--config", default="configs/training.yaml")
    p.add_argument(
        "--training-key", default="fm_training", help="fm_training (OT-CFM prior) or rl_training."
    )
    p.add_argument("--ckpt", default="checkpoints/training", help=".ckpt file or a checkpoint dir.")
    p.add_argument("--targets", default="outputs/ood/ood_targets.npz")
    p.add_argument("--out-dir", default="outputs/ood")
    p.add_argument("--k", type=int, default=128, help="Samples per target (best-of-K).")
    p.add_argument(
        "--sample-chunk", type=int, default=32, help="Sub-batch size for sampling large K."
    )
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
        # bucket / solidity / hole_count are properties of the target set, not
        # of an eval run -- take them from the npz so a re-bucketed target file
        # is reflected without re-sampling (the sIoU numbers are mask-determined
        # and unchanged). Only the per-run sIoU columns come from the CSV.
        idx_by_name = {n: j for j, n in enumerate(list(blob["names"]))}
        bk, so, hc = blob["buckets"], blob["solidity"], blob["hole_count"]
        for r in rows_out:  # csv reads everything as str; restore the types used below
            j = idx_by_name[r["name"]]
            r["bucket"] = str(bk[j])
            r["solidity"] = round(float(so[j]), 4)
            r["target_hole_count"] = int(hc[j])
            r["siou_k1"] = float(r["siou_k1"])
            r["siou_bestk"] = float(r["siou_bestk"])
            r["success"] = r["success"] == "True"
            r["build_ok"] = r["build_ok"] == "True"
        # Persist so ood_results.csv / ood_summary.csv agree with the figure
        # and the re-bucketed npz (numbers unchanged, only the labels move).
        csv_path, summary_path = _write_csvs(rows_out, args.out_dir)
        fields = np.load(os.path.join(args.out_dir, "ood_pred_fields.npz"), allow_pickle=True)
        field_by_name = dict(zip(list(fields["names"]), fields["pred_x"]))
        masks_for_fig = np.stack([mask_by_name[r["name"]] for r in rows_out])
        best_pred_x = {k: field_by_name[r["name"]] for k, r in enumerate(rows_out)}
        fig_path = _make_figures(
            rows_out, best_pred_x, masks_for_fig, context, rows, cols, x_min, x_max, args.out_dir
        )
        print("Rebuilt artifacts from cached results (no sampling):")
        print(f"  per-target : {csv_path}")
        print(f"  summary    : {summary_path}")
        print(f"  figure     : {fig_path}")
        with open(summary_path, encoding="utf-8") as fh:
            print("\n" + fh.read())
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
    h_i, w_i = masks_np.shape[1], masks_np.shape[2]
    for i in range(n):
        gt_np = masks_np[i]
        gt_bool = gt_np >= 0.5

        # Sample K candidates in chunks (K can be large, e.g. best-of-128).
        pred_masks, pred_metrics, pred_x_all = [], [], []
        remaining = args.k
        while remaining > 0:
            cb = min(args.sample_chunk, remaining)
            remaining -= cb
            masks_cb = torch.from_numpy(gt_np)[None, None].repeat(cb, 1, 1, 1).to(device)
            x0 = source_std * torch.randn(cb, 1, rows, cols, device=device)
            pred_z = sample_with_solver(
                model, x0, solver_config, masks=masks_cb, return_intermediates=False
            )
            pred_x = model_to_x_space(pred_z, x_min=x_min, x_max=x_max)[:, 0].cpu().numpy()
            for k in range(cb):
                pmask, pmet, _, _ = render_structure_mask_and_metrics(
                    rows, cols, pred_x[k], context, h_i, w_i, x_min=x_min, x_max=x_max
                )
                pred_masks.append(pmask)
                pred_metrics.append(pmet)
                pred_x_all.append(pred_x[k])

        # Area-balanced max-coverage sIoU: coarse rank, exact refine on the
        # chosen best and on the K=1 candidate for the reported numbers.
        rank = np.array(
            [best_alignment(pm, gt_np, refine=False)[0] for pm in pred_masks],
            dtype=np.float32,
        )
        best = int(np.argmax(rank))
        siou = rank.copy()
        siou[best] = best_alignment(pred_masks[best], gt_np, refine=True)[0]
        siou[0] = best_alignment(pred_masks[0], gt_np, refine=True)[0]
        best_pred_x[i] = pred_x_all[best]

        mb = pred_metrics[best]
        pb = pred_masks[best] >= 0.5
        inter = float(np.count_nonzero(pb & gt_bool))
        union = float(np.count_nonzero(pb | gt_bool))
        row = {
            "name": names[i],
            "bucket": buckets[i],
            "family": families[i],
            "solidity": round(float(solidity[i]), 4),
            "target_hole_count": int(hole_count[i]),
            "siou_k1": round(float(siou[0]), 4),
            "siou_bestk": round(float(siou[best]), 4),
            "siou_mean": round(float(siou.mean()), 4),
            "iou_bestk": round(inter / union if union else 0.0, 4),
            "build_ok": bool(mb.get("ok", False)),
            "invalid_quad_count": int(mb.get("invalid_quad_count", 0) or 0),
            "overlap_ratio": round(float(mb.get("overlap_ratio", 1.0) or 0.0), 4),
            "fill_error": round(abs(float(mb.get("fill_ratio", 0.0)) - float(gt_np.mean())), 4),
            "clipped_fraction": round(float(mb.get("clipped_fraction", 0.0) or 0.0), 4),
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
    csv_path, summary_path = _write_csvs(rows_out, args.out_dir)

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


# Representative (deliberately not pathological) cases spanning the story,
# ordered high -> low solidity with the hole case last.
CURATED_PANEL = [
    ("convex/superellipse_e4.0", "convex - works"),
    ("concave/L_notch0.40", "concave - works"),
    ("literal/doodle", "hand-drawn - works"),
    ("concave/thick_plus_arm0.78", "concave - works (~0.8 solidity)"),
    ("concave/plus_arm0.55", "concave - partial"),
    ("literal/ring_O", "hole - decoder limit"),
]

# A few scatter points to call out with a small (target | overlay) image.
CURATED_CALLOUTS = {
    "concave/L_notch0.40": (-22.0, 18.0),
    "concave/plus_arm0.40": (-22.0, -14.0),
    "literal/letter_S": (25.0, 18.0),
    "topological_limit/annulus_in0.30": (0.0, -22.0),
}


def _overlay_target_frame(pred_x, gt_mask, context, rows, cols, x_min, x_max):
    """Decoded mask + overlay at the pose/size that maximizes coverage (sIoU)."""
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
    overlay[~np.any(overlay > 0, axis=2)] = 1.0
    return pred_mask, overlay, float(siou)


def _load_pickle_compat(path: str):
    # Some generated pickles refer to numpy._core, while older NumPy exposes numpy.core.
    import numpy.core as np_core
    import numpy.core.multiarray as np_multiarray
    import numpy.core.numeric as np_numeric

    sys.modules.setdefault("numpy._core", np_core)
    sys.modules.setdefault("numpy._core.multiarray", np_multiarray)
    sys.modules.setdefault("numpy._core.numeric", np_numeric)
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _load_dataset_masks(dataset_path: str, split: str) -> np.ndarray:
    data = _load_pickle_compat(dataset_path)
    masks = []
    for entry in data[split]:
        mask = np.asarray(entry["mask"], dtype=np.float32)
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        masks.append(mask)
    return np.stack(masks, axis=0)


def _flatten_masks(masks: np.ndarray) -> np.ndarray:
    return np.asarray(masks, dtype=np.float32).reshape(masks.shape[0], -1)


def _pca_ood_scores(ood_masks: np.ndarray, *, components: int = 64) -> dict:
    from scipy.stats import gaussian_kde
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    dataset_path = os.path.join(
        os.path.dirname(_REPO_ROOT), "fm_inverse_2d", "data", "kirigami_dataset_5000_500.pkl"
    )
    train_masks = _load_dataset_masks(dataset_path, "train")
    valid_masks = _load_dataset_masks(dataset_path, "valid")
    train_flat = _flatten_masks(train_masks)
    valid_flat = _flatten_masks(valid_masks)
    ood_flat = _flatten_masks(ood_masks)

    pca = PCA(n_components=components, svd_solver="randomized", random_state=0)
    train_scores = pca.fit_transform(train_flat)
    valid_scores = pca.transform(valid_flat)
    ood_scores = pca.transform(ood_flat)

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(train_scores)
    valid_nn = nn.kneighbors(valid_scores, return_distance=True)[0][:, 0]
    ood_nn = nn.kneighbors(ood_scores, return_distance=True)[0][:, 0]

    valid_rec = pca.inverse_transform(valid_scores)
    ood_rec = pca.inverse_transform(ood_scores)
    valid_rmse = np.sqrt(np.mean((valid_flat - valid_rec) ** 2, axis=1))
    ood_rmse = np.sqrt(np.mean((ood_flat - ood_rec) ** 2, axis=1))

    x_valid = valid_nn / np.median(valid_nn)
    y_valid = valid_rmse / np.median(valid_rmse)
    x_ood = ood_nn / np.median(valid_nn)
    y_ood = ood_rmse / np.median(valid_rmse)

    kde = gaussian_kde(np.vstack([x_valid, y_valid]))
    valid_density = kde(np.vstack([x_valid, y_valid]))
    mass_levels = [0.99, 0.90, 0.75, 0.50]
    levels = np.array([np.quantile(valid_density, 1.0 - mass) for mass in mass_levels])
    order = np.argsort(levels)

    return {
        "x_valid": x_valid,
        "y_valid": y_valid,
        "x_ood": x_ood,
        "y_ood": y_ood,
        "kde": kde,
        "levels": levels[order],
        "mass_levels": [mass_levels[i] for i in order],
        "explained": float(pca.explained_variance_ratio_.sum()),
        "nn_median": float(np.median(x_ood)),
        "rmse_median": float(np.median(y_ood)),
        "outside_99": int((kde(np.vstack([x_ood, y_ood])) < np.quantile(valid_density, 0.01)).sum()),
    }


def _make_figures(rows_out, best_pred_x, masks_np, context, rows, cols, x_min, x_max, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Latin Modern Roman",
                "Liberation Serif",
                "Nimbus Roman",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "cm",
            "font.size": 9,
            "font.weight": "normal",
            "axes.labelsize": 11,
            "axes.labelweight": "bold",
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.linewidth": 0.9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    colors = BUCKET_COLORS
    name_to_idx = {r["name"]: i for i, r in enumerate(rows_out)}

    fig = plt.figure(figsize=(7.45, 7.55))
    # Top band: (a) vertical preview | (b) representative cases.
    # Bottom band: (c) performance scatter | (d) PCA OOD-score contours.
    top_top, top_bot = 0.940, 0.515
    sc_top, sc_bot = 0.470, 0.058

    # ---- (a) OOD target set: vertical thumbnail strip, border = category ------
    order = sorted(
        range(len(rows_out)),
        key=lambda i: (BUCKET_ORDER.index(rows_out[i]["bucket"]), -rows_out[i]["solidity"]),
    )
    pcols = 7
    prows = int(np.ceil(len(order) / pcols))
    gsa = fig.add_gridspec(
        prows, pcols, left=0.040, right=0.675, top=top_top, bottom=top_bot, wspace=0.05, hspace=0.05
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
        Line2D(
            [0], [0], marker="s", ls="", mec="k", mfc=colors[b], ms=11, label=b.replace("_", " ")
        )
        for b in BUCKET_ORDER
        if any(r["bucket"] == b for r in rows_out)
    ]

    # ---- (b) representative results -------------------------------------------
    picks = [(nm, name_to_idx[nm]) for nm, _ in CURATED_PANEL if nm in name_to_idx]
    if len(picks) < 3:  # smoke-test fallback
        ranked = sorted(range(len(rows_out)), key=lambda i: rows_out[i]["siou_bestk"])
        picks = [
            (
                rows_out[ranked[int(q * (len(ranked) - 1))]]["name"],
                ranked[int(q * (len(ranked) - 1))],
            )
            for q in (0.9, 0.55, 0.2)
        ]
    # Paper terms / paper rendering (cf. RL-Kirigami-Latex table figure): target
    # silhouette y, the compact rectangle (phi=pi cut sheet) decoded from x, and
    # the generated deployed structure with the *aligned target* overlaid
    # (matches _siou_and_aligned_target_mask in generate_table2_visual_comparison).
    headers = ["Target", "Compact\nrectangle", "Generated"]
    b_left = 0.705
    gsb = fig.add_gridspec(
        len(picks),
        3,
        left=b_left,
        right=0.985,
        top=top_top,
        bottom=top_bot,
        wspace=0.08,
        hspace=0.06,
    )
    for r_idx, (_, i) in enumerate(picks):
        gt_mask = masks_np[i]
        pred_x = best_pred_x[i]
        pred_mask, _, _, _ = render_structure_mask_and_metrics(
            rows, cols, pred_x, context, gt_mask.shape[0], gt_mask.shape[1],
            x_min=x_min, x_max=x_max,
        )
        cells = []
        for c in range(3):
            axc = fig.add_subplot(gsb[r_idx, c])
            cells.append(axc)
            axc.set_xticks([])
            axc.set_yticks([])
        cells[0].imshow(gt_mask, cmap="gray_r", vmin=0.0, vmax=1.0)
        try:  # compact rectangle = phi=pi pose of the decoded cut sheet
            plot_x_matrix_structure(
                cells[1], pred_x, context, phi=np.pi,
                x_min=x_min, x_max=x_max, normalize_phi=np.pi,
            )
        except Exception:
            cells[1].text(0.5, 0.5, "invalid", ha="center", va="center", fontsize=9)
        try:  # generated deployed structure + target silhouette aligned onto it
            _, aligned_target = best_alignment(gt_mask, pred_mask, refine=True)
            plot_x_matrix_structure(
                cells[2], pred_x, context, mask_2d=aligned_target.astype(np.float32),
                x_min=x_min, x_max=x_max, normalize_phi=None,
            )
        except Exception:
            cells[2].text(0.5, 0.5, "invalid", ha="center", va="center", fontsize=9)
        for c in range(3):  # structure plotters clear ticks; re-assert for all
            cells[c].set_xticks([])
            cells[c].set_yticks([])
        if r_idx == 0:
            for c in range(3):
                cells[c].set_title(
                    headers[c], fontsize=8, fontweight="bold", linespacing=0.95, pad=3
                )

    # ---- (c) sIoU vs. solidity ------------------------------------------------
    gsc = fig.add_gridspec(1, 1, left=0.060, right=0.505, top=sc_top, bottom=sc_bot)
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
            s=64,
            marker="o",
            edgecolors="k",
            linewidths=0.6,
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
                OffsetImage(composite, zoom=0.120),
                (r["solidity"], r["siou_bestk"]),
                xybox=(dx, dy),
                xycoords="data",
                boxcoords="offset points",
                frameon=True,
                pad=0.10,
                bboxprops=dict(edgecolor=colors[r["bucket"]], linewidth=0.9),
                arrowprops=dict(arrowstyle="-", color="0.45", lw=0.7),
                zorder=5,
            )
        )
    # "with_hole" is its own bucket -> no separate "has hole" entry (duplication).
    bucket_leg = axc.legend(
        handles=bucket_handles,
        loc="lower left",
        fontsize=7,
        framealpha=0.95,
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.3,
    )
    axc.add_artist(bucket_leg)
    axc.set_xlabel("solidity")
    axc.set_ylabel("sIoU")
    axc.set_xlim(min(sols) - 0.05, 1.05)
    axc.set_ylim(0.0, 1.02)
    axc.grid(alpha=0.3)

    # ---- (d) PCA OOD-score contours -------------------------------------------
    gsd = fig.add_gridspec(1, 1, left=0.585, right=0.985, top=sc_top, bottom=sc_bot)
    axd = fig.add_subplot(gsd[0])
    pca_scores = _pca_ood_scores(masks_np, components=64)
    x_valid, y_valid = pca_scores["x_valid"], pca_scores["y_valid"]
    x_ood, y_ood = pca_scores["x_ood"], pca_scores["y_ood"]
    x_min_grid = 0.30
    x_max_grid = max(float(np.max(x_ood)), float(np.quantile(x_valid, 0.995))) + 0.20
    y_min_grid = 0.80
    y_max_grid = max(float(np.max(y_ood)), float(np.quantile(y_valid, 0.995))) + 0.55
    x_grid, y_grid = np.mgrid[x_min_grid:x_max_grid:220j, y_min_grid:y_max_grid:220j]
    density = pca_scores["kde"](np.vstack([x_grid.ravel(), y_grid.ravel()])).reshape(x_grid.shape)
    contour = axd.contour(
        x_grid,
        y_grid,
        density,
        levels=pca_scores["levels"],
        colors=["#8c564b", "#2f7f5f", "#1f77b4", "#34495e"],
        linewidths=[0.8, 0.9, 1.0, 1.1],
        zorder=1,
    )
    label_map = {
        level: f"{int(round(mass * 100))}%"
        for level, mass in zip(pca_scores["levels"], pca_scores["mass_levels"])
    }
    axd.clabel(contour, contour.levels, inline=True, fontsize=7, fmt=label_map)
    for b in BUCKET_ORDER:
        idx = [i for i, r in enumerate(rows_out) if r["bucket"] == b]
        if not idx:
            continue
        axd.scatter(
            x_ood[idx],
            y_ood[idx],
            c=colors[b],
            s=42,
            marker="o",
            edgecolors="k",
            linewidths=0.45,
            zorder=3,
        )
    axd.legend(
        handles=bucket_handles,
        loc="upper right",
        fontsize=7,
        framealpha=0.95,
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.3,
    )
    axd.text(
        0.03,
        0.97,
        "validation contours",
        transform=axd.transAxes,
        ha="left",
        va="top",
        fontsize=7,
    )
    axd.text(
        0.97,
        0.03,
        f"{pca_scores['outside_99']}/{len(rows_out)} outside 99%",
        transform=axd.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="0.82", alpha=0.96),
    )
    axd.set_xlabel("PCA distance / val. median")
    axd.set_ylabel("PCA recon. RMSE / val. median")
    axd.set_xlim(x_min_grid, x_max_grid)
    axd.set_ylim(y_min_grid, y_max_grid)
    axd.grid(alpha=0.3)

    fig.text(0.004, top_top + 0.035, "(a)", fontsize=13, fontweight="bold", ha="left", va="bottom")
    fig.text(
        b_left, top_top + 0.035, "(b)", fontsize=13, fontweight="bold", ha="left", va="bottom"
    )
    fig.text(0.004, sc_top + 0.014, "(c)", fontsize=13, fontweight="bold", ha="left", va="bottom")
    fig.text(0.525, sc_top + 0.014, "(d)", fontsize=13, fontweight="bold", ha="left", va="bottom")

    out = os.path.join(out_dir, "ood_overview.pdf")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


if __name__ == "__main__":
    main()
