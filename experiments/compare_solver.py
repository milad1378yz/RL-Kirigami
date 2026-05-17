"""Solver baseline vs. RL-Kirigami on the OOD targets.

A separate repo (../data_generator_kirigami) implements a *different
formulation* of the same compact parallelogram-quad kirigami: epsilon-offset
parametrization with a linear inverse-design matrix solver
(`MatrixStructure.linear_inverse_design`). `optimize_eps_shapes.optimize_shape`
wraps it in `scipy.optimize.least_squares` to fit a target boundary given as a
radial profile r(theta).

This script feeds each OOD target mask to that solver (as a centroid radial
profile -- the form the solver consumes; inherently single-valued, so deep
non-star concavity and holes are lossy for the solver by construction),
decodes the solver result to a mask, and scores it with the SAME area-balanced
max-coverage sIoU (`best_alignment`) used for RL-Kirigami. Solver runs are
parallelised across processes (each least_squares finite-differences a
100-parameter Jacobian and is slow); scoring stays in the main process so the
metric is byte-identical to the RL-Kirigami evaluation.

Run from the repo root, after `experiments.eval_ood_targets`:
    python -m experiments.compare_solver --max-nfev 80 --workers 8
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SOLVER_REPO = os.path.abspath(os.path.join(_REPO_ROOT, "..", "data_generator_kirigami"))


def _ensure_paths():
    for _p in (_REPO_ROOT, _SOLVER_REPO):
        if _p not in sys.path:
            sys.path.insert(0, _p)


def radial_fn_from_mask(mask: np.ndarray, n_bins: int = 360):
    """Centroid ray-cast radial profile r(theta) of the mask's outer boundary.

    y flipped to math convention (up) to match the solver's
    arctan2(rel_y, rel_x); absolute scale is irrelevant (solver and sIoU are
    both scale-invariant).
    """
    m = np.asarray(mask, dtype=np.float32) >= 0.5
    ys, xs = np.nonzero(m)
    if xs.size < 8:
        return None
    cx, cy = xs.mean(), ys.mean()
    ang = np.mod(np.arctan2(-(ys - cy), xs - cx), 2.0 * np.pi)
    rad = np.hypot(xs - cx, ys - cy)
    step = 2.0 * np.pi / n_bins
    b = np.minimum((ang / step).astype(int), n_bins - 1)
    prof = np.zeros(n_bins)
    np.maximum.at(prof, b, rad)
    known = prof > 0
    if known.sum() < 6:
        return None
    grid = (np.arange(n_bins) + 0.5) * step
    ka, kp = grid[known], prof[known]
    ka_ext = np.concatenate([ka - 2 * np.pi, ka, ka + 2 * np.pi])
    kp_ext = np.concatenate([kp, kp, kp])
    full = np.interp(grid, ka_ext, kp_ext)
    full = full / max(full.mean(), 1e-6)

    def r(theta: np.ndarray) -> np.ndarray:
        return np.clip(
            np.interp(np.mod(theta, 2.0 * np.pi), grid, full, period=2.0 * np.pi), 1e-4, None
        )

    return r


def _solve_one(task):
    """Worker: fit one target with the LSQ solver.

    -> (name, mask, points, quads, status, t). points/quads are the solver's
    deployed MatrixStructure (for drawing its structure in the figure).
    """
    name, gt, max_nfev = task
    _ensure_paths()
    import optimize_eps_shapes as oe  # heavy; import inside worker

    rfn = radial_fn_from_mask(gt)
    if rfn is None:
        return name, None, None, None, "no_radial", 0.0
    try:
        res = oe.optimize_shape(
            name.split("/")[-1], rfn, width=10, height=10,
            mask_size=gt.shape[0], max_nfev=max_nfev, verbose=0,
        )
        return (
            name,
            res["mask"].astype(np.float32),
            np.asarray(res["points"], dtype=np.float64),
            np.asarray(res["structure"].quads),
            "ok",
            float(res["time_sec"]),
        )
    except Exception as exc:  # solver / geometry failure -> count as a miss
        return name, None, None, None, f"fail:{type(exc).__name__}", 0.0


def _overlay_from_solver_mask(solver_mask, gt, best_alignment, mask_overlay_rgb):
    """Solver mask aligned onto the undistorted target -> (overlay_rgb, sIoU)."""
    siou, aligned = best_alignment(solver_mask, gt, refine=True)
    ov = mask_overlay_rgb(aligned.astype(np.float32), gt).astype(np.float32)
    ov[~np.any(ov > 0, axis=2)] = 1.0  # black bg -> white (print-clean)
    return ov, float(siou)


def _make_solver_figure(out_dir, targets_npz, solver_csv, artifacts_npz):
    """Same (a)/(b)/(c) layout as the RL figure, but with solver results.

    (b) draws the solver's own deployed MatrixStructure (plot_structure).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from matplotlib.patches import Patch

    _ensure_paths()
    from experiments.eval_ood_targets import (
        BUCKET_COLORS,
        BUCKET_ORDER,
        CURATED_CALLOUTS,
        CURATED_PANEL,
        best_alignment,
    )
    from data_generator.utils import mask_overlay_rgb
    from kirigami.utils import plot_structure  # from ../data_generator_kirigami

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 13,
            "font.weight": "normal",
            "axes.labelsize": 15,
            "axes.labelweight": "bold",
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.linewidth": 1.1,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )
    colors = BUCKET_COLORS
    overlay_key = [
        Patch(facecolor=(0.0, 0.8, 0.0), edgecolor="k", label="match"),
        Patch(facecolor=(1.0, 0.0, 0.0), edgecolor="k", label="generated only"),
        Patch(facecolor=(0.0, 0.0, 1.0), edgecolor="k", label="target only"),
    ]

    blob = np.load(targets_npz, allow_pickle=True)
    masks = blob["masks"].astype(np.float32)
    names = list(blob["names"])
    bucket_of = dict(zip(names, list(blob["buckets"])))
    sol_of = dict(zip(names, blob["solidity"].astype(float)))
    hole_of = dict(zip(names, blob["hole_count"].astype(int)))
    mask_of = {nm: masks[i] for i, nm in enumerate(names)}

    siou_solver = {}
    with open(solver_csv, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            siou_solver[row["name"]] = float(row["siou_solver"])

    art = np.load(artifacts_npz, allow_pickle=True)
    a_names = list(art["names"])
    a_mask = {nm: np.asarray(art["masks"][i], dtype=np.float32) for i, nm in enumerate(a_names)}
    a_pts = {nm: np.asarray(art["pts"][i], dtype=np.float64) for i, nm in enumerate(a_names)}
    a_quads = {nm: np.asarray(art["quads"][i], dtype=np.int64) for i, nm in enumerate(a_names)}

    rows_out = [
        {
            "name": nm,
            "bucket": bucket_of[nm],
            "solidity": sol_of[nm],
            "target_hole_count": hole_of[nm],
            "siou_bestk": siou_solver[nm],
        }
        for nm in names
        if nm in siou_solver
    ]

    fig = plt.figure(figsize=(13.0, 11.2))
    top_top, top_bot = 0.955, 0.470
    sc_top, sc_bot = 0.405, 0.085

    # ---- (a) OOD target set: vertical thumbnail strip ------------------------
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
        ax = fig.add_subplot(gsa[slot // pcols, slot % pcols])
        ax.imshow(mask_of[rows_out[idx]["name"]], cmap="gray_r", vmin=0.0, vmax=1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(colors[rows_out[idx]["bucket"]])
            sp.set_linewidth(2.0)
    for slot in range(len(order), prows * pcols):
        fig.add_subplot(gsa[slot // pcols, slot % pcols]).axis("off")
    bucket_handles = [
        Line2D([0], [0], marker="s", ls="", mec="k", mfc=colors[b], ms=11, label=b.replace("_", " "))
        for b in BUCKET_ORDER
        if any(r["bucket"] == b for r in rows_out)
    ]

    # ---- (b) representative results (solver structure) ----------------------
    picks = [nm for nm, _ in CURATED_PANEL if nm in a_mask]
    headers = ["target", "structure", "generated", "overlay"]
    b_left = 0.365
    gsb = fig.add_gridspec(
        len(picks), 4, left=b_left, right=0.99, top=top_top, bottom=top_bot,
        wspace=0.04, hspace=0.10,
    )
    for r_idx, nm in enumerate(picks):
        gt = mask_of[nm]
        ov, _ = _overlay_from_solver_mask(a_mask[nm], gt, best_alignment, mask_overlay_rgb)
        cells = [fig.add_subplot(gsb[r_idx, c]) for c in range(4)]
        for ax in cells:
            ax.set_xticks([])
            ax.set_yticks([])
        cells[0].imshow(gt, cmap="gray_r", vmin=0.0, vmax=1.0)
        try:
            plot_structure(a_pts[nm], a_quads[nm], None, cells[1])
            cells[1].set_xticks([])
            cells[1].set_yticks([])
        except Exception:
            cells[1].text(0.5, 0.5, "invalid", ha="center", va="center", fontsize=9)
        cells[2].imshow(a_mask[nm], cmap="gray_r", vmin=0.0, vmax=1.0)
        cells[3].imshow(ov)
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

    # ---- (c) sIoU vs. solidity (solver, full width) -------------------------
    gsc = fig.add_gridspec(1, 1, left=0.065, right=0.99, top=sc_top, bottom=sc_bot)
    axc = fig.add_subplot(gsc[0])
    name_to_row = {r["name"]: r for r in rows_out}
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
        if nm not in a_mask or nm not in name_to_row:
            continue
        r = name_to_row[nm]
        gt = mask_of[nm]
        ov, _ = _overlay_from_solver_mask(a_mask[nm], gt, best_alignment, mask_overlay_rgb)
        tgt_rgb = np.repeat(1.0 - gt[..., None], 3, axis=2)
        sep = np.full((gt.shape[0], 4, 3), 0.6, dtype=np.float32)
        composite = np.concatenate([tgt_rgb, sep, ov], axis=1)
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
        handles=bucket_handles + [hole_handle], loc="lower right", fontsize=11,
        framealpha=0.95, ncol=1,
    )
    axc.set_xlabel("solidity")
    axc.set_ylabel("sIoU")
    axc.set_xlim(min(sols) - 0.05, 1.05)
    axc.set_ylim(0.0, 1.02)
    axc.grid(alpha=0.3)

    fig.text(0.010, 0.962, "(a)", fontsize=18, fontweight="bold", ha="left", va="bottom")
    fig.text(b_left - 0.030, 0.962, "(b)", fontsize=18, fontweight="bold", ha="left", va="bottom")
    fig.text(0.010, sc_top + 0.012, "(c)", fontsize=18, fontweight="bold", ha="left", va="bottom")

    out = os.path.join(out_dir, "solver_overview.pdf")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Wrote {out}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Solver baseline vs RL-Kirigami on OOD targets.")
    p.add_argument("--targets", default="outputs/ood/ood_targets.npz")
    p.add_argument("--rl-csv", default="outputs/ood/ood_results.csv")
    p.add_argument("--out-dir", default="outputs/ood")
    p.add_argument("--max-nfev", type=int, default=80, help="least_squares evaluation budget.")
    p.add_argument("--workers", type=int, default=8, help="Parallel solver processes.")
    p.add_argument("--limit", type=int, default=0, help="First N targets only (smoke test).")
    p.add_argument(
        "--artifacts-only",
        action="store_true",
        help="Only solve the curated panel/callout targets and cache their "
        "mask+structure to solver_artifacts.npz (for the figure).",
    )
    p.add_argument(
        "--figure-only",
        action="store_true",
        help="Skip solving; build solver_overview.pdf from cached artifacts + "
        "solver_vs_rl.csv.",
    )
    args = p.parse_args()

    artifacts_path = os.path.join(args.out_dir, "solver_artifacts.npz")

    if args.figure_only:
        _make_solver_figure(
            args.out_dir, args.targets, os.path.join(args.out_dir, "solver_vs_rl.csv"),
            artifacts_path,
        )
        return

    _ensure_paths()
    from experiments.eval_ood_targets import (
        BUCKET_ORDER,
        CURATED_CALLOUTS,
        CURATED_PANEL,
        best_alignment,
        pretty_name,
    )

    blob = np.load(args.targets, allow_pickle=True)
    masks = blob["masks"].astype(np.float32)
    names = list(blob["names"])
    buckets = list(blob["buckets"])

    if args.artifacts_only:
        want = {nm for nm, _ in CURATED_PANEL} | set(CURATED_CALLOUTS)
        idxs = [i for i, nm in enumerate(names) if nm in want]
        tasks = [(names[i], masks[i], args.max_nfev) for i in idxs]
        a_names, a_masks, a_pts, a_quads = [], [], [], []
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as ex:
            for k, (nm, m, pts, q, status, ts) in enumerate(ex.map(_solve_one, tasks), 1):
                print(f"  solved {k}/{len(tasks)}: {pretty_name(nm):28s} ({status}, {ts:.0f}s)")
                if m is not None:
                    a_names.append(nm)
                    a_masks.append(m)
                    a_pts.append(pts)
                    a_quads.append(q)
        os.makedirs(args.out_dir, exist_ok=True)
        np.savez(
            artifacts_path,
            names=np.array(a_names, dtype=object),
            masks=np.array(a_masks, dtype=object),
            pts=np.array(a_pts, dtype=object),
            quads=np.array(a_quads, dtype=object),
        )
        print(f"\nDone in {time.time()-t0:.0f}s. Cached {len(a_names)} -> {artifacts_path}")
        return

    n = masks.shape[0] if not args.limit else min(args.limit, masks.shape[0])
    rl = {}
    with open(args.rl_csv, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rl[row["name"]] = float(row["siou_bestk"])

    tasks = [(names[i], masks[i], args.max_nfev) for i in range(n)]
    solved = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for k, (name, mask, pts, q, status, tsec) in enumerate(ex.map(_solve_one, tasks), 1):
            solved[name] = (mask, status, tsec)
            print(f"  solved {k}/{n}: {pretty_name(name):28s} ({status}, {tsec:.0f}s)")

    rows = []
    for i in range(n):
        name = names[i]
        mask, status, tsec = solved[name]
        s_siou = 0.0 if mask is None else float(best_alignment(mask, masks[i], refine=True)[0])
        rl_v = rl.get(name, float("nan"))
        rows.append(
            {
                "name": name,
                "bucket": buckets[i],
                "siou_rl_bo128": round(rl_v, 4),
                "siou_solver": round(s_siou, 4),
                "delta_rl_minus_solver": round(rl_v - s_siou, 4),
                "solver_status": status,
                "solver_time_s": round(tsec, 2),
            }
        )

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "solver_vs_rl.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def mean(vals):
        vals = [v for v in vals if v == v]
        return sum(vals) / len(vals) if vals else float("nan")

    print(f"\nDone in {time.time()-t0:.0f}s. Per-target: {csv_path}\n")
    hdr = f"{'bucket':<20}{'n':>4}{'RL best-of-128':>18}{'solver (LSQ)':>16}{'RL - solver':>14}"
    print(hdr)
    print("-" * len(hdr))
    for b in BUCKET_ORDER + ["ALL"]:
        grp = rows if b == "ALL" else [r for r in rows if r["bucket"] == b]
        if not grp:
            continue
        mr = mean([r["siou_rl_bo128"] for r in grp])
        ms = mean([r["siou_solver"] for r in grp])
        print(f"{b:<20}{len(grp):>4}{mr:>18.3f}{ms:>16.3f}{mr-ms:>14.3f}")


if __name__ == "__main__":
    main()
