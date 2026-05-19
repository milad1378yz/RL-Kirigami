"""Mine representative failure cases from the OOD evaluation (reviewer R1.5).

Reviewer 1.5 asks for an appendix that shows where the generator fails: where
overlap occurs, where the target is poorly matched, and where the feasibility
filters reject a design. This script does not re-sample the model. It reuses the
artifacts already written by ``experiments.eval_ood_targets``:

  - ``outputs/ood/ood_targets.npz``     : target masks + metadata,
  - ``outputs/ood/ood_pred_fields.npz`` : the best-of-K ratio field the method
                                          actually chose for each target,
  - ``outputs/ood/ood_results.csv``     : per-target sIoU.

Each cached best-of-K ratio field is decoded again through the same geometry
simulator (``render_structure_mask_and_metrics``) so the metrics printed next to
a panel are self-consistent with the structure drawn in it. Showing the
*chosen* best-of-K output (not a cherry-picked bad sample) is the honest "here
is where it still fails after best-of-K selection" statement.

Every failing target is assigned exactly one *primary* failure category by an
ordered rule, so the appendix panel has one distinct shape per row and no shape
illustrates two categories at once:

  build_failed        the simulator could not decode the field at all.
  overlap             decoded parallelogram quads physically collide
                      (overlap_ratio > --overlap-tau); a feasibility-filter
                      reject in the paper's pipeline.
  invalid_quads       degenerate / inverted quads but overlap below the cut.
  hole_unrepresentable target has an interior hole; the compact
                      parallelogram-quad decoder provably cannot open one, so
                      the silhouette match is capped.
  range_clipping      the prior pushed enough ratios outside [x_min, x_max]
                      that the clamp distorted the design
                      (clipped_fraction >= --clip-tau, default 0.10). A few
                      per-cent of clipping is normal and is *not* counted, so
                      this category honestly reports "none" when no design is
                      actually distorted by the clamp.
  poor_match          geometry is valid and hole-free, yet sIoU is below
                      --success-threshold: a boundary-complexity miss.

Within a category, candidates are ranked worst-first by a severity key
(overlap_ratio, invalid_quad_count, then -sIoU) and the worst
``--per-category`` are kept.

Outputs (default ``outputs/failures/``):
  - ``failure_cases.csv``       : the selected shapes, their primary category,
                                  severity metrics, and a one-line reason.
  - ``failure_case_panel.pdf``  : appendix figure, one row per selected case:
                                  Target | Compact rectangle | Generated
                                  (deployed + aligned target) | Overlay |
                                  Ratio field.
  - ``failure_case_panel.png``  : raster preview of the same figure.
  - ``cases/<id>.png``          : the same row saved on its own, for slides.
  - ``failure_minimal.pdf``     : the minimal paper figure -- one row per
                                  failure mode (overlap, hole not
                                  representable, poor silhouette match), in the
                                  paper's panel-(b) style (Target | Compact
                                  rectangle | Generated).
  - ``failure_minimal.png``     : raster preview of the minimal figure.

Run from the repo root, after ``experiments.eval_ood_targets``:
    python -m experiments.mine_failure_cases
"""

import argparse
import csv
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data_generator.utils import (  # noqa: E402
    build_geometry_context,
    mask_overlay_rgb,
    render_structure_mask_and_metrics,
)
from data_generator.visualization import plot_x_matrix_structure  # noqa: E402
from experiments.eval_ood_targets import best_alignment, pretty_name  # noqa: E402
from kirigami_training.data import prepare_training_config  # noqa: E402
from kirigami_training.utils import load_config, select_training_config  # noqa: E402

# Ordered primary-category rule. The first matching row wins, so a target that
# both overlaps and has a hole is reported as the (more fundamental) overlap
# feasibility reject, never double-counted.
CATEGORY_ORDER = [
    "build_failed",
    "overlap",
    "invalid_quads",
    "hole_unrepresentable",
    "range_clipping",
    "poor_match",
]
CATEGORY_LABEL = {
    "build_failed": "Build failed",
    "overlap": "Overlap (quads collide)",
    "invalid_quads": "Invalid quads",
    "hole_unrepresentable": "Hole not representable",
    "range_clipping": "Ratio range clipping",
    "poor_match": "Poor silhouette match",
}


def classify_primary(metrics: dict, siou: float, target_hole_count: int,
                      *, overlap_tau: float, clip_tau: float,
                      success_threshold: float) -> str:
    """Single primary failure category, ordered (see CATEGORY_ORDER)."""
    if not bool(metrics.get("ok", False)):
        return "build_failed"
    if float(metrics.get("overlap_ratio", 0.0) or 0.0) > overlap_tau:
        return "overlap"
    if int(metrics.get("invalid_quad_count", 0) or 0) > 0:
        return "invalid_quads"
    if target_hole_count > 0 and siou < success_threshold:
        return "hole_unrepresentable"
    if float(metrics.get("clipped_fraction", 0.0) or 0.0) >= clip_tau and siou < success_threshold:
        return "range_clipping"
    if siou < success_threshold:
        return "poor_match"
    return "ok"


def severity_key(row: dict) -> tuple:
    """Worst-first sort within a category: more overlap, more invalid quads,
    then lower sIoU."""
    return (
        -float(row["overlap_ratio"]),
        -int(row["invalid_quad_count"]),
        float(row["siou_bestk"]),
    )


def reason_text(cat: str, row: dict) -> str:
    """One-line cause. No commas -- keeps the CSV column readable in a
    spreadsheet / LaTeX table without quoting noise."""
    if cat == "build_failed":
        return "simulator could not decode the chosen ratio field"
    if cat == "overlap":
        return (
            f"decoded quads overlap (overlap_ratio={row['overlap_ratio']:.3f}; "
            f"{row['invalid_quad_count']} invalid quads) -- feasibility filter rejects it"
        )
    if cat == "invalid_quads":
        return f"{row['invalid_quad_count']} degenerate/inverted quads in the decoded sheet"
    if cat == "hole_unrepresentable":
        return (
            "target has an interior hole the compact parallelogram-quad decoder "
            f"cannot open -- sIoU capped at {row['siou_bestk']:.3f}"
        )
    if cat == "range_clipping":
        return (
            f"{row['clipped_fraction']*100:.0f} percent of ratios clipped to "
            "[x_min x_max] -- clamp distorted the design"
        )
    return (
        f"valid hole-free geometry but sIoU={row['siou_bestk']:.3f} "
        "below threshold -- boundary too complex for the quad grid"
    )


def _load_cached(out_ood: str):
    targets = np.load(os.path.join(out_ood, "ood_targets.npz"), allow_pickle=True)
    fields = np.load(os.path.join(out_ood, "ood_pred_fields.npz"), allow_pickle=True)
    with open(os.path.join(out_ood, "ood_results.csv"), encoding="utf-8") as fh:
        csv_rows = {r["name"]: r for r in csv.DictReader(fh)}
    mask_by_name = dict(zip(list(targets["names"]), targets["masks"].astype(np.float32)))
    field_by_name = dict(zip(list(fields["names"]), fields["pred_x"].astype(np.float32)))
    meta = {
        n: {
            "bucket": str(b),
            "solidity": float(s),
            "hole_count": int(h),
        }
        for n, b, s, h in zip(
            list(targets["names"]),
            list(targets["buckets"]),
            targets["solidity"],
            targets["hole_count"],
        )
    }
    return mask_by_name, field_by_name, csv_rows, meta


def _draw_row(axes, name, gt_mask, pred_x, context, rows, cols, x_min, x_max,
              cat, row):
    """One appendix row: Target | Compact rectangle | Generated | Overlay |
    Ratio field. Mirrors the rendering in eval_ood_targets._make_figures."""
    import matplotlib.pyplot as plt

    pred_mask, _, _, _ = render_structure_mask_and_metrics(
        rows, cols, pred_x, context, gt_mask.shape[0], gt_mask.shape[1],
        x_min=x_min, x_max=x_max,
    )
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].imshow(gt_mask, cmap="gray_r", vmin=0.0, vmax=1.0)

    try:  # compact rectangle = phi=pi pose of the decoded cut sheet
        plot_x_matrix_structure(
            axes[1], pred_x, context, phi=np.pi,
            x_min=x_min, x_max=x_max, normalize_phi=np.pi,
        )
    except Exception:
        axes[1].text(0.5, 0.5, "invalid", ha="center", va="center", fontsize=9)

    siou, aligned_pred = best_alignment(pred_mask, gt_mask, refine=True)
    try:  # deployed structure with the aligned target silhouette overlaid
        _, aligned_target = best_alignment(gt_mask, pred_mask, refine=True)
        plot_x_matrix_structure(
            axes[2], pred_x, context, mask_2d=aligned_target.astype(np.float32),
            x_min=x_min, x_max=x_max, normalize_phi=None,
        )
    except Exception:
        axes[2].text(0.5, 0.5, "invalid", ha="center", va="center", fontsize=9)

    overlay = mask_overlay_rgb(aligned_pred.astype(np.float32), gt_mask).astype(np.float32)
    overlay[~np.any(overlay > 0, axis=2)] = 1.0
    axes[3].imshow(overlay)

    im = axes[4].imshow(pred_x, cmap="viridis", vmin=x_min, vmax=x_max)
    for ax in axes:  # structure plotters clear ticks; re-assert
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].set_ylabel(
        f"{pretty_name(name)}\n[{CATEGORY_LABEL[cat]}]",
        fontsize=8.5, fontweight="bold", rotation=0, ha="right", va="center",
        labelpad=58,
    )
    return im, siou


def make_panel(selected, mask_by_name, field_by_name, context, rows, cols,
               x_min, x_max, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman", "Liberation Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
        }
    )
    headers = ["Target", "Compact\nrectangle", "Generated", "Overlay", "Ratio field"]
    n = len(selected)
    fig, axes = plt.subplots(n, 5, figsize=(12.2, 2.05 * n), squeeze=False)
    last_im = None
    for r_idx, sel in enumerate(selected):
        name = sel["name"]
        last_im, _ = _draw_row(
            axes[r_idx], name, mask_by_name[name], field_by_name[name],
            context, rows, cols, x_min, x_max, sel["primary_category"], sel,
        )
        if r_idx == 0:
            for c, htxt in enumerate(headers):
                axes[0][c].set_title(htxt, fontsize=9, fontweight="bold",
                                     linespacing=0.95, pad=4)

    fig.subplots_adjust(left=0.205, right=0.935, top=0.95, bottom=0.045,
                        wspace=0.08, hspace=0.18)
    if last_im is not None:
        cax = fig.add_axes([0.945, 0.1, 0.012, 0.78])
        cb = fig.colorbar(last_im, cax=cax)
        cb.set_label("ratio (x)", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    pdf_path = os.path.join(out_dir, "failure_case_panel.pdf")
    png_path = os.path.join(out_dir, "failure_case_panel.png")
    fig.savefig(pdf_path, dpi=200)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    # Per-case standalone rows (for slides / response-to-reviewers letter).
    cases_dir = os.path.join(out_dir, "cases")
    os.makedirs(cases_dir, exist_ok=True)
    for sel in selected:
        name = sel["name"]
        f1, ax1 = plt.subplots(1, 5, figsize=(12.2, 2.4), squeeze=False)
        im, _ = _draw_row(
            ax1[0], name, mask_by_name[name], field_by_name[name],
            context, rows, cols, x_min, x_max, sel["primary_category"], sel,
        )
        for c, htxt in enumerate(headers):
            ax1[0][c].set_title(htxt, fontsize=9, fontweight="bold",
                               linespacing=0.95, pad=4)
        f1.subplots_adjust(left=0.205, right=0.985, top=0.86, bottom=0.04,
                          wspace=0.08)
        slug = name.replace("/", "__")
        f1.savefig(os.path.join(cases_dir, f"{slug}.png"), dpi=150)
        plt.close(f1)
    return pdf_path, png_path, cases_dir


# The three failure modes shown in the minimal paper figure, in reading order.
MINIMAL_CATEGORIES = ["overlap", "hole_unrepresentable", "poor_match"]
# Short, plain-word row label + the metric that names the failure.
MINIMAL_LABEL = {
    "overlap": "Overlap",
    "hole_unrepresentable": "Hole not\nrepresentable",
    "poor_match": "Poor\nmatch",
}


def select_minimal(classified: list[dict]) -> list[dict]:
    """One representative per failure mode for the minimal paper figure.

    Within a mode the worst case by ``severity_key`` is taken. For ``overlap``
    a hole-free target is preferred when available, so the row isolates the
    overlap mechanism instead of also showing a hole the decoder cannot open.
    """
    picks: list[dict] = []
    for cat in MINIMAL_CATEGORIES:
        grp = sorted([c for c in classified if c["primary_category"] == cat],
                     key=severity_key)
        if not grp:
            continue
        if cat == "overlap":
            hole_free = [g for g in grp if g["target_hole_count"] == 0]
            grp = hole_free or grp
        pick = dict(grp[0])
        pick["reason"] = reason_text(cat, pick)
        picks.append(pick)
    return picks


def make_minimal_panel(picks, mask_by_name, field_by_name, context, rows, cols,
                       x_min, x_max, out_dir):
    """Compact figure in the paper's panel-(b) style: one row per failure mode,
    columns Target | Compact rectangle | Generated (aligned target overlaid)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman", "Liberation Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 9,
        }
    )
    headers = ["Target", "Compact rectangle", "Generated"]
    n = len(picks)
    fig, axes = plt.subplots(n, 3, figsize=(6.0, 1.95 * n), squeeze=False)
    for r_idx, sel in enumerate(picks):
        name = sel["name"]
        gt_mask = mask_by_name[name]
        pred_x = field_by_name[name]
        ax = axes[r_idx]
        for a in ax:
            a.set_xticks([])
            a.set_yticks([])

        pred_mask, _, _, _ = render_structure_mask_and_metrics(
            rows, cols, pred_x, context, gt_mask.shape[0], gt_mask.shape[1],
            x_min=x_min, x_max=x_max,
        )
        ax[0].imshow(gt_mask, cmap="gray_r", vmin=0.0, vmax=1.0)
        try:
            plot_x_matrix_structure(
                ax[1], pred_x, context, phi=np.pi,
                x_min=x_min, x_max=x_max, normalize_phi=np.pi,
            )
        except Exception:
            ax[1].text(0.5, 0.5, "invalid", ha="center", va="center", fontsize=9)
        try:
            _, aligned_target = best_alignment(gt_mask, pred_mask, refine=True)
            plot_x_matrix_structure(
                ax[2], pred_x, context, mask_2d=aligned_target.astype(np.float32),
                x_min=x_min, x_max=x_max, normalize_phi=None,
            )
        except Exception:
            ax[2].text(0.5, 0.5, "invalid", ha="center", va="center", fontsize=9)
        for a in ax:
            a.set_xticks([])
            a.set_yticks([])

        cat = sel["primary_category"]
        if cat == "overlap":
            metric = (f"$r_{{\\mathrm{{ov}}}}$={sel['overlap_ratio']:.2f}\n"
                      f"{sel['invalid_quad_count']} invalid quads")
        else:
            metric = f"sIoU {sel['siou_bestk']:.2f}"
        ax[0].set_ylabel(
            f"{MINIMAL_LABEL[cat]}\n({metric})",
            fontsize=9, fontweight="bold", rotation=0, ha="right", va="center",
            labelpad=14,
        )
        if r_idx == 0:
            for c, htxt in enumerate(headers):
                ax[c].set_title(htxt, fontsize=9, fontweight="bold", pad=4)

    fig.subplots_adjust(left=0.235, right=0.99, top=0.93, bottom=0.02,
                        wspace=0.06, hspace=0.10)
    pdf_path = os.path.join(out_dir, "failure_minimal.pdf")
    png_path = os.path.join(out_dir, "failure_minimal.png")
    fig.savefig(pdf_path, dpi=200)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return pdf_path, png_path


def main() -> None:
    p = argparse.ArgumentParser(description="Mine OOD failure cases (reviewer R1.5).")
    p.add_argument("--config", default="configs/training.yaml")
    p.add_argument("--training-key", default="fm_training")
    p.add_argument("--ood-dir", default="outputs/ood",
                   help="Directory with the cached eval_ood_targets artifacts.")
    p.add_argument("--out-dir", default="outputs/failures")
    p.add_argument("--per-category", type=int, default=2,
                   help="Worst-N failing shapes to keep per category.")
    p.add_argument("--overlap-tau", type=float, default=0.02,
                   help="overlap_ratio above this is an overlap feasibility reject.")
    p.add_argument("--clip-tau", type=float, default=0.10,
                   help="clipped_fraction at/above this counts as range clipping; "
                        "a few per-cent of clipping is normal and ignored.")
    p.add_argument("--success-threshold", type=float, default=0.5,
                   help="sIoU below this counts as a poor match.")
    args = p.parse_args()

    config = prepare_training_config(
        select_training_config(load_config(args.config), args.training_key)
    )
    data_cfg = config["data"]
    rows, cols = int(data_cfg["grid_rows"]), int(data_cfg["grid_cols"])
    x_min, x_max = float(data_cfg["x_min"]), float(data_cfg["x_max"])
    context = build_geometry_context(rows, cols)

    mask_by_name, field_by_name, csv_rows, meta = _load_cached(args.ood_dir)

    # Re-decode every cached best-of-K field so the panel and the printed
    # numbers come from one pass (the CSV sIoU is kept for cross-checking).
    classified: list[dict] = []
    for name, pred_x in field_by_name.items():
        gt = mask_by_name[name]
        pred_mask, metrics, _, _ = render_structure_mask_and_metrics(
            rows, cols, pred_x, context, gt.shape[0], gt.shape[1],
            x_min=x_min, x_max=x_max,
        )
        siou, _ = best_alignment(pred_mask, gt, refine=True)
        hole_ct = meta[name]["hole_count"]
        cat = classify_primary(
            metrics, siou, hole_ct,
            overlap_tau=args.overlap_tau,
            clip_tau=args.clip_tau,
            success_threshold=args.success_threshold,
        )
        if cat == "ok":
            continue
        classified.append(
            {
                "name": name,
                "bucket": meta[name]["bucket"],
                "solidity": round(meta[name]["solidity"], 4),
                "target_hole_count": hole_ct,
                "primary_category": cat,
                "siou_bestk": round(float(siou), 4),
                "siou_csv": float(csv_rows.get(name, {}).get("siou_bestk", "nan") or "nan"),
                "build_ok": bool(metrics.get("ok", False)),
                "invalid_quad_count": int(metrics.get("invalid_quad_count", 0) or 0),
                "overlap_ratio": round(float(metrics.get("overlap_ratio", 0.0) or 0.0), 4),
                "clipped_fraction": round(float(metrics.get("clipped_fraction", 0.0) or 0.0), 4),
                "hole_count_pred": int(metrics.get("hole_count", 0) or 0),
            }
        )

    selected: list[dict] = []
    print(f"Failing OOD targets: {len(classified)} / {len(field_by_name)}")
    for cat in CATEGORY_ORDER:
        grp = sorted([c for c in classified if c["primary_category"] == cat],
                     key=severity_key)
        if not grp:
            print(f"  {cat:<22} : none")
            continue
        keep = grp[: args.per_category]
        print(f"  {cat:<22} : {len(grp)} found -> keeping "
              f"{', '.join(pretty_name(k['name']) for k in keep)}")
        for k in keep:
            k["reason"] = reason_text(cat, k)
            selected.append(k)

    if not selected:
        raise SystemExit("No failures found -- nothing to mine.")

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "failure_cases.csv")
    fieldnames = [
        "name", "primary_category", "reason", "bucket", "solidity",
        "target_hole_count", "siou_bestk", "build_ok", "invalid_quad_count",
        "overlap_ratio", "clipped_fraction", "hole_count_pred",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(selected)

    pdf_path, png_path, cases_dir = make_panel(
        selected, mask_by_name, field_by_name, context, rows, cols,
        x_min, x_max, args.out_dir,
    )

    picks = select_minimal(classified)
    min_pdf, min_png = make_minimal_panel(
        picks, mask_by_name, field_by_name, context, rows, cols,
        x_min, x_max, args.out_dir,
    )

    print(f"\nSelected {len(selected)} representative failures.")
    print(f"  table       : {csv_path}")
    print(f"  full panel  : {pdf_path}")
    print(f"  full png    : {png_path}")
    print(f"  cases       : {cases_dir}/")
    print(f"  minimal fig : {min_pdf} (rows: "
          f"{', '.join(pretty_name(p['name']) for p in picks)})")
    print(f"  minimal png : {min_png}")


if __name__ == "__main__":
    main()
