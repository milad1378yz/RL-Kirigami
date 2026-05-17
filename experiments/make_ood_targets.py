"""Build out-of-distribution (OOD) target silhouettes for reviewer Priority 1.

The paper's test masks come from the same feasible-ratio-field generator as the
training set, so they do not establish performance on arbitrary user-specified
silhouettes (Reviewer 1.1 / 2.3). This script produces target masks from a
generative process *unrelated* to the kirigami ratio-field generator, organised
into four buckets and parametrised along a continuous difficulty axis:

  - convex            : ellipses, regular polygons, rounded rects, superellipses
                        (expected to succeed -- the easy controls).
  - concave           : stars, crescents, dumbbells, plus/L/T, astroid, swept by
                        a concavity parameter (the graceful-degradation core).
  - topological_limit : rings/annuli and spirals -- the compact parallelogram
                        quad decoder cannot represent interior holes, so these
                        are documented, explainable failures.
  - literal           : the exact shapes the reviewers named -- letters C/U/S/A,
                        a ring (O) and a free-form "hand-drawn" doodle. Kept
                        small and tagged so the reply can point to them directly.

Each target stores a ``solidity`` scalar (mask area / convex-hull area; 1.0 for
convex, -> 0 as concavity/holes grow). Plotting sIoU vs. solidity, coloured by
bucket and hole count, gives a single rigorous "where it works / where it
breaks" figure instead of cherry-picked anecdotes.

Masks are float32 ``{0, 1}`` arrays at the dataset resolution (``img_h`` x
``img_w`` from ``configs/data_generator.yaml``), so they drop straight into the
existing evaluation path. sIoU (``data_generator.utils.mask_siou``) aligns
prediction to target over rotation / scale / translation, so this is a fair
shape-only OOD test and absolute placement does not matter.

Outputs (default ``outputs/ood/``):
  - ``ood_targets.npz``        : masks + per-target metadata (load with
                                 ``np.load(path, allow_pickle=True)``;
                                 ``masks`` is float32 [N, H, W]).
  - ``ood_targets_index.csv``  : human-readable catalogue.
  - ``ood_preview.png``        : grid preview, grouped by bucket and difficulty.

Run from the repo root:
    python -m experiments.make_ood_targets
"""

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# Allow ``python experiments/make_ood_targets.py`` as well as ``-m``.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import yaml  # noqa: E402

from data_generator.utils import mask_hole_metrics  # noqa: E402

BUCKET_ORDER = ["convex", "concave", "topological_limit", "literal"]
BUCKET_COLORS = {
    "convex": "#2ca02c",
    "concave": "#1f77b4",
    "topological_limit": "#d62728",
    "literal": "#9467bd",
}


# --------------------------------------------------------------------------- #
# Geometry primitives. Everything is authored in an arbitrary 2D frame; a final
# fit step normalises each shape into the unit square before rasterisation.
# --------------------------------------------------------------------------- #
@dataclass
class Primitive:
    """One drawable element. ``poly`` = filled polygon, ``stroke`` = thick path."""

    kind: str  # "poly" | "stroke"
    op: str  # "add" | "sub"
    pts: np.ndarray  # (M, 2) frame coordinates
    r: float = 0.0  # stroke half-width, frame units (ignored for "poly")


@dataclass
class Shape:
    name: str
    bucket: str
    family: str
    params: dict
    prims: list[Primitive] = field(default_factory=list)


def _circle(cx: float, cy: float, r: float, n: int = 256) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1)


def _rect(cx: float, cy: float, w: float, h: float) -> np.ndarray:
    hw, hh = 0.5 * w, 0.5 * h
    return np.array(
        [[cx - hw, cy - hh], [cx + hw, cy - hh], [cx + hw, cy + hh], [cx - hw, cy + hh]],
        dtype=np.float64,
    )


def _rounded_rect(cx: float, cy: float, w: float, h: float, rad: float, n: int = 24) -> np.ndarray:
    hw, hh = 0.5 * w, 0.5 * h
    rad = min(rad, hw, hh)
    pts = []
    # corner centres: (sign_x, sign_y, start_angle)
    corners = [
        (hw - rad, hh - rad, 0.0),
        (-(hw - rad), hh - rad, 0.5 * math.pi),
        (-(hw - rad), -(hh - rad), math.pi),
        (hw - rad, -(hh - rad), 1.5 * math.pi),
    ]
    for ox, oy, a0 in corners:
        a = np.linspace(a0, a0 + 0.5 * math.pi, n)
        pts.append(np.stack([cx + ox + rad * np.cos(a), cy + oy + rad * np.sin(a)], axis=1))
    return np.concatenate(pts, axis=0)


def _regular_polygon(n: int, rot: float = 0.0) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False) + rot
    return np.stack([np.cos(t), np.sin(t)], axis=1)


def _star(n_points: int, inner_ratio: float) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * math.pi, 2 * n_points, endpoint=False) + 0.5 * math.pi
    radii = np.where(np.arange(2 * n_points) % 2 == 0, 1.0, float(inner_ratio))
    return np.stack([radii * np.cos(t), radii * np.sin(t)], axis=1)


def _superellipse(exponent: float, n: int = 400) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    ct, st = np.cos(t), np.sin(t)
    p = 2.0 / float(exponent)
    x = np.sign(ct) * np.abs(ct) ** p
    y = np.sign(st) * np.abs(st) ** p
    return np.stack([x, y], axis=1)


def _arc(cx: float, cy: float, r: float, a0: float, a1: float, n: int = 160) -> np.ndarray:
    a = np.linspace(a0, a1, n)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1)


# --------------------------------------------------------------------------- #
# Shape catalogue. Each builder returns a list[Shape].
# --------------------------------------------------------------------------- #
def _convex_shapes() -> list[Shape]:
    out: list[Shape] = []
    for ar in (1.0, 1.5, 2.0, 3.0):
        c = _circle(0.0, 0.0, 1.0)
        c = c * np.array([1.0, 1.0 / ar])
        out.append(
            Shape(
                f"convex/ellipse_ar{ar:.2f}",
                "convex",
                "ellipse",
                {"aspect_ratio": ar},
                [Primitive("poly", "add", c)],
            )
        )
    for n in (3, 4, 5, 6, 8):
        out.append(
            Shape(
                f"convex/regular_polygon_n{n}",
                "convex",
                "regular_polygon",
                {"n_sides": n},
                [Primitive("poly", "add", _regular_polygon(n, rot=0.5 * math.pi))],
            )
        )
    for aspect, rad in ((1.0, 0.25), (2.0, 0.20), (1.5, 0.35)):
        out.append(
            Shape(
                f"convex/rounded_rect_a{aspect:.1f}_r{rad:.2f}",
                "convex",
                "rounded_rect",
                {"aspect": aspect, "corner_radius": rad},
                [Primitive("poly", "add", _rounded_rect(0.0, 0.0, 2.0 * aspect, 2.0, rad))],
            )
        )
    for exp in (2.0, 4.0, 8.0):
        out.append(
            Shape(
                f"convex/superellipse_e{exp:.1f}",
                "convex",
                "superellipse",
                {"exponent": exp},
                [Primitive("poly", "add", _superellipse(exp))],
            )
        )
    return out


def _concave_shapes() -> list[Shape]:
    out: list[Shape] = []
    for inner in (0.65, 0.50, 0.38, 0.28, 0.18):
        out.append(
            Shape(
                f"concave/star5_in{inner:.2f}",
                "concave",
                "star",
                {"n_points": 5, "inner_ratio": inner},
                [Primitive("poly", "add", _star(5, inner))],
            )
        )
    for inner in (0.60, 0.45, 0.30):
        out.append(
            Shape(
                f"concave/star6_in{inner:.2f}",
                "concave",
                "star",
                {"n_points": 6, "inner_ratio": inner},
                [Primitive("poly", "add", _star(6, inner))],
            )
        )
    for off in (0.35, 0.55, 0.75, 0.95):
        out.append(
            Shape(
                f"concave/crescent_off{off:.2f}",
                "concave",
                "crescent",
                {"offset": off},
                [
                    Primitive("poly", "add", _circle(0.0, 0.0, 1.0)),
                    Primitive("poly", "sub", _circle(off, 0.0, 1.0)),
                ],
            )
        )
    for waist in (0.45, 0.30, 0.18, 0.10):
        out.append(
            Shape(
                f"concave/dumbbell_w{waist:.2f}",
                "concave",
                "dumbbell",
                {"waist": waist},
                [
                    Primitive("poly", "add", _circle(-0.62, 0.0, 0.45)),
                    Primitive("poly", "add", _circle(0.62, 0.0, 0.45)),
                    Primitive("poly", "add", _rect(0.0, 0.0, 1.3, waist)),
                ],
            )
        )
    for arm in (0.55, 0.40, 0.28):
        out.append(
            Shape(
                f"concave/plus_arm{arm:.2f}",
                "concave",
                "plus",
                {"arm_width": arm},
                [
                    Primitive("poly", "add", _rect(0.0, 0.0, 2.0, arm)),
                    Primitive("poly", "add", _rect(0.0, 0.0, arm, 2.0)),
                ],
            )
        )
    for notch in (0.40, 0.60):
        out.append(
            Shape(
                f"concave/L_notch{notch:.2f}",
                "concave",
                "L_shape",
                {"notch": notch},
                [
                    Primitive("poly", "add", _rect(0.0, 0.0, 2.0, 2.0)),
                    Primitive(
                        "poly", "sub", _rect(1.0 - notch, 1.0 - notch, 2.0 * notch, 2.0 * notch)
                    ),
                ],
            )
        )
    out.append(
        Shape(
            "concave/T_shape",
            "concave",
            "T_shape",
            {},
            [
                Primitive("poly", "add", _rect(0.0, 0.7, 2.0, 0.6)),
                Primitive("poly", "add", _rect(0.0, -0.3, 0.55, 1.4)),
            ],
        )
    )
    out.append(
        Shape(
            "concave/astroid",
            "concave",
            "superellipse",
            {"exponent": 0.6},
            [Primitive("poly", "add", _superellipse(0.6))],
        )
    )
    return out


def _topological_limit_shapes() -> list[Shape]:
    out: list[Shape] = []
    for inner in (0.30, 0.50, 0.70):
        out.append(
            Shape(
                f"topological_limit/annulus_in{inner:.2f}",
                "topological_limit",
                "annulus",
                {"inner_ratio": inner},
                [
                    Primitive("poly", "add", _circle(0.0, 0.0, 1.0)),
                    Primitive("poly", "sub", _circle(0.0, 0.0, inner)),
                ],
            )
        )
    for turns in (1.0, 1.5):
        t = np.linspace(0.0, turns * 2.0 * math.pi, 400)
        rr = 0.08 + 0.92 * (t / t.max())
        path = np.stack([rr * np.cos(t), rr * np.sin(t)], axis=1)
        out.append(
            Shape(
                f"topological_limit/spiral_t{turns:.1f}",
                "topological_limit",
                "spiral",
                {"turns": turns},
                [Primitive("stroke", "add", path, r=0.10)],
            )
        )
    return out


def _letter_C() -> list[Primitive]:
    return [
        Primitive("stroke", "add", _arc(0.0, 0.0, 0.6, math.radians(55), math.radians(305)), r=0.16)
    ]


def _letter_U() -> list[Primitive]:
    left = np.array([[-0.5, 0.7], [-0.5, -0.1]])
    bottom = _arc(0.0, -0.1, 0.5, math.pi, 2.0 * math.pi)
    right = np.array([[0.5, -0.1], [0.5, 0.7]])
    path = np.concatenate([left, bottom, right], axis=0)
    return [Primitive("stroke", "add", path, r=0.16)]


def _letter_S() -> list[Primitive]:
    # Single smooth sigmoid spine: one open polyline -> recognisably "S" and
    # provably hole-free (a non-self-intersecting stroke encloses nothing).
    t = np.linspace(0.0, 1.0, 240)
    path = np.stack([0.42 * np.sin(2.0 * math.pi * t), 0.9 - 1.8 * t], axis=1)
    return [Primitive("stroke", "add", path, r=0.15)]


def _letter_A() -> list[Primitive]:
    apex = np.array([0.0, 0.95])
    left = np.array([apex, [-0.62, -0.85]])
    right = np.array([apex, [0.62, -0.85]])
    bar = np.array([[-0.34, -0.05], [0.34, -0.05]])
    return [
        Primitive("stroke", "add", left, r=0.16),
        Primitive("stroke", "add", right, r=0.16),
        Primitive("stroke", "add", bar, r=0.13),
    ]


def _doodle(rng: np.random.Generator) -> list[Primitive]:
    t = np.linspace(0.0, 2.0 * math.pi, 360, endpoint=False)
    r = np.ones_like(t)
    for k in (2, 3, 4, 5):
        r += rng.uniform(0.08, 0.22) * np.cos(k * t + rng.uniform(0.0, 2.0 * math.pi))
    r = np.clip(r, 0.25, None)
    pts = np.stack([r * np.cos(t), r * np.sin(t)], axis=1)
    return [Primitive("poly", "add", pts)]


def _literal_shapes(rng: np.random.Generator) -> list[Shape]:
    return [
        Shape("literal/letter_C", "literal", "letter", {"glyph": "C"}, _letter_C()),
        Shape("literal/letter_U", "literal", "letter", {"glyph": "U"}, _letter_U()),
        Shape("literal/letter_S", "literal", "letter", {"glyph": "S"}, _letter_S()),
        Shape("literal/letter_A", "literal", "letter", {"glyph": "A"}, _letter_A()),
        Shape(
            "literal/ring_O",
            "literal",
            "letter",
            {"glyph": "O"},
            [
                Primitive("poly", "add", _circle(0.0, 0.0, 1.0)),
                Primitive("poly", "sub", _circle(0.0, 0.0, 0.55)),
            ],
        ),
        Shape("literal/doodle", "literal", "freeform", {"hand_drawn": True}, _doodle(rng)),
    ]


def build_catalogue(seed: int) -> list[Shape]:
    rng = np.random.default_rng(seed)
    shapes = (
        _convex_shapes() + _concave_shapes() + _topological_limit_shapes() + _literal_shapes(rng)
    )
    return shapes


# --------------------------------------------------------------------------- #
# Rasterisation: fit a shape into the unit square, then map to the pixel grid
# with a margin. Binary float32 {0, 1}, same convention as the dataset masks.
# --------------------------------------------------------------------------- #
def _fit_to_unit(prims: list[Primitive]) -> list[Primitive]:
    lo = np.array([np.inf, np.inf])
    hi = np.array([-np.inf, -np.inf])
    for p in prims:
        pad = p.r if p.kind == "stroke" else 0.0
        lo = np.minimum(lo, p.pts.min(axis=0) - pad)
        hi = np.maximum(hi, p.pts.max(axis=0) + pad)
    span = float(np.max(hi - lo))
    if span <= 0.0:
        span = 1.0
    scale = 1.0 / span
    center = 0.5 * (lo + hi)
    out: list[Primitive] = []
    for p in prims:
        pts = (p.pts - center) * scale + 0.5
        out.append(Primitive(p.kind, p.op, pts, r=p.r * scale))
    return out


def _poly_mask(verts_px: np.ndarray, grid: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    from matplotlib.path import Path

    inside = Path(verts_px).contains_points(grid)
    return inside.reshape(shape)


def _stroke_mask(path_px: np.ndarray, r_px: float, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Pixels within ``r_px`` of the polyline ``path_px``."""
    g = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)
    best = np.full(g.shape[0], np.inf)
    for a, b in zip(path_px[:-1], path_px[1:]):
        ab = b - a
        denom = float(ab @ ab)
        if denom < 1e-12:
            d = np.linalg.norm(g - a, axis=1)
        else:
            t = np.clip(((g - a) @ ab) / denom, 0.0, 1.0)
            proj = a + t[:, None] * ab
            d = np.linalg.norm(g - proj, axis=1)
        np.minimum(best, d, out=best)
    return (best <= r_px).reshape(gy.shape)


def rasterize_shape(prims: list[Primitive], h: int, w: int, margin: int) -> np.ndarray:
    prims = _fit_to_unit(prims)
    side = min(h, w)
    inner = max(1.0, float(side - 2 * margin))
    off_x = 0.5 * (w - inner)
    off_y = 0.5 * (h - inner)

    def to_px(pts: np.ndarray) -> np.ndarray:
        x = off_x + pts[:, 0] * inner
        y = off_y + (1.0 - pts[:, 1]) * inner  # flip y so frame-up = image-up
        return np.stack([x, y], axis=1)

    xv, yv = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    grid = np.column_stack([xv.ravel(), yv.ravel()])

    mask = np.zeros((h, w), dtype=bool)
    for p in prims:
        if p.kind == "poly":
            m = _poly_mask(to_px(p.pts), grid, (h, w))
        else:
            m = _stroke_mask(to_px(p.pts), p.r * inner, xv, yv)
        mask = np.logical_and(mask, ~m) if p.op == "sub" else np.logical_or(mask, m)
    return mask.astype(np.float32)


# --------------------------------------------------------------------------- #
# Difficulty scalar: solidity = mask area / convex-hull area. 1.0 for convex,
# decreasing with concavity and interior holes.
# --------------------------------------------------------------------------- #
def compute_solidity(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask >= 0.5)
    area = float(xs.size)
    if area == 0:
        return 0.0
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    try:
        from scipy.spatial import ConvexHull

        hull_area = float(ConvexHull(pts).volume)  # 2D: .volume is the area
    except Exception:
        hull_area = float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
    if hull_area <= 0.0:
        return 0.0
    return max(0.0, min(1.0, area / hull_area))


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #
def _load_resolution(config_path: str) -> tuple[int, int]:
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    return int(cfg.get("img_h", 128)), int(cfg.get("img_w", 128))


def save_preview(shapes: list[Shape], masks: list[np.ndarray], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from matplotlib.patches import Patch

    order = sorted(
        range(len(shapes)),
        key=lambda i: (BUCKET_ORDER.index(shapes[i].bucket), -compute_solidity(masks[i])),
    )
    n = len(order)
    cols = 10
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(1.4 * cols, 1.4 * rows))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    for slot, i in enumerate(order):
        ax = axes[slot // cols][slot % cols]
        ax.imshow(masks[i], cmap="gray_r", vmin=0.0, vmax=1.0)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor(BUCKET_COLORS[shapes[i].bucket])
            sp.set_linewidth(2.0)
    handles = [
        Patch(facecolor=BUCKET_COLORS[b], edgecolor="k", label=b.replace("_", " "))
        for b in BUCKET_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build OOD target silhouettes (reviewer Priority 1)."
    )
    parser.add_argument("--config", default="configs/data_generator.yaml")
    parser.add_argument("--out-dir", default="outputs/ood")
    parser.add_argument("--img-h", type=int, default=None, help="Override config img_h.")
    parser.add_argument("--img-w", type=int, default=None, help="Override config img_w.")
    parser.add_argument("--margin", type=int, default=6, help="Canvas margin in pixels.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the free-form doodle.")
    parser.add_argument(
        "--only-buckets",
        nargs="*",
        choices=BUCKET_ORDER,
        default=None,
        help="Restrict generation to these buckets.",
    )
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()

    cfg_h, cfg_w = _load_resolution(args.config)
    h = args.img_h or cfg_h
    w = args.img_w or cfg_w

    shapes = build_catalogue(args.seed)
    if args.only_buckets:
        shapes = [s for s in shapes if s.bucket in set(args.only_buckets)]

    masks: list[np.ndarray] = []
    rows_csv: list[dict] = []
    for s in shapes:
        mask = rasterize_shape(s.prims, h, w, args.margin)
        fill = float(mask.mean())
        if fill <= 0.0:
            raise RuntimeError(f"Shape '{s.name}' rasterised empty -- check its definition.")
        sol = compute_solidity(mask)
        holes = mask_hole_metrics(mask)
        masks.append(mask)
        rows_csv.append(
            {
                "name": s.name,
                "bucket": s.bucket,
                "family": s.family,
                "solidity": round(sol, 4),
                "nonconvexity": round(1.0 - sol, 4),
                "fill_ratio": round(fill, 4),
                "hole_count": int(holes["hole_count"]),
                "params": json.dumps(s.params, sort_keys=True),
            }
        )

    os.makedirs(args.out_dir, exist_ok=True)
    mask_arr = np.stack(masks, axis=0).astype(np.float32)
    npz_path = os.path.join(args.out_dir, "ood_targets.npz")
    np.savez_compressed(
        npz_path,
        masks=mask_arr,
        names=np.array([r["name"] for r in rows_csv], dtype=object),
        buckets=np.array([r["bucket"] for r in rows_csv], dtype=object),
        families=np.array([r["family"] for r in rows_csv], dtype=object),
        solidity=np.array([r["solidity"] for r in rows_csv], dtype=np.float32),
        nonconvexity=np.array([r["nonconvexity"] for r in rows_csv], dtype=np.float32),
        fill_ratio=np.array([r["fill_ratio"] for r in rows_csv], dtype=np.float32),
        hole_count=np.array([r["hole_count"] for r in rows_csv], dtype=np.int32),
        params=np.array([r["params"] for r in rows_csv], dtype=object),
        resolution=np.array([h, w], dtype=np.int32),
    )

    csv_path = os.path.join(args.out_dir, "ood_targets_index.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_csv[0].keys()))
        writer.writeheader()
        writer.writerows(rows_csv)

    if not args.no_preview:
        save_preview(shapes, masks, os.path.join(args.out_dir, "ood_preview.png"))

    by_bucket: dict[str, int] = {}
    for r in rows_csv:
        by_bucket[r["bucket"]] = by_bucket.get(r["bucket"], 0) + 1
    print(f"Wrote {mask_arr.shape[0]} OOD targets at {h}x{w} -> {npz_path}")
    print("  per bucket: " + ", ".join(f"{b}={by_bucket.get(b, 0)}" for b in BUCKET_ORDER))
    print(f"  index: {csv_path}")
    if not args.no_preview:
        print(f"  preview: {os.path.join(args.out_dir, 'ood_preview.png')}")


if __name__ == "__main__":
    main()
