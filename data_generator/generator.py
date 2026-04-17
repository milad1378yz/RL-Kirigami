import argparse
import math
import os
import pickle
from contextlib import nullcontext
import numpy as np
import yaml
from scipy import ndimage
from tqdm import tqdm

from data_generator.utils import build_dataset_entry, build_geometry_context
from data_generator.visualization import save_gifs, save_preview


HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLER_NAMES = ("uniform", "structured")
STRUCTURED_FAMILY_NAMES = ("global", "piecewise", "symmetric", "bumps")


def _build_sampling_basis(rows, cols):
    grid_y, grid_x = np.mgrid[0:rows, 0:cols]
    u = (grid_x - (cols - 1) / 2.0) / max((cols - 1) / 2.0, 1.0)
    v = (grid_y - (rows - 1) / 2.0) / max((rows - 1) / 2.0, 1.0)
    r = np.sqrt(u * u + v * v)
    theta = np.arctan2(v, u)
    checker = (((grid_x + grid_y) % 2) * 2 - 1).astype(np.float64)

    corner_sigma = 0.65
    corners = []
    for cx, cy in ((-0.85, -0.85), (0.85, -0.85), (0.85, 0.85), (-0.85, 0.85)):
        corners.append(
            np.exp(-((u - cx) ** 2 + (v - cy) ** 2) / (2.0 * corner_sigma * corner_sigma))
        )

    return {
        "rows": rows,
        "cols": cols,
        "u": u,
        "v": v,
        "r": r,
        "theta": theta,
        "checker": checker,
        "corners": tuple(corners),
    }


def _bilinear_upsample(coarse, out_shape):
    coarse_rows, coarse_cols = coarse.shape
    rows, cols = out_shape
    yy = np.linspace(0, coarse_rows - 1, rows)
    xx = np.linspace(0, coarse_cols - 1, cols)
    x0 = np.floor(xx).astype(int)
    y0 = np.floor(yy).astype(int)
    x1 = np.clip(x0 + 1, 0, coarse_cols - 1)
    y1 = np.clip(y0 + 1, 0, coarse_rows - 1)
    tx = xx - x0
    ty = yy - y0
    out = np.zeros((rows, cols), dtype=np.float64)
    for i in range(rows):
        a = coarse[y0[i], x0]
        b = coarse[y0[i], x1]
        c = coarse[y1[i], x0]
        d = coarse[y1[i], x1]
        out[i] = (1.0 - ty[i]) * ((1.0 - tx) * a + tx * b) + ty[i] * ((1.0 - tx) * c + tx * d)
    return out


def _sample_global_field(rng, basis):
    coeff = rng.uniform(-1.3, 1.3, size=10)
    return (
        coeff[0]
        + coeff[1] * basis["u"]
        + coeff[2] * basis["v"]
        + coeff[3] * (basis["u"] * basis["v"])
        + coeff[4] * (basis["u"] * basis["u"] - basis["v"] * basis["v"])
        + coeff[5] * basis["r"]
        + coeff[6] * np.cos(2.0 * basis["theta"])
        + coeff[7] * np.sin(2.0 * basis["theta"])
        + coeff[8] * basis["checker"]
        + coeff[9]
        * (
            basis["corners"][0]
            + basis["corners"][2]
            - basis["corners"][1]
            - basis["corners"][3]
        )
    )


def _sample_piecewise_field(rng, basis):
    coarse = rng.uniform(-1.0, 1.0, size=(4, 4))
    field = _bilinear_upsample(coarse, (basis["rows"], basis["cols"]))
    return ndimage.gaussian_filter(field, sigma=0.55, mode="nearest")


def _sample_symmetric_field(rng, basis):
    coeff = rng.uniform(-1.3, 1.3, size=10)
    field = (
        coeff[0]
        + coeff[1] * np.abs(basis["u"])
        + coeff[2] * np.abs(basis["v"])
        + coeff[3] * (basis["u"] * basis["v"])
        + coeff[4] * basis["r"]
        + coeff[5] * np.cos(2.0 * basis["theta"])
        + coeff[6] * np.cos(4.0 * basis["theta"])
        + coeff[7] * np.cos(6.0 * basis["theta"])
        + coeff[8] * basis["checker"]
        + coeff[9]
        * (
            basis["corners"][0]
            + basis["corners"][1]
            + basis["corners"][2]
            + basis["corners"][3]
        )
    )
    symmetry_mode = int(rng.integers(0, 4))
    if symmetry_mode == 0:
        field = 0.5 * (field + field[:, ::-1])
    elif symmetry_mode == 1:
        field = 0.5 * (field + field[::-1, :])
    elif symmetry_mode == 2:
        field = 0.25 * (field + field[:, ::-1] + field[::-1, :] + field[::-1, ::-1])
    return field


def _sample_bump_field(rng, basis):
    field = np.zeros((basis["rows"], basis["cols"]), dtype=np.float64)
    for _ in range(int(rng.integers(2, 5))):
        cx = rng.uniform(-0.95, 0.95)
        cy = rng.uniform(-0.95, 0.95)
        sigma = rng.uniform(0.15, 0.80)
        amp = rng.uniform(-1.35, 1.35)
        field += amp * np.exp(
            -((basis["u"] - cx) ** 2 + (basis["v"] - cy) ** 2) / (2.0 * sigma * sigma)
        )
    field += rng.uniform(-0.55, 0.55) * basis["u"]
    field += rng.uniform(-0.55, 0.55) * basis["v"]
    field += rng.uniform(-0.9, 0.9) * np.cos(4.0 * basis["theta"])
    field += rng.uniform(-0.7, 0.7) * basis["checker"]
    return field


def _map_field_to_x_range(field, rng, x_min, x_max):
    if x_max <= x_min:
        return np.full_like(field, fill_value=x_min, dtype=np.float64)

    log_min = math.log(float(x_min))
    log_max = math.log(float(x_max))
    span = log_max - log_min
    centered = np.asarray(field, dtype=np.float64) - float(np.mean(field))
    scale = float(np.max(np.abs(centered))) if centered.size else 0.0
    if not np.isfinite(scale) or scale < 1e-8:
        normalized = np.zeros_like(centered)
    else:
        normalized = centered / scale

    amplitude = rng.uniform(0.18, 0.5) * span
    midpoint_min = log_min + amplitude
    midpoint_max = log_max - amplitude
    if midpoint_max > midpoint_min:
        midpoint = rng.uniform(midpoint_min, midpoint_max)
    else:
        midpoint = 0.5 * (log_min + log_max)
    return np.exp(midpoint + amplitude * normalized).astype(np.float64)


def _sample_structured_x_matrix(rng, basis, x_min, x_max, family_name):
    if family_name == "global":
        field = _sample_global_field(rng, basis)
    elif family_name == "piecewise":
        field = _sample_piecewise_field(rng, basis)
    elif family_name == "symmetric":
        field = _sample_symmetric_field(rng, basis)
    elif family_name == "bumps":
        field = _sample_bump_field(rng, basis)
    else:
        raise ValueError(f"unknown structured sampler family '{family_name}'")
    return _map_field_to_x_range(field, rng, x_min, x_max)


def _sample_x_matrix(rng, rows, cols, x_min, x_max, sampler, basis, family_idx):
    if sampler == "uniform":
        return rng.uniform(x_min, x_max, size=(rows, cols))
    family_name = STRUCTURED_FAMILY_NAMES[family_idx % len(STRUCTURED_FAMILY_NAMES)]
    return _sample_structured_x_matrix(rng, basis, x_min, x_max, family_name)


def generate_valid_samples(
    rows,
    cols,
    height,
    width,
    target_count,
    rng,
    x_min,
    x_max,
    context,
    sampler,
    progress_desc=None,
):
    samples = []
    attempts = 0
    max_attempts = max(100, 12 * target_count)
    basis = _build_sampling_basis(rows, cols) if sampler == "structured" else None
    progress_context = (
        tqdm(total=target_count, desc=progress_desc, unit="sample", disable=target_count <= 0)
        if progress_desc
        else nullcontext()
    )
    with progress_context as progress:
        while len(samples) < target_count and attempts < max_attempts:
            x_matrix = _sample_x_matrix(rng, rows, cols, x_min, x_max, sampler, basis, attempts)
            entry = build_dataset_entry(rows, cols, x_matrix, context, height, width)
            if entry is not None:
                samples.append(entry)
                if progress is not None:
                    progress.update(1)
            attempts += 1
    return samples, attempts


def load_generator_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return config


def resolve_generator_output_paths(out="", preview="", gif_dir=""):
    out_path = out or os.path.join(HERE, "kirigami_x_dataset.pkl")
    preview_path = preview or os.path.join(HERE, "preview.png")
    gif_path = gif_dir or os.path.join(HERE, "gifs")
    return out_path, preview_path, gif_path


def parse_args():
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default="configs/data_generator.yaml")
    known, _ = bootstrap.parse_known_args()
    config = load_generator_config(known.config)

    parser = argparse.ArgumentParser(description="Kirigami generator using x_ij = a / b.")
    parser.add_argument("--config", default=known.config)
    parser.add_argument("--grid-rows", type=int, default=config["grid_rows"])
    parser.add_argument("--grid-cols", type=int, default=config["grid_cols"])
    parser.add_argument("--img-h", type=int, default=config["img_h"])
    parser.add_argument("--img-w", type=int, default=config["img_w"])
    parser.add_argument("--train", type=int, default=config["train"])
    parser.add_argument("--valid", type=int, default=config["valid"])
    parser.add_argument("--test", type=int, default=config.get("test", 0))
    parser.add_argument("--x-min", type=float, default=config["x_min"])
    parser.add_argument("--x-max", type=float, default=config["x_max"])
    parser.add_argument("--sampler", choices=SAMPLER_NAMES, default=config.get("sampler", "structured"))
    parser.add_argument("--seed", type=int, default=config["seed"])
    parser.add_argument("--preview-count", type=int, default=config["preview_count"])
    parser.add_argument("--gif-count", type=int, default=config["gif_count"])
    parser.add_argument("--gif-frames", type=int, default=config["gif_frames"])
    parser.add_argument("--gif-duration", type=float, default=config["gif_duration"])
    parser.add_argument("--out", default="")
    parser.add_argument("--preview", default="")
    parser.add_argument("--gif-dir", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.x_min <= 0.0:
        raise ValueError("x_min must stay positive because x = a / b.")
    if args.x_max < args.x_min:
        raise ValueError("x_max must be greater than or equal to x_min.")

    out_path, preview_path, gif_dir = resolve_generator_output_paths(
        out=args.out,
        preview=args.preview,
        gif_dir=args.gif_dir,
    )

    rows = args.grid_rows
    cols = args.grid_cols
    geometry_context = build_geometry_context(rows, cols)

    train_samples, train_attempts = generate_valid_samples(
        rows,
        cols,
        args.img_h,
        args.img_w,
        args.train,
        np.random.default_rng(args.seed),
        args.x_min,
        args.x_max,
        geometry_context,
        args.sampler,
        progress_desc="train",
    )
    valid_samples, valid_attempts = generate_valid_samples(
        rows,
        cols,
        args.img_h,
        args.img_w,
        args.valid,
        np.random.default_rng(args.seed + 1),
        args.x_min,
        args.x_max,
        geometry_context,
        args.sampler,
        progress_desc="valid",
    )
    test_samples, test_attempts = generate_valid_samples(
        rows,
        cols,
        args.img_h,
        args.img_w,
        args.test,
        np.random.default_rng(args.seed + 2),
        args.x_min,
        args.x_max,
        geometry_context,
        args.sampler,
        progress_desc="test",
    )

    with open(out_path, "wb") as handle:
        pickle.dump(
            {"train": train_samples, "valid": valid_samples, "test": test_samples},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    samples_for_visuals = train_samples or valid_samples or test_samples
    save_preview(preview_path, samples_for_visuals, geometry_context, args.preview_count)
    gif_paths = save_gifs(
        gif_dir,
        samples_for_visuals,
        geometry_context,
        args.gif_count,
        args.gif_frames,
        args.gif_duration,
    )

    print(f"saved dataset: {out_path}")
    print(f"saved preview: {preview_path}")
    if gif_paths:
        print(f"saved gifs: {gif_dir}")
    print(f"sampler: {args.sampler}")
    print(f"accepted train={len(train_samples)}/{args.train} after {train_attempts} attempts")
    print(f"accepted valid={len(valid_samples)}/{args.valid} after {valid_attempts} attempts")
    print(f"accepted test={len(test_samples)}/{args.test} after {test_attempts} attempts")


if __name__ == "__main__":
    main()
