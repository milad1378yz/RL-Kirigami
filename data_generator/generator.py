import argparse
import os
import pickle

os.environ.setdefault("MPLCONFIGDIR", "/tmp/kirigami_x_mplconfig")

import numpy as np
import yaml

from data_generator.utils import build_context, make_sample
from data_generator.visualization import save_gifs, save_preview


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "configs", "default.yaml")


def generate_dataset(rows, cols, height, width, n_target, rng, x_min, x_max, context):
    samples = []
    attempts = 0
    max_attempts = max(100, 12 * n_target)
    while len(samples) < n_target and attempts < max_attempts:
        x_matrix = rng.uniform(x_min, x_max, size=(rows, cols))
        sample = make_sample(rows, cols, x_matrix, context, height, width)
        if sample is not None:
            samples.append(sample)
        attempts += 1
    return samples, attempts


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return config


def parse_args():
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default=DEFAULT_CONFIG)
    known, _ = bootstrap.parse_known_args()
    config = load_config(known.config)

    parser = argparse.ArgumentParser(description="Standalone kirigami generator using x_ij = a / b.")
    parser.add_argument("--config", default=known.config)
    parser.add_argument("--grid-rows", type=int, default=config["grid_rows"])
    parser.add_argument("--grid-cols", type=int, default=config["grid_cols"])
    parser.add_argument("--img-h", type=int, default=config["img_h"])
    parser.add_argument("--img-w", type=int, default=config["img_w"])
    parser.add_argument("--train", type=int, default=config["train"])
    parser.add_argument("--valid", type=int, default=config["valid"])
    parser.add_argument("--x-min", type=float, default=config["x_min"])
    parser.add_argument("--x-max", type=float, default=config["x_max"])
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

    out_path = args.out or os.path.join(HERE, "kirigami_x_dataset.pkl")
    preview_path = args.preview or os.path.join(HERE, "preview.png")
    gif_dir = args.gif_dir or os.path.join(HERE, "gifs")

    rows = args.grid_rows
    cols = args.grid_cols
    context = build_context(rows, cols)

    train, train_attempts = generate_dataset(
        rows,
        cols,
        args.img_h,
        args.img_w,
        args.train,
        np.random.default_rng(args.seed),
        args.x_min,
        args.x_max,
        context,
    )
    valid, valid_attempts = generate_dataset(
        rows,
        cols,
        args.img_h,
        args.img_w,
        args.valid,
        np.random.default_rng(args.seed + 1),
        args.x_min,
        args.x_max,
        context,
    )

    with open(out_path, "wb") as handle:
        pickle.dump({"train": train, "valid": valid}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    samples_for_visuals = train or valid
    save_preview(preview_path, samples_for_visuals, context, args.preview_count)
    gif_paths = save_gifs(
        gif_dir,
        samples_for_visuals,
        context,
        args.gif_count,
        args.gif_frames,
        args.gif_duration,
    )

    print(f"saved dataset: {out_path}")
    print(f"saved preview: {preview_path}")
    if gif_paths:
        print(f"saved gifs: {gif_dir}")
    print(f"accepted train={len(train)}/{args.train} after {train_attempts} attempts")
    print(f"accepted valid={len(valid)}/{args.valid} after {valid_attempts} attempts")


if __name__ == "__main__":
    main()
