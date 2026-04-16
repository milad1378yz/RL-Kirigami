import math
import pickle
from functools import partial
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data_generator.generator import load_generator_config, resolve_generator_output_paths


class KirigamiDataset(Dataset):
    def __init__(self, images: torch.Tensor, masks: torch.Tensor, mask_transform=None):
        self.images = images
        self.masks = masks
        self.mask_transform = mask_transform

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        mask = self.masks[idx]
        cond_mask = self.mask_transform(mask) if self.mask_transform is not None else mask
        return {
            "images": self.images[idx],
            "masks": cond_mask,
            "metric_masks": mask,
        }


def resolve_data_settings(data_cfg: dict) -> dict:
    generator_config_path = data_cfg.get("generator_config", "configs/data_generator.yaml")
    generator_cfg = load_generator_config(generator_config_path)
    default_pickle_path, _, _ = resolve_generator_output_paths()
    pickle_path = data_cfg.get("pickle_path") or default_pickle_path

    return {
        "generator_config": generator_config_path,
        "pickle_path": pickle_path,
        "split_train": data_cfg.get("split_train", "train"),
        "split_val": data_cfg.get("split_val", "valid"),
        "grid_rows": int(generator_cfg["grid_rows"]),
        "grid_cols": int(generator_cfg["grid_cols"]),
        "x_min": float(generator_cfg["x_min"]),
        "x_max": float(generator_cfg["x_max"]),
        "mask_size": (int(generator_cfg["img_h"]), int(generator_cfg["img_w"])),
    }


def prepare_training_config(config: dict) -> dict:
    data_cfg = resolve_data_settings(config["data"])
    config["data"].update(data_cfg)
    model_cfg = config.setdefault("model_config", {})
    input_size = [int(data_cfg["grid_rows"]), int(data_cfg["grid_cols"])]
    model_cfg.setdefault("input_size", input_size)
    model_cfg.setdefault("output_size", list(model_cfg["input_size"]))
    model_cfg.setdefault("mask_size", [int(v) for v in data_cfg["mask_size"]])
    return config


def load_dataset_split(pickle_path: str, split: str) -> dict:
    with open(pickle_path, "rb") as handle:
        data = pickle.load(handle)

    entries = data.get(split, [])
    if not entries:
        raise ValueError(f"No data found for split '{split}' in '{pickle_path}'.")

    images = torch.stack(
        [torch.tensor(entry["image"], dtype=torch.float32) for entry in entries],
        dim=0,
    )
    masks = torch.stack(
        [torch.tensor(entry["mask"], dtype=torch.float32) for entry in entries],
        dim=0,
    )

    if images.dim() != 4:
        raise ValueError(f"Expected images with shape [N,1,H,W], got {tuple(images.shape)}.")
    if masks.dim() != 4:
        raise ValueError(f"Expected masks with shape [N,1,H,W], got {tuple(masks.shape)}.")

    meta = entries[0].get("metadata", {})
    spec = {
        "grid_rows": int(meta.get("grid_rows", images.shape[-2])),
        "grid_cols": int(meta.get("grid_cols", images.shape[-1])),
        "input_size": tuple(int(v) for v in images.shape[-2:]),
        "mask_size": tuple(int(v) for v in masks.shape[-2:]),
        "x_min": float(images.min().item()),
        "x_max": float(images.max().item()),
    }
    return {
        "images": images,
        "masks": masks,
        "spec": spec,
    }


def _mask_bbox(mask: torch.Tensor, threshold: float = 0.5) -> Optional[tuple[float, float, float, float]]:
    fg = mask[0] >= float(threshold)
    if not bool(fg.any()):
        return None
    ys, xs = fg.nonzero(as_tuple=True)
    return (
        float(xs.min().item()),
        float(xs.max().item()),
        float(ys.min().item()),
        float(ys.max().item()),
    )


def _sample_uniform(low: float, high: float) -> float:
    if high <= low:
        return float(low)
    return float(torch.empty((), dtype=torch.float32).uniform_(float(low), float(high)).item())


def _warp_mask_similarity(
    mask: torch.Tensor,
    *,
    angle_deg: float,
    scale: float,
    shift_x: float,
    shift_y: float,
    center_x: float,
    center_y: float,
) -> torch.Tensor:
    _, height, width = mask.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=mask.dtype),
        torch.arange(width, dtype=mask.dtype),
        indexing="ij",
    )

    theta = math.radians(float(angle_deg))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    x_out = xx - float(shift_x) - float(center_x)
    y_out = yy - float(shift_y) - float(center_y)

    safe_scale = max(float(scale), 1e-6)
    x_in = (cos_t * x_out + sin_t * y_out) / safe_scale + float(center_x)
    y_in = (-sin_t * x_out + cos_t * y_out) / safe_scale + float(center_y)

    if width > 1:
        grid_x = (2.0 * x_in / float(width - 1)) - 1.0
    else:
        grid_x = torch.zeros_like(x_in)
    if height > 1:
        grid_y = (2.0 * y_in / float(height - 1)) - 1.0
    else:
        grid_y = torch.zeros_like(y_in)

    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
    warped = F.grid_sample(
        mask.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return warped.squeeze(0).clamp_(0.0, 1.0)


def _maybe_transform_mask(
    mask: torch.Tensor,
    *,
    p: float,
    rotate_deg: float,
    scale_min: float,
    scale_max: float,
    shift_frac_x: float,
    shift_frac_y: float,
    threshold: float = 0.5,
) -> torch.Tensor:
    if p <= 0.0 or torch.rand(()) >= float(p):
        return mask

    bbox = _mask_bbox(mask, threshold=threshold)
    if bbox is None:
        return mask

    _, height, width = mask.shape
    x0, x1, y0, y1 = bbox
    center_x = 0.5 * (x0 + x1)
    center_y = 0.5 * (y0 + y1)

    angle = _sample_uniform(-abs(float(rotate_deg)), abs(float(rotate_deg))) if rotate_deg else 0.0
    scale_lo = max(1e-3, min(float(scale_min), float(scale_max)))
    scale_hi = max(scale_lo, max(float(scale_min), float(scale_max)))
    scale = _sample_uniform(scale_lo, scale_hi)

    theta = math.radians(angle)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    corners = torch.tensor(
        [
            [x0 - center_x, y0 - center_y],
            [x0 - center_x, y1 - center_y],
            [x1 - center_x, y0 - center_y],
            [x1 - center_x, y1 - center_y],
        ],
        dtype=torch.float32,
    )
    rot_x = cos_t * corners[:, 0] - sin_t * corners[:, 1]
    rot_y = sin_t * corners[:, 0] + cos_t * corners[:, 1]

    max_scale_fit = float("inf")
    x_max_rel = float(rot_x.max().item())
    x_min_rel = float(rot_x.min().item())
    y_max_rel = float(rot_y.max().item())
    y_min_rel = float(rot_y.min().item())
    if x_max_rel > 0.0:
        max_scale_fit = min(max_scale_fit, (float(width - 1) - center_x) / x_max_rel)
    if x_min_rel < 0.0:
        max_scale_fit = min(max_scale_fit, center_x / (-x_min_rel))
    if y_max_rel > 0.0:
        max_scale_fit = min(max_scale_fit, (float(height - 1) - center_y) / y_max_rel)
    if y_min_rel < 0.0:
        max_scale_fit = min(max_scale_fit, center_y / (-y_min_rel))
    if math.isfinite(max_scale_fit):
        scale = min(scale, max(1e-3, max_scale_fit))

    x_min = center_x + scale * x_min_rel
    x_max = center_x + scale * x_max_rel
    y_min = center_y + scale * y_min_rel
    y_max = center_y + scale * y_max_rel

    shift_limit_x = max(0.0, float(shift_frac_x)) * float(width - 1)
    shift_limit_y = max(0.0, float(shift_frac_y)) * float(height - 1)

    tx_min = max(-shift_limit_x, -x_min)
    tx_max = min(shift_limit_x, float(width - 1) - x_max)
    ty_min = max(-shift_limit_y, -y_min)
    ty_max = min(shift_limit_y, float(height - 1) - y_max)

    shift_x = _sample_uniform(tx_min, tx_max) if tx_max >= tx_min else 0.0
    shift_y = _sample_uniform(ty_min, ty_max) if ty_max >= ty_min else 0.0

    return _warp_mask_similarity(
        mask,
        angle_deg=angle,
        scale=scale,
        shift_x=shift_x,
        shift_y=shift_y,
        center_x=center_x,
        center_y=center_y,
    )


class KirigamiDataModule(pl.LightningDataModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.train_data = None
        self.val_data = None
        self.train_loader = None
        self.val_loader = None
        self.data_spec = None

    def setup(self, stage: Optional[str] = None) -> None:
        data_cfg = resolve_data_settings(self.config["data"])
        tr = self.config["training"]

        self.train_data = load_dataset_split(data_cfg["pickle_path"], data_cfg["split_train"])
        self.val_data = load_dataset_split(data_cfg["pickle_path"], data_cfg["split_val"])
        self.data_spec = dict(self.train_data["spec"])

        model_cfg = self.config["model_config"]
        input_size = tuple(int(v) for v in model_cfg["input_size"])
        mask_size = tuple(int(v) for v in model_cfg["mask_size"])
        if input_size != self.data_spec["input_size"]:
            raise ValueError(
                f"model_config.input_size={input_size} does not match dataset {self.data_spec['input_size']}."
            )
        if mask_size != self.data_spec["mask_size"]:
            raise ValueError(
                f"model_config.mask_size={mask_size} does not match dataset {self.data_spec['mask_size']}."
            )

        train_mask_transform = partial(
            _maybe_transform_mask,
            p=float(tr.get("augment_mask_p", 0.0)),
            rotate_deg=float(tr.get("augment_mask_rotate_deg", 0.0)),
            scale_min=float(tr.get("augment_mask_scale_min", 1.0)),
            scale_max=float(tr.get("augment_mask_scale_max", 1.0)),
            shift_frac_x=float(tr.get("augment_mask_shift_frac_x", 0.0)),
            shift_frac_y=float(tr.get("augment_mask_shift_frac_y", 0.0)),
        )

        self.train_loader = DataLoader(
            KirigamiDataset(
                self.train_data["images"],
                self.train_data["masks"],
                mask_transform=train_mask_transform,
            ),
            batch_size=int(tr["batch_size"]),
            shuffle=True,
            num_workers=int(tr.get("num_workers", 0)),
            pin_memory=bool(tr.get("pin_memory", False)),
        )
        self.val_loader = DataLoader(
            KirigamiDataset(
                self.val_data["images"],
                self.val_data["masks"],
                mask_transform=None,
            ),
            batch_size=int(tr.get("val_batch_size", tr["batch_size"])),
            shuffle=False,
            num_workers=int(tr.get("num_workers", 0)),
            pin_memory=bool(tr.get("pin_memory", False)),
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
