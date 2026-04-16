import pickle
from functools import partial
from typing import Optional

from kirigami_training import ensure_lightning_compat

ensure_lightning_compat()

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from data_generator.generator import load_config as load_generator_config
from data_generator.generator import resolve_output_paths


class KirigamiDataset(Dataset):
    def __init__(self, images: torch.Tensor, masks: torch.Tensor, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = {
            "images": self.images[idx],
            "masks": self.masks[idx],
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def resolve_data_settings(data_cfg: dict) -> dict:
    generator_config_path = data_cfg.get("generator_config", "configs/data_generator.yaml")
    generator_cfg = load_generator_config(generator_config_path)
    default_pickle_path, _, _ = resolve_output_paths()
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


def _maybe_flip(sample: dict[str, torch.Tensor], hflip_p: float, vflip_p: float) -> dict[str, torch.Tensor]:
    images = sample["images"]
    masks = sample["masks"]

    if hflip_p > 0.0 and torch.rand(()) < float(hflip_p):
        images = torch.flip(images, dims=[-1])
        masks = torch.flip(masks, dims=[-1])
    if vflip_p > 0.0 and torch.rand(()) < float(vflip_p):
        images = torch.flip(images, dims=[-2])
        masks = torch.flip(masks, dims=[-2])

    return {"images": images, "masks": masks}


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

        train_transform = partial(
            _maybe_flip,
            hflip_p=float(tr.get("augment_hflip_p", 0.0)),
            vflip_p=float(tr.get("augment_vflip_p", 0.0)),
        )

        self.train_loader = DataLoader(
            KirigamiDataset(
                self.train_data["images"],
                self.train_data["masks"],
                transform=train_transform,
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
                transform=None,
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
