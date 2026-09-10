"""Shared frozen-backbone and MSRS data utilities for SAHCA.

This module intentionally contains no optimizer or training loop. Evaluation
and visualization therefore remain usable when the training entry point is
not present on a machine.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from fusion_model import FusionSerialBridge


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
RESAMPLING = getattr(Image, "Resampling", Image)


@dataclass
class BackboneConfig:
    fusion_weights: str = "checkpoints/fusion.pkl"
    lyt_weights: Optional[str] = "checkpoints/lyt.pth"
    lyt_root: Optional[str] = "third_party/LYT"
    device: str = "cuda"


def torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_fusion_state(model: torch.nn.Module, path: str) -> None:
    checkpoint = torch_load(path, "cpu")
    if isinstance(checkpoint, dict):
        state = checkpoint.get(
            "weight", checkpoint.get("state_dict", checkpoint)
        )
    else:
        state = checkpoint
    state = {
        key.removeprefix("module."): value
        for key, value in state.items()
    }
    current = model.state_dict()
    compatible = {
        key: value for key, value in state.items()
        if key in current and current[key].shape == value.shape
    }
    if not compatible:
        raise RuntimeError(
            f"No compatible fusion tensors were found in {path}"
        )
    model.load_state_dict(compatible, strict=False)
    print(
        f"Loaded frozen fusion backbone: "
        f"{len(compatible)}/{len(state)} tensors from {path}"
    )


def build_frozen_backbone(config, device: torch.device):
    model = FusionSerialBridge(
        use_lyt=True,
        lyt_weights=getattr(config, "lyt_weights", None),
        lyt_root=getattr(config, "lyt_root", None),
        lyt_device=device,
        use_credibility=False,
        use_polar_chroma=False,
    ).to(device)
    load_fusion_state(model, str(config.fusion_weights))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.eval()


@torch.no_grad()
def final_luminance(backbone, visible, infrared):
    _, diagnostics = backbone(
        visible, infrared, return_diagnostics=True
    )
    return diagnostics["y_in"], diagnostics["y_fused"]


def image_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = array[..., None]
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class PairedMSRS(Dataset):
    def __init__(
        self,
        data_root,
        split,
        width,
        height,
        limit=None,
        augment=False,
    ):
        root = Path(data_root) / split
        visible_dir = root / "vi"
        infrared_dir = root / "ir"
        if not visible_dir.is_dir() or not infrared_dir.is_dir():
            raise FileNotFoundError(
                f"Expected paired folders {visible_dir} and {infrared_dir}"
            )

        infrared_by_stem = {
            path.stem: path
            for path in infrared_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
        self.pairs = [
            (visible_path, infrared_by_stem[visible_path.stem])
            for visible_path in sorted(visible_dir.iterdir())
            if (
                visible_path.is_file()
                and visible_path.suffix.lower() in IMAGE_SUFFIXES
                and visible_path.stem in infrared_by_stem
            )
        ]
        if limit is not None:
            self.pairs = self.pairs[: int(limit)]
        if not self.pairs:
            raise RuntimeError(f"No paired images found under {root}")

        self.width = int(width)
        self.height = int(height)
        self.augment = bool(augment)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        visible_path, infrared_path = self.pairs[index]
        size = (self.width, self.height)
        visible = Image.open(visible_path).convert("RGB").resize(
            size, RESAMPLING.BILINEAR
        )
        infrared = Image.open(infrared_path).convert("L").resize(
            size, RESAMPLING.BILINEAR
        )
        visible_tensor = image_tensor(visible)
        infrared_tensor = image_tensor(infrared)

        if self.augment and random.random() < 0.5:
            visible_tensor = visible_tensor.flip(-1)
            infrared_tensor = infrared_tensor.flip(-1)

        return visible_tensor, infrared_tensor, visible_path.name
