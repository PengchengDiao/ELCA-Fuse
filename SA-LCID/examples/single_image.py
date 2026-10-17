"""Minimal SA-LCID example for one registered image triplet."""

from pathlib import Path

import numpy as np
from PIL import Image

from sa_lcid import diagnose, reconstruct_with_fixed_luminance, relative_luminance


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float64) / 255.0


visible = load_rgb(Path("visible.png"))
infrared = load_gray(Path("infrared.png"))
fused = load_rgb(Path("fused.png"))

fixed_y = relative_luminance(fused)
controlled = reconstruct_with_fixed_luminance(fused, fixed_y)
result = diagnose(visible, infrared, controlled, fixed_luminance=fixed_y)

for key, value in result.items():
    if np.isscalar(value):
        print(f"{key}: {float(value):.8f}")

