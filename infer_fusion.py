"""Run ELCA-Fuse inference on paired visible and infrared images."""

from __future__ import annotations

import re
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from chroma_polar_interpretable import SAHCA
from chroma_polar_interpretable.style_adapter import (
    source_safe_ich_chroma_trim,
)
from fusion_model import (
    FusionSerialBridge,
    rgb_to_ycbcr,
    ycbcr_to_rgb,
)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Inference configuration
import argparse

_PARSER = argparse.ArgumentParser(description="ELCA-Fuse inference")
_PARSER.add_argument("--visible", type=str, default="data/MSRS/test/vi")
_PARSER.add_argument("--infrared", type=str, default="data/MSRS/test/ir")
_PARSER.add_argument("--output", type=str, default="outputs/MSRS")
_PARSER.add_argument("--device", type=str, default="cuda")
_PARSER.add_argument("--max_size", type=int, default=1280)
_PARSER.add_argument("--lyt_device", type=str, default="cpu")
_PARSER.add_argument("--fusion_weights", type=str, default="checkpoints/fusion.pkl")
_PARSER.add_argument("--micg1_weights", type=str, default="checkpoints/micg1.pkl")
_PARSER.add_argument("--micg2_weights", type=str, default="checkpoints/micg2.pkl")
_PARSER.add_argument("--sahca_weights", type=str, default="checkpoints/sahca.pkl")
_PARSER.add_argument("--lyt_weights", type=str, default="checkpoints/lyt.pth")
_PARSER.add_argument("--lyt_root", type=str, default="third_party/LYT")
_PARSER.add_argument("--save_diagnostics", action="store_true")
_ARGS, _ = _PARSER.parse_known_args()

VISIBLE = Path(_ARGS.visible)
INFRARED = Path(_ARGS.infrared)
OUTPUT = Path(_ARGS.output)
FUSION_WEIGHTS = Path(_ARGS.fusion_weights)
CALIBRATOR_WEIGHTS = Path(_ARGS.micg1_weights)
MAIN_CALIBRATOR_WEIGHTS = Path(_ARGS.micg2_weights)
SAHCA_WEIGHTS = Path(_ARGS.sahca_weights)
LYT_WEIGHTS = _ARGS.lyt_weights
LYT_ROOT = _ARGS.lyt_root
DEVICE = _ARGS.device

USE_LYT = True
USE_CREDIBILITY = True
USE_SAHCA_CHROMA = True
USE_CHROMA_TRIM = True
CHROMA_TRIM_STRENGTH = 1.00
CHROMA_TRIM_MAXIMUM_GAIN = float("inf")
SAVE_MICG_DIAGNOSTICS = _ARGS.save_diagnostics


def load_state(path: Path):
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        state = state.get("weight", state.get("state_dict", state))
    return {key.removeprefix("module."): value for key, value in state.items()}


def load_partial(model: torch.nn.Module, path: Path) -> None:
    state = load_state(path)
    current = model.state_dict()
    compatible = {
        key: value for key, value in state.items()
        if key in current and current[key].shape == value.shape
    }
    model.load_state_dict(compatible, strict=False)
    print(f"Loaded {len(compatible)}/{len(state)} tensors from {path}")


def load_sahca(path: Path, device: torch.device) -> SAHCA:
    try:
        checkpoint = torch.load(
            path, map_location=device, weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model", checkpoint)
    model = SAHCA().to(device)
    model.load_state_dict(state, strict=True)
    return model.eval()


def image_tensor(path: Path, mode: str, device: torch.device, max_size: int = 640) -> torch.Tensor:
    img = Image.open(path).convert(mode)
    if max(img.size) > max_size:
        w, h = img.size
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    array = np.asarray(img, dtype=np.float32) / 255.0
    if mode == "RGB":
        tensor = torch.from_numpy(array).permute(2, 0, 1)
    else:
        tensor = torch.from_numpy(array).unsqueeze(0)
    return tensor.unsqueeze(0).to(device)


def save_rgb(tensor: torch.Tensor, path: Path) -> None:
    array = tensor[0].detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((array * 255.0).round().astype(np.uint8)).save(path)


def save_gray(tensor: torch.Tensor, path: Path) -> None:
    array = tensor[0, 0].detach().clamp(0, 1).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((array * 255.0).round().astype(np.uint8)).save(path)


def pair_key(path: Path) -> str:
    return re.sub(
        r"(?i)(?:[_-](?:vi|vis|visible|rgb|ir|thermal|infrared|lwir))+$",
        "",
        path.stem,
    )


def collect_pairs(visible: Path, infrared: Path):
    if visible.is_file() and infrared.is_file():
        return [(visible, infrared)]
    if not visible.is_dir() or not infrared.is_dir():
        raise ValueError("visible and infrared must both be files or both be directories")

    vi_files = {
        pair_key(path): path for path in visible.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    ir_files = {
        pair_key(path): path for path in infrared.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    return [(vi_files[key], ir_files[key]) for key in sorted(vi_files.keys() & ir_files.keys())]


def main() -> None:
    device = torch.device(DEVICE if DEVICE.startswith("cpu") or torch.cuda.is_available() else "cpu")
    if USE_CREDIBILITY and not CALIBRATOR_WEIGHTS.is_file():
        raise FileNotFoundError(f"Calibrator checkpoint not found: {CALIBRATOR_WEIGHTS}")
    if USE_CREDIBILITY and not MAIN_CALIBRATOR_WEIGHTS.is_file():
        raise FileNotFoundError(
            f"MICG-2 checkpoint not found: {MAIN_CALIBRATOR_WEIGHTS}"
        )

    lyt_dev = torch.device(_ARGS.lyt_device if (_ARGS.lyt_device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = FusionSerialBridge(
        use_lyt=USE_LYT,
        lyt_weights=LYT_WEIGHTS,
        lyt_root=LYT_ROOT,
        lyt_device=lyt_dev,
        use_credibility=USE_CREDIBILITY,
        credibility_weights=str(CALIBRATOR_WEIGHTS) if USE_CREDIBILITY else None,
        main_credibility_weights=(
            str(MAIN_CALIBRATOR_WEIGHTS) if USE_CREDIBILITY else None
        ),
        use_polar_chroma=False,
    )

    if FUSION_WEIGHTS.is_file():
        load_partial(model, FUSION_WEIGHTS)
    else:
        raise FileNotFoundError(f"Fusion checkpoint not found: {FUSION_WEIGHTS}")
    model.to(device).eval()
    sahca = None
    if USE_SAHCA_CHROMA:
        if not SAHCA_WEIGHTS.is_file():
            raise FileNotFoundError(
                f"SAHCA checkpoint not found: {SAHCA_WEIGHTS}"
            )
        sahca = load_sahca(SAHCA_WEIGHTS, device)

    pairs = collect_pairs(VISIBLE, INFRARED)
    if not pairs:
        raise RuntimeError("No matching visible/infrared image pairs found")

    output_is_file = len(pairs) == 1 and bool(OUTPUT.suffix)
    with torch.no_grad():
        for index, (vi_path, ir_path) in enumerate(pairs, 1):
            vis = image_tensor(vi_path, "RGB", device, _ARGS.max_size)
            infra = image_tensor(ir_path, "L", device, _ARGS.max_size)
            if vis.shape[-2:] != infra.shape[-2:]:
                raise ValueError(f"Image size mismatch: {vi_path.name} and {ir_path.name}")
            _, diagnostics = model(
                vis, infra, return_diagnostics=True
            )
            y_fused = diagnostics["y_fused"]
            sahca_aux = None
            trim_aux = None
            if sahca is not None:
                _, cb_visible, cr_visible = rgb_to_ycbcr(vis)
                cb_sahca, cr_sahca, sahca_aux = sahca(
                    diagnostics["y_in"],
                    y_fused,
                    cb_visible,
                    cr_visible,
                    infra,
                )
                if USE_CHROMA_TRIM:
                    cb_out, cr_out, trim_aux = (
                        source_safe_ich_chroma_trim(
                            cb_sahca=cb_sahca,
                            cr_sahca=cr_sahca,
                            visible_rgb=vis,
                            y_fused=y_fused,
                            source_confidence=(
                                sahca_aux["source_confidence"]
                            ),
                            ir_only=sahca_aux["ir_only"],
                            visible_noise=sahca_aux["visible_noise"],
                            clipping_risk=sahca_aux["clipping_risk"],
                            strength=CHROMA_TRIM_STRENGTH,
                            maximum_gain=CHROMA_TRIM_MAXIMUM_GAIN,
                        )
                    )
                else:
                    cb_out, cr_out = cb_sahca, cr_sahca
                fused = ycbcr_to_rgb(y_fused, cb_out, cr_out)
            else:
                _, cb_visible, cr_visible = rgb_to_ycbcr(vis)
                fused = ycbcr_to_rgb(
                    y_fused, cb_visible, cr_visible
                )
            destination = OUTPUT if output_is_file else OUTPUT / f"{vi_path.stem}.png"
            save_rgb(fused, destination)
            if SAVE_MICG_DIAGNOSTICS:
                diagnostic_dir = destination.parent / "micg" / vi_path.stem
                save_gray(diagnostics["weight_lyt"], diagnostic_dir / "weight_lyt.png")
                save_gray(diagnostics["weight_tcdg"], diagnostic_dir / "weight_tcdg.png")
                save_gray(
                    diagnostics["enhancement"]["confidence_total"],
                    diagnostic_dir / "confidence_enhancement.png",
                )
                if "confidence_total" in diagnostics["fusion"]:
                    save_gray(
                        diagnostics["fusion"]["confidence_total"],
                        diagnostic_dir / "confidence_fusion.png",
                    )
                save_gray(
                    diagnostics["enhancement"]["features"]["artifact_ratio"],
                    diagnostic_dir / "artifact_ratio.png",
                )
                if sahca_aux is not None:
                    save_gray(
                        sahca_aux["source_confidence"],
                        diagnostic_dir / "source_confidence.png",
                    )
                    save_gray(
                        sahca_aux["ir_only"],
                        diagnostic_dir / "ir_only.png",
                    )
                if trim_aux is not None:
                    save_gray(
                        trim_aux["safe_support"],
                        diagnostic_dir / "chroma_safe_support.png",
                    )
                    save_gray(
                        (
                            torch.log(
                                trim_aux["applied_gain"].clamp_min(1.0)
                            )
                            / math.log(2.0)
                        ).clamp(0.0, 1.0),
                        diagnostic_dir / "chroma_gain_log2.png",
                    )
            print(f"[{index}/{len(pairs)}] {destination}")


if __name__ == "__main__":
    main()
