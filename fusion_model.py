from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from credibility.models.prob_fusion_calibrated import (
    ProbabilisticCalibratedFusion,
)
# -------------------------
# 基础：颜色空间
# -------------------------


def rgb_to_y(vis: torch.Tensor) -> torch.Tensor:
    r, g, b = vis[:, 0:1], vis[:, 1:2], vis[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def brightness_score(y_raw: torch.Tensor) -> torch.Tensor:
    return y_raw.mean(dim=(2, 3), keepdim=True)


def rgb_to_ycbcr(vis: torch.Tensor):
    assert vis.shape[1] == 3, "vis 必须是 3 通道 RGB"
    r, g, b = vis[:, 0:1], vis[:, 1:2], vis[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return y, cb, cr


def ycbcr_to_rgb(y: torch.Tensor, cb: torch.Tensor, cr: torch.Tensor) -> torch.Tensor:
    if cb.shape[2:] != y.shape[2:]:
        cb = F.interpolate(cb, size=y.shape[2:], mode="bilinear", align_corners=False)
    if cr.shape[2:] != y.shape[2:]:
        cr = F.interpolate(cr, size=y.shape[2:], mode="bilinear", align_corners=False)
    r = y + 1.402 * (cr - 0.5)
    g = y - 0.344136 * (cb - 0.5) - 0.714136 * (cr - 0.5)
    b = y + 1.772 * (cb - 0.5)
    rgb = torch.cat([r, g, b], dim=1)
    return torch.clamp(rgb, 0.0, 1.0)


# -------------------------
# LYT：预训练增强
# -------------------------


class PretrainedLYTWithFeatures(nn.Module):
    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[Union[torch.device, str]] = None,
        lyt_root: Optional[str] = None,
    ):
        super().__init__()
        self.weights_path = Path(weights_path) if weights_path else None
        self.device = torch.device(device) if device is not None else None
        self.lyt_root = Path(lyt_root) if lyt_root else None
        self._model: Optional[nn.Module] = None

    def _resolve_lyt_root(self) -> Path:
        if self.lyt_root is not None:
            return self.lyt_root
        return Path(__file__).resolve().parent / "third_party" / "LYT"

    def _resolve_weights(self) -> Path:
        if self.weights_path is not None:
            return self.weights_path
        return Path(__file__).resolve().parent / "checkpoints" / "lyt.pth"

    def _lazy_load(self, runtime_device: torch.device):
        if self._model is not None:
            return

        target = self.device or runtime_device or torch.device("cpu")
        self.device = target

        model_path = self._resolve_lyt_root() / "model.py"
        if not model_path.exists():
            raise FileNotFoundError(f"LYT model not found at {model_path}")

        spec = importlib.util.spec_from_file_location("lyt_model", model_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load LYT model from {model_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        lyt_cls = getattr(mod, "LYT")

        self._model = lyt_cls().to(target)
        for p in self._model.parameters():
            p.requires_grad = False
        self._model.eval()

        # load weights
        try:
            state = torch.load(str(self._resolve_weights()), map_location=target, weights_only=True)
        except TypeError:
            state = torch.load(str(self._resolve_weights()), map_location=target)
        self._model.load_state_dict(state)

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        runtime_device = x.device
        self._lazy_load(runtime_device)
        assert self._model is not None

        h, w = x.shape[-2:]
        if h * w > 512 * 512:  # 大图分块, 避免 LYT 自注意力 O(N²) 爆显存
            return self._forward_tiled(x, runtime_device)
        y = torch.clamp(self._model(x.to(self.device)), 0.0, 1.0)
        return y.to(runtime_device)

    def _forward_tiled(self, x, runtime_device, tile=512, overlap=64):
        b, c, h, w = x.shape
        device = self.device or runtime_device
        self._lazy_load(device)
        out = torch.zeros(b, c, h, w, device=device)
        weight = torch.zeros(b, c, h, w, device=device)
        ys = list(range(0, h, tile - overlap))
        xs = list(range(0, w, tile - overlap))
        if ys[-1] + tile < h: ys.append(h - tile)
        if xs[-1] + tile < w: xs.append(w - tile)
        for y0 in ys:
            y1 = min(y0 + tile, h)
            for x0 in xs:
                x1 = min(x0 + tile, w)
                tile_out = torch.clamp(self._model(x[:, :, y0:y1, x0:x1].to(device)), 0.0, 1.0)
                out[:, :, y0:y1, x0:x1] += tile_out
                weight[:, :, y0:y1, x0:x1] += 1.0
        return (out / weight.clamp_min(1.0)).to(runtime_device)


class LumaGate(nn.Module):
    """Interpretable raw/LYT precision gate."""

    def __init__(self):
        super().__init__()
        self.credibility = ProbabilisticCalibratedFusion()

    def forward(self, y_in: torch.Tensor, y_lyt: torch.Tensor, infra: torch.Tensor):

        gY, confidence, diagnostics, _, _ = self.credibility(
            y_in, y_lyt, infra
        )
        y_mix = y_in + gY * (y_lyt - y_in)
        diagnostics["confidence_total"] = confidence
        return y_mix, gY, diagnostics

# -------------------------
# 主模型：FusionSerialBridge
# -------------------------


class FusionSerialBridge(nn.Module):
    def __init__(
        self,
        use_lyt: bool = True,
        lyt_weights: Optional[str] = None,
        lyt_device: Optional[Union[torch.device, str]] = None,
        lyt_root: Optional[str] = None,
        use_credibility: bool = True,
        credibility_weights: Optional[str] = None,
        main_credibility_weights: Optional[str] = None,
        use_polar_chroma: bool = False,
        feat_ch: int = 32,
    ):
        super().__init__()

        # ---- import your project modules ----
        from CGE import LIP
        from GetMap import GetGradMap
        from SFE import Encode
        from TGE import GIP

        self.GetMap = GetGradMap
        self.encode = Encode()
        self.LIP = LIP()
        self.GIP = GIP()

        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels=feat_ch, out_channels=feat_ch, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=feat_ch, out_channels=1, kernel_size=3, stride=1, padding=1),
        )

        # Sample-level toggle (same as your code)
        self.lyt_brightness_thr = 0.31
        self.use_lyt = bool(use_lyt)
        self.lyt_enhancer = PretrainedLYTWithFeatures(lyt_weights, lyt_device, lyt_root) if use_lyt else None

        self.luma_gate = LumaGate()
        self.credibility = ProbabilisticCalibratedFusion() if use_credibility else None
        if self.credibility is not None and credibility_weights is not None:
            self.load_credibility_weights(credibility_weights)
        if (
            self.credibility is not None
            and main_credibility_weights is not None
        ):
            self.load_main_credibility_weights(main_credibility_weights)

        self.use_polar_chroma = bool(use_polar_chroma)
        if use_polar_chroma:
            from chroma_polar import ChromaPolarPhysical
            self.chroma_head = ChromaPolarPhysical(base_ch=feat_ch)
        else:
            self.chroma_head = None

    def _compatible_credibility_state(self, weights_path: str):
        try:
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if any(key.startswith("calibrator.") for key in state):
            state = {
                key.removeprefix("calibrator."): value
                for key, value in state.items()
                if key.startswith("calibrator.")
            }
        if self.credibility is None:
            raise RuntimeError("Credibility is disabled")
        target = self.credibility.calibrator.state_dict()
        compatible = {
            key: value for key, value in state.items()
            if key in target and target[key].shape == value.shape
        }
        if not compatible:
            raise RuntimeError(
                "The checkpoint uses the retired CNN calibrator. Retrain MICG "
                "with credibility/train/train_calibrator_v2.py."
            )
        if len(compatible) != len(target):
            missing = sorted(set(target).difference(compatible))
            raise RuntimeError(
                "Incomplete credibility checkpoint; missing: "
                + ", ".join(missing[:10])
            )
        return compatible

    def load_credibility_weights(self, weights_path: str) -> None:
        """Backward-compatible shared MICG-1/MICG-2 weight loading."""
        compatible = self._compatible_credibility_state(weights_path)
        self.credibility.calibrator.load_state_dict(compatible, strict=False)
        self.luma_gate.credibility.calibrator.load_state_dict(
            compatible, strict=False
        )

    def load_main_credibility_weights(self, weights_path: str) -> None:
        """Load the independently trained MICG-2 final-luminance gate."""
        compatible = self._compatible_credibility_state(weights_path)
        self.credibility.calibrator.load_state_dict(compatible, strict=True)

    def forward(
        self,
        vis: torch.Tensor,
        infra: torch.Tensor,
        return_diagnostics: bool = False,
    ):
        # ---- split ----
        y_in, cb_in, cr_in = rgb_to_ycbcr(vis)
        meanY = brightness_score(y_in)

        enable_lyt = (meanY < self.lyt_brightness_thr) & self.use_lyt

        # ---- LYT forward (pretrained, no_grad) ----
        vis_lyt = vis
        if enable_lyt.any() and self.lyt_enhancer is not None:
            vis_lyt = self.lyt_enhancer(vis)

        # LYT candidates (pixel-level)
        y_lyt = rgb_to_y(vis_lyt)

        # if disabled, fall back to input channels
        y_lyt = torch.where(enable_lyt, y_lyt, y_in)

        # ---- luma gate (same as your current stage1) ----
        y_mix, gY, enhancement_diagnostics = self.luma_gate(
            y_in, y_lyt, infra
        )
        y_en = y_mix

        # ---- TCDG encode ----
        mask = self.GetMap(y_en, infra)
        x = torch.cat([y_en, infra], dim=1)
        Fs, Fm, Fd = self.encode(x)
        F_m = Fm * (1 - mask)
        F_d = Fd * mask

        # ---- original TCDG-style heads ----
        Ec = self.LIP(Fs, F_m)

        # texture branch (keep your code shape)
        size = [Fs.shape[2], Fs.shape[3]]
        fea1 = F.interpolate(Fs, scale_factor=0.25, mode="nearest")
        fea2 = F.interpolate(F_d, scale_factor=0.25, mode="nearest")
        map_size = [fea1.shape[2], fea1.shape[3]]
        mask1 = F.interpolate(mask, size=map_size, mode="nearest")
        fea_g = self.GIP(fea1, fea2, mask1, size) + Fs + F_d
        Et = self.fc2(fea_g)

        y_tcdg = ((1 - mask) * Ec + mask * Et).clamp(0.0, 1.0)

        # ---- calibrated credibility weighting ----
        if self.credibility is not None:
            w_tcdg, _, fusion_diagnostics, _, _ = self.credibility(
                y_en, y_tcdg, infra
            )
            y_fused = w_tcdg * y_tcdg + (1 - w_tcdg) * y_en
        else:
            w_tcdg = torch.ones_like(y_tcdg)
            fusion_diagnostics = {}
            y_fused = y_tcdg

        # ---- chroma head ----
        if self.chroma_head is not None:
            Y_for_chroma = y_fused
            cb_out, cr_out, _ = self.chroma_head(
                Y_in=y_in, Y_out=Y_for_chroma,
                Cb_in=cb_in, Cr_in=cr_in, IR=infra)
        else:
            cb_out, cr_out = cb_in, cr_in

        rgb_out = ycbcr_to_rgb(y_fused, cb_out, cr_out)

        if not return_diagnostics:
            return rgb_out
        return rgb_out, {
            "y_in": y_in,
            "y_lyt": y_lyt,
            "y_en": y_en,
            "y_tcdg": y_tcdg,
            "y_fused": y_fused,
            "weight_lyt": gY,
            "weight_tcdg": w_tcdg,
            "enhancement": enhancement_diagnostics,
            "fusion": fusion_diagnostics,
        }
