"""Core implementation of SA-LCID.

SA-LCID (Source-Aware Luminance--Chroma Intervention Diagnostic) evaluates
registered visible/infrared/fused triplets after the fused result has been
reconstructed at a controlled luminance.  This module is the cleaned public
version of the project's earlier ``SA-LCI v2`` implementation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, sobel


EPS = 1e-8


@dataclass(frozen=True)
class DiagnosticConfig:
    """Declared constants used by SA-LCID."""

    visible_black_level: float = 0.02
    visible_reliable_level: float = 0.18
    snr_low: float = 1.0
    snr_high: float = 8.0
    chroma_floor: float = 0.02
    alignment_tolerance: int = 5
    chroma_smoothing_sigma: float = 1.5
    attenuation_tolerance: float = 0.90
    overshoot_tolerance: float = 1.25

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def as_float_rgb(image: np.ndarray) -> np.ndarray:
    """Convert an HxWx3 image to clipped float64 RGB in [0, 1]."""

    value = np.asarray(image)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("Expected an HxWx3 RGB image")
    value = value.astype(np.float64)
    if value.max(initial=0.0) > 1.0:
        value /= 255.0
    return np.clip(value, 0.0, 1.0)


def as_float_gray(image: np.ndarray) -> np.ndarray:
    """Convert an HxW or HxWx3 image to clipped float64 gray in [0, 1]."""

    value = np.asarray(image)
    if value.ndim == 3 and value.shape[2] >= 3:
        value = value[..., :3].astype(np.float64)
        if value.max(initial=0.0) > 1.0:
            value /= 255.0
        value = (
            0.2126 * value[..., 0]
            + 0.7152 * value[..., 1]
            + 0.0722 * value[..., 2]
        )
    elif value.ndim == 2:
        value = value.astype(np.float64)
        if value.max(initial=0.0) > 1.0:
            value /= 255.0
    else:
        raise ValueError("Expected an HxW or HxWx3 image")
    return np.clip(value, 0.0, 1.0)


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = as_float_rgb(rgb)
    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    value = np.asarray(rgb, dtype=np.float64)
    positive = np.maximum(value, 0.0)
    encoded = np.where(
        positive <= 0.0031308,
        12.92 * positive,
        1.055 * positive ** (1.0 / 2.4) - 0.055,
    )
    return np.clip(encoded, 0.0, 1.0)


def relative_luminance(rgb: np.ndarray) -> np.ndarray:
    linear = srgb_to_linear(rgb)
    return (
        0.2126 * linear[..., 0]
        + 0.7152 * linear[..., 1]
        + 0.0722 * linear[..., 2]
    )


def reconstruct_with_fixed_luminance(
    chroma_source_rgb: np.ndarray,
    fixed_luminance: np.ndarray,
) -> np.ndarray:
    """Place source chromaticity on a requested linear-light luminance.

    Linear RGB is first scaled along its ray from black.  Rays outside the
    sRGB gamut are moved toward the neutral point at the same luminance.
    """

    source_linear = srgb_to_linear(chroma_source_rgb)
    source_y = (
        0.2126 * source_linear[..., 0]
        + 0.7152 * source_linear[..., 1]
        + 0.0722 * source_linear[..., 2]
    )
    target_y = as_float_gray(fixed_luminance)
    if target_y.shape != source_y.shape:
        raise ValueError("fixed_luminance must match the RGB image size")

    chromaticity = source_linear / (source_y[..., None] + EPS)
    neutral = np.ones_like(chromaticity)
    maximum = np.max(chromaticity, axis=2)
    available = (1.0 / np.maximum(target_y, EPS)) - 1.0
    required = np.maximum(maximum - 1.0, EPS)
    gamut_scale = np.where(maximum > 1.0, available / required, 1.0)
    gamut_scale = np.clip(gamut_scale, 0.0, 1.0)
    chromaticity = neutral + gamut_scale[..., None] * (chromaticity - neutral)
    reconstructed = target_y[..., None] * chromaticity
    dark = source_y <= EPS
    reconstructed[dark] = target_y[dark][..., None]
    return linear_to_srgb(reconstructed)


def linear_rgb_to_oklab(linear: np.ndarray) -> np.ndarray:
    rgb = np.asarray(linear, dtype=np.float64)
    lms = np.einsum(
        "...c,dc->...d",
        rgb,
        np.asarray(
            [
                [0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005],
            ]
        ),
    )
    lms_cbrt = np.cbrt(lms)
    return np.einsum(
        "...c,dc->...d",
        lms_cbrt,
        np.asarray(
            [
                [0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660],
            ]
        ),
    )


def oklab_to_linear_rgb(lab: np.ndarray) -> np.ndarray:
    value = np.asarray(lab, dtype=np.float64)
    lms_cbrt = np.einsum(
        "...c,dc->...d",
        value,
        np.asarray(
            [
                [1.0, 0.3963377774, 0.2158037573],
                [1.0, -0.1055613458, -0.0638541728],
                [1.0, -0.0894841775, -1.2914855480],
            ]
        ),
    )
    lms = lms_cbrt**3
    return np.einsum(
        "...c,dc->...d",
        lms,
        np.asarray(
            [
                [4.0767416621, -3.3077115913, 0.2309699292],
                [-1.2684380046, 2.6097574011, -0.3413193965],
                [-0.0041960863, -0.7034186147, 1.7076147010],
            ]
        ),
    )


def oklch(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lab = linear_rgb_to_oklab(srgb_to_linear(rgb))
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    hue = np.arctan2(lab[..., 2], lab[..., 1])
    return lab[..., 0], chroma, hue


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image, dtype=np.float64)
    gx = sobel(value, axis=1, mode="reflect") / 8.0
    gy = sobel(value, axis=0, mode="reflect") / 8.0
    return np.hypot(gx, gy)


def _weighted_mean(value: np.ndarray, weight: np.ndarray) -> float:
    denominator = float(np.sum(weight))
    if denominator <= EPS:
        return float("nan")
    return float(np.sum(value * weight) / denominator)


def _robust_response(value: np.ndarray) -> np.ndarray:
    low = float(np.percentile(value, 70.0))
    high = float(np.percentile(value, 99.0))
    return np.clip((value - low) / max(high - low, EPS), 0.0, 1.0)


def persistent_edge(image: np.ndarray) -> np.ndarray:
    """Return noise-resistant structure present at fine and coarse scales."""

    fine = gradient_magnitude(gaussian_filter(image, sigma=1.2, mode="reflect"))
    coarse = gradient_magnitude(gaussian_filter(image, sigma=2.5, mode="reflect"))
    return np.sqrt(_robust_response(fine) * _robust_response(coarse))


def independent_source_weights(
    visible_rgb: np.ndarray,
    infrared: np.ndarray,
    config: DiagnosticConfig = DiagnosticConfig(),
) -> Mapping[str, np.ndarray]:
    """Build method-independent masks from a registered source pair."""

    visible = as_float_rgb(visible_rgb)
    ir = as_float_gray(infrared)
    if visible.shape[:2] != ir.shape:
        raise ValueError("Visible and infrared images must be registered")

    y_visible = relative_luminance(visible)
    _, c_visible, _ = oklch(visible)
    smooth = gaussian_filter(y_visible, sigma=1.2, mode="reflect")
    local_noise = gaussian_filter(
        np.abs(y_visible - smooth), sigma=1.2, mode="reflect"
    )
    local_signal = gaussian_filter(y_visible, sigma=2.0, mode="reflect")
    snr_proxy = local_signal / (local_noise + 0.005)
    light_weight = np.clip(
        (local_signal - config.visible_black_level)
        / (config.visible_reliable_level - config.visible_black_level),
        0.0,
        1.0,
    )
    snr_weight = np.clip(
        (snr_proxy - config.snr_low) / (config.snr_high - config.snr_low),
        0.0,
        1.0,
    )
    visible_reliability = light_weight * snr_weight

    visible_structure = persistent_edge(y_visible)
    infrared_structure = persistent_edge(ir)
    visible_chroma_structure = persistent_edge(
        gaussian_filter(c_visible, sigma=1.0, mode="reflect")
    )
    size = int(config.alignment_tolerance)
    visible_support = maximum_filter(visible_structure, size=size, mode="reflect")
    chroma_support = maximum_filter(
        visible_chroma_structure, size=size, mode="reflect"
    )
    ir_only = infrared_structure * (1.0 - visible_support) ** 2
    ir_only *= (1.0 - chroma_support) ** 2
    ir_only *= np.clip((infrared_structure - 0.15) / 0.85, 0.0, 1.0)

    return {
        "visible_reliability": visible_reliability,
        "visible_colored": (c_visible >= config.chroma_floor).astype(np.float64),
        "ir_only_structure": ir_only,
        "visible_luminance": y_visible,
        "visible_chroma": c_visible,
        "snr_proxy": snr_proxy,
    }


def diagnose(
    visible_rgb: np.ndarray,
    infrared: np.ndarray,
    fused_rgb: np.ndarray,
    fixed_luminance: np.ndarray | None = None,
    config: DiagnosticConfig = DiagnosticConfig(),
    return_maps: bool = False,
) -> Dict[str, object]:
    """Compute SA-LCID for one registered visible/IR/fused triplet."""

    visible = as_float_rgb(visible_rgb)
    fused = as_float_rgb(fused_rgb)
    ir = as_float_gray(infrared)
    if visible.shape[:2] != fused.shape[:2] or ir.shape != visible.shape[:2]:
        raise ValueError("Visible, infrared and fused images must be registered")

    weights = independent_source_weights(visible, ir, config)
    y_fused = relative_luminance(fused)
    y_fixed = y_fused if fixed_luminance is None else as_float_gray(fixed_luminance)
    if y_fixed.shape != y_fused.shape:
        raise ValueError("fixed_luminance must match the fused image size")

    _, c_visible, h_visible = oklch(visible)
    _, c_fused, h_fused = oklch(fused)
    reliable_color = weights["visible_reliability"] * weights["visible_colored"]
    hue_error = 0.5 * (1.0 - np.cos(h_fused - h_visible))
    ratio = (c_fused + 1e-4) / (c_visible + 1e-4)
    log_ratio = np.log(ratio)

    attenuation_start = -math.log(config.attenuation_tolerance)
    attenuation = np.clip(
        (-log_ratio - attenuation_start) / math.log(4.0), 0.0, 1.0
    )
    overshoot_start = math.log(config.overshoot_tolerance)
    overshoot = np.clip(
        (log_ratio - overshoot_start) / math.log(4.0), 0.0, 1.0
    )

    sigma = config.chroma_smoothing_sigma
    c_visible_smooth = gaussian_filter(c_visible, sigma=sigma, mode="reflect")
    c_fused_smooth = gaussian_filter(c_fused, sigma=sigma, mode="reflect")
    g_visible = gradient_magnitude(c_visible_smooth)
    g_fused = gradient_magnitude(c_fused_smooth)
    induced_structure = np.maximum(g_fused - g_visible, 0.0)
    structure_leakage_map = induced_structure / (g_fused + g_visible + 0.01)

    hf_visible = np.abs(c_visible - c_visible_smooth)
    hf_fused = np.abs(c_fused - c_fused_smooth)
    excess_noise = np.maximum(hf_fused - hf_visible, 0.0)
    chroma_noise_map = excess_noise / (hf_fused + hf_visible + 0.01)
    noise_weight = 0.25 + 0.75 * (1.0 - weights["visible_reliability"])

    components = {
        "visible_hue_damage": _weighted_mean(hue_error, reliable_color),
        "ir_structure_chroma_leakage": _weighted_mean(
            structure_leakage_map, weights["ir_only_structure"]
        ),
        "high_frequency_chroma_noise": _weighted_mean(
            chroma_noise_map, noise_weight
        ),
        "chroma_attenuation_penalty": _weighted_mean(attenuation, reliable_color),
        "chroma_overshoot_penalty": _weighted_mean(overshoot, reliable_color),
    }
    finite = np.asarray(
        [value for value in components.values() if np.isfinite(value)],
        dtype=np.float64,
    )
    result: Dict[str, object] = {
        **components,
        "provisional_artifact_index": (
            float(np.sqrt(np.mean(finite**2))) if finite.size else float("nan")
        ),
        "fixed_luminance_mae": float(np.mean(np.abs(y_fused - y_fixed))),
        "visible_reliable_weight": float(weights["visible_reliability"].mean()),
        "ir_only_weight": float(weights["ir_only_structure"].mean()),
        "mean_log_chroma_response": _weighted_mean(log_ratio, reliable_color),
    }
    if return_maps:
        result["maps"] = {
            "visible_reliability": weights["visible_reliability"],
            "ir_only_structure": weights["ir_only_structure"],
            "structure_leakage": structure_leakage_map,
            "weighted_structure_leakage": (
                structure_leakage_map * weights["ir_only_structure"]
            ),
            "high_frequency_chroma_noise": chroma_noise_map,
            "attenuation": attenuation,
            "overshoot": overshoot,
        }
    return result


def rotate_oklab_hue(rgb: np.ndarray, degrees: float) -> np.ndarray:
    lab = linear_rgb_to_oklab(srgb_to_linear(rgb))
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    a_value = lab[..., 1].copy()
    b_value = lab[..., 2].copy()
    lab[..., 1] = cosine * a_value - sine * b_value
    lab[..., 2] = sine * a_value + cosine * b_value
    return linear_to_srgb(oklab_to_linear_rgb(lab))


def scale_oklab_chroma(rgb: np.ndarray, scale: float) -> np.ndarray:
    lab = linear_rgb_to_oklab(srgb_to_linear(rgb))
    lab[..., 1:] *= float(scale)
    return linear_to_srgb(oklab_to_linear_rgb(lab))


def inject_ir_false_colour(
    rgb: np.ndarray,
    visible_rgb: np.ndarray,
    infrared: np.ndarray,
    strength: float,
    config: DiagnosticConfig = DiagnosticConfig(),
) -> np.ndarray:
    lab = linear_rgb_to_oklab(srgb_to_linear(rgb))
    weight = independent_source_weights(visible_rgb, infrared, config)[
        "ir_only_structure"
    ]
    lab[..., 1] += float(strength) * weight
    lab[..., 2] -= 0.6 * float(strength) * weight
    return linear_to_srgb(oklab_to_linear_rgb(lab))


def inject_chroma_noise(
    rgb: np.ndarray,
    strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    lab = linear_rgb_to_oklab(srgb_to_linear(rgb))
    noise = rng.normal(size=rgb.shape[:2])
    noise -= gaussian_filter(noise, sigma=1.5, mode="reflect")
    noise /= max(float(np.std(noise)), EPS)
    lab[..., 1] += float(strength) * noise
    lab[..., 2] -= 0.6 * float(strength) * noise
    return linear_to_srgb(oklab_to_linear_rgb(lab))

