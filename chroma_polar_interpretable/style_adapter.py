"""Training-free, source-safe chroma style controls."""

from __future__ import annotations

import math

import torch

from chroma_polar_interpretable.model import chroma_magnitude


def _srgb_to_linear(rgb: torch.Tensor) -> torch.Tensor:
    return torch.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055).pow(2.4),
    )


def _linear_rgb_to_ipt(rgb: torch.Tensor) -> torch.Tensor:
    """Convert linear sRGB to IPT using the standard fixed transforms."""
    rgb_to_xyz = rgb.new_tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz_to_lms = rgb.new_tensor([
        [0.4002, 0.7075, -0.0807],
        [-0.2280, 1.1500, 0.0612],
        [0.0000, 0.0000, 0.9184],
    ])
    lms_to_ipt = rgb.new_tensor([
        [0.4000, 0.4000, 0.2000],
        [4.4550, -4.8510, 0.3960],
        [0.8056, 0.3572, -1.1628],
    ])
    xyz = torch.einsum("ij,bjhw->bihw", rgb_to_xyz, rgb)
    lms = torch.einsum("ij,bjhw->bihw", xyz_to_lms, xyz)
    lms_prime = lms.sign() * lms.abs().clamp_min(1e-8).pow(0.43)
    return torch.einsum("ij,bjhw->bihw", lms_to_ipt, lms_prime)


def _ycbcr_to_rgb_unclamped(
    y: torch.Tensor,
    cb: torch.Tensor,
    cr: torch.Tensor,
) -> torch.Tensor:
    u = cb - 0.5
    v = cr - 0.5
    return torch.cat([
        y + 1.402 * v,
        y - 0.344136 * u - 0.714136 * v,
        y + 1.772 * u,
    ], dim=1)


def _fixed_y_gamut_limit(
    y: torch.Tensor,
    direction_u: torch.Tensor,
    direction_v: torch.Tensor,
) -> torch.Tensor:
    """Largest chroma radius whose fixed-Y RGB remains inside [0, 1]."""
    slopes = (
        1.402 * direction_v,
        -0.344136 * direction_u - 0.714136 * direction_v,
        1.772 * direction_u,
    )
    limits = []
    for slope in slopes:
        positive_limit = (1.0 - y) / slope.clamp_min(1e-8)
        negative_limit = y / (-slope).clamp_min(1e-8)
        limits.append(torch.where(
            slope > 1e-8,
            positive_limit,
            torch.where(
                slope < -1e-8,
                negative_limit,
                torch.full_like(slope, 1e6),
            ),
        ))
    return torch.minimum(
        torch.minimum(limits[0], limits[1]),
        limits[2],
    ).clamp_min(0.0)


def source_safe_ich_chroma_trim(
    cb_sahca: torch.Tensor,
    cr_sahca: torch.Tensor,
    visible_rgb: torch.Tensor,
    y_fused: torch.Tensor,
    source_confidence: torch.Tensor,
    ir_only: torch.Tensor,
    visible_noise: torch.Tensor,
    clipping_risk: torch.Tensor,
    strength: float,
    maximum_gain: float = float("inf"),
):
    """Apply an Old-Polar-like vividness trim without an old model.

    A parameter-free ICh saturation target is estimated from the visible
    image and the fixed-luminance SAHCA result. Only positive corrections are
    retained, optionally bounded by ``maximum_gain`` and blended in log-chroma
    space according to visible-source safety. The production default is
    unbounded; source safety and the fixed-Y RGB gamut remain active. Visible
    hue and fused luminance are preserved by construction.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    if maximum_gain < 1.0:
        raise ValueError("maximum_gain must be >= 1")

    eps = 1e-5
    _, cb_visible, cr_visible = _rgb_to_ycbcr(visible_rgb)
    chroma_sahca, _, _ = chroma_magnitude(
        cb_sahca, cr_sahca, eps
    )
    chroma_visible, visible_u, visible_v = chroma_magnitude(
        cb_visible, cr_visible, eps
    )

    sahca_rgb = _ycbcr_to_rgb_unclamped(
        y_fused, cb_sahca, cr_sahca
    ).clamp(0.0, 1.0)
    ipt_visible = _linear_rgb_to_ipt(
        _srgb_to_linear(visible_rgb.clamp(0.0, 1.0))
    )
    ipt_sahca = _linear_rgb_to_ipt(
        _srgb_to_linear(sahca_rgb)
    )
    i_visible = ipt_visible[:, 0:1].abs().clamp_min(1e-4)
    i_sahca = ipt_sahca[:, 0:1].abs().clamp_min(1e-4)
    c_visible_ipt = torch.sqrt(
        ipt_visible[:, 1:2].square()
        + ipt_visible[:, 2:3].square()
        + eps
    )
    c_sahca_ipt = torch.sqrt(
        ipt_sahca[:, 1:2].square()
        + ipt_sahca[:, 2:3].square()
        + eps
    )

    c_prime = c_sahca_ipt * i_visible / i_sahca
    saturation_visible = c_visible_ipt / torch.sqrt(
        c_visible_ipt.square() + i_visible.square() + eps
    )
    saturation_prime = c_prime / torch.sqrt(
        c_prime.square() + i_sahca.square() + eps
    )
    c_parameter_free = (
        c_prime * saturation_visible
        / saturation_prime.clamp_min(eps)
    )
    parameter_free_gain = (
        c_parameter_free / c_sahca_ipt.clamp_min(eps)
    ).clamp(1.0, float(maximum_gain))

    # Confidence is square-root tempered because ir_only already expresses
    # the strongest source-ownership veto. This avoids counting it twice.
    safe_support = (
        source_confidence.clamp(0.0, 1.0).sqrt()
        * (1.0 - ir_only).clamp(0.0, 1.0)
        * (1.0 - clipping_risk).clamp(0.0, 1.0)
        * (1.0 - 0.5 * visible_noise.clamp(0.0, 1.0))
    )
    style_weight = float(strength) * safe_support
    applied_gain = torch.exp(
        style_weight * torch.log(parameter_free_gain)
    )
    chroma_styled = chroma_sahca * applied_gain

    direction_u = visible_u / chroma_visible.clamp_min(eps)
    direction_v = visible_v / chroma_visible.clamp_min(eps)
    maximum_chroma = _fixed_y_gamut_limit(
        y_fused, direction_u, direction_v
    )
    # Never desaturate the existing SAHCA result. Pixels already outside the
    # strict fixed-Y RGB gamut keep their original chroma and receive no
    # further boost; valid pixels are projected to the exact gamut boundary.
    has_gamut_room = maximum_chroma >= chroma_sahca
    chroma_styled = torch.where(
        has_gamut_room,
        torch.minimum(chroma_styled, maximum_chroma),
        chroma_sahca,
    )

    cb_styled = 0.5 + chroma_styled * direction_u
    cr_styled = 0.5 + chroma_styled * direction_v
    if float(strength) == 0.0:
        cb_styled = cb_sahca
        cr_styled = cr_sahca
        chroma_styled = chroma_sahca
    diagnostics = {
        "style_weight": style_weight,
        "safe_support": safe_support,
        "parameter_free_gain": parameter_free_gain,
        "applied_gain": (
            chroma_styled / chroma_sahca.clamp_min(eps)
        ),
        "chroma_sahca": chroma_sahca,
        "chroma_styled": chroma_styled,
        "maximum_chroma": maximum_chroma,
        "ipt_intensity_visible": i_visible,
        "ipt_intensity_sahca": i_sahca,
        "ipt_chroma_visible": c_visible_ipt,
        "ipt_chroma_sahca": c_sahca_ipt,
        "ipt_chroma_target": c_parameter_free,
    }
    return (
        cb_styled.clamp(0.0, 1.0),
        cr_styled.clamp(0.0, 1.0),
        diagnostics,
    )


def _rgb_to_ycbcr(rgb: torch.Tensor):
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return y, cb, cr


def safe_old_polar_style(
    cb_sahca: torch.Tensor,
    cr_sahca: torch.Tensor,
    cb_old: torch.Tensor,
    cr_old: torch.Tensor,
    cb_visible: torch.Tensor,
    cr_visible: torch.Tensor,
    ir_only: torch.Tensor,
    clipping_risk: torch.Tensor,
    strength: float,
    minimum_ratio: float = 0.5,
    maximum_ratio: float = 2.0,
):
    """Move SAHCA chroma magnitude toward Old Polar without changing hue.

    ``strength=0`` returns SAHCA. ``strength=1`` adopts the Old Polar chroma
    magnitude only where visible support is safe. Infrared-only structures and
    clipping-risk regions remain protected.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")

    eps = 1e-5
    chroma_sahca, _, _ = chroma_magnitude(cb_sahca, cr_sahca, eps)
    chroma_old, _, _ = chroma_magnitude(cb_old, cr_old, eps)
    chroma_visible, visible_u, visible_v = chroma_magnitude(
        cb_visible, cr_visible, eps
    )

    log_style_ratio = (
        torch.log(chroma_old.clamp_min(eps))
        - torch.log(chroma_sahca.clamp_min(eps))
    ).clamp(math.log(minimum_ratio), math.log(maximum_ratio))
    safe_support = (
        (1.0 - ir_only).clamp(0.0, 1.0)
        * (1.0 - clipping_risk).clamp(0.0, 1.0)
    )
    style_weight = float(strength) * safe_support
    chroma_styled = chroma_sahca * torch.exp(
        style_weight * log_style_ratio
    )

    # Preserve the visible chroma direction by construction.
    direction_u = visible_u / chroma_visible.clamp_min(eps)
    direction_v = visible_v / chroma_visible.clamp_min(eps)

    # Radial gamut projection avoids channel clipping while retaining hue.
    max_u = 0.5 / direction_u.abs().clamp_min(eps)
    max_v = 0.5 / direction_v.abs().clamp_min(eps)
    max_chroma = torch.minimum(max_u, max_v)
    chroma_styled = torch.minimum(chroma_styled, max_chroma)

    cb_styled = 0.5 + chroma_styled * direction_u
    cr_styled = 0.5 + chroma_styled * direction_v
    diagnostics = {
        "style_weight": style_weight,
        "safe_support": safe_support,
        "chroma_sahca": chroma_sahca,
        "chroma_old": chroma_old,
        "chroma_styled": chroma_styled,
        "log_style_ratio": log_style_ratio,
    }
    return (
        cb_styled.clamp(0.0, 1.0),
        cr_styled.clamp(0.0, 1.0),
        diagnostics,
    )
