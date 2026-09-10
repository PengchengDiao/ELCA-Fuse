"""Monotonic, interpretable calibration for probabilistic luminance fusion.

The physical estimator supplies quality and variance estimates.  MICG learns
only bounded, monotonic corrections whose individual feature contributions can
be inspected directly.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from credibility.models.prob_fusion import (
    ProbabilisticCredibilityFusion,
    lyt_noise_propagate,
    scharr,
    structure_tensor,
)


class MonotonicShape1D(nn.Module):
    """Piecewise-linear monotonic shape function on ``[0, 1]``."""

    def __init__(self, knots=8, direction=1, max_scale=1.0):
        super().__init__()
        if direction not in (-1, 1):
            raise ValueError("direction must be -1 or 1")
        self.knots = int(knots)
        self.direction = float(direction)
        self.max_scale = float(max_scale)
        self.raw_increments = nn.Parameter(torch.full((self.knots - 1,), -5.0))
        self.raw_scale = nn.Parameter(torch.tensor(-2.0))

    def knot_values(self):
        increments = F.softplus(self.raw_increments)
        values = torch.cat([increments.new_zeros(1), torch.cumsum(increments, 0)])
        values = values / values[-1].clamp_min(1e-6)
        values = values - values.mean()
        scale = self.max_scale * torch.sigmoid(self.raw_scale)
        return self.direction * scale * values

    def forward(self, x):
        x = x.clamp(0.0, 1.0)
        position = x * (self.knots - 1)
        left = position.floor().long().clamp(0, self.knots - 2)
        right = left + 1
        fraction = position - left.to(position.dtype)
        values = self.knot_values()
        return values[left] * (1.0 - fraction) + values[right] * fraction


class AdditiveMonotonicHead(nn.Module):
    """Sum of named feature contributions with guaranteed directions."""

    def __init__(self, specifications, knots=8, max_scale=1.0):
        super().__init__()
        self.shapes = nn.ModuleDict({
            name: MonotonicShape1D(knots, direction, max_scale)
            for name, direction in specifications.items()
        })
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, features):
        contributions = {
            name: shape(features[name])
            for name, shape in self.shapes.items()
        }
        total = self.bias.view(1, 1, 1, 1)
        for value in contributions.values():
            total = total + value
        return total, contributions


class InterpretableCredibilityCalibrator(nn.Module):
    """Physical baseline plus auditable monotonic corrections."""

    def __init__(self, lambda_q=1.0, variance_scale=4.0, knots=8):
        super().__init__()
        self.lambda_q = float(lambda_q)
        self.lambda_v = math.log(float(variance_scale))

        self.q_in_head = AdditiveMonotonicHead({
            "q_in": +1,
            "k_in": +1,
            "darkness": -1,
        }, knots, max_scale=0.75)
        self.q_lyt_head = AdditiveMonotonicHead({
            "q_lyt": +1,
            "k_lyt": +1,
            "edge_agreement": +1,
            "clean_gain": +1,
        }, knots, max_scale=0.75)
        self.v_in_head = AdditiveMonotonicHead({
            "v_in_norm": +1,
            "darkness": +1,
        }, knots, max_scale=0.50)
        self.v_lyt_head = AdditiveMonotonicHead({
            "v_lyt_norm": +1,
            "artifact_ratio": +1,
            "clipping_risk": +1,
            "noise_gain": +1,
        }, knots, max_scale=0.75)

    def forward(self, features, q_in_0, q_lyt_0, v_in_0, v_lyt_0):
        dq_in, cq_in = self.q_in_head(features)
        dq_lyt, cq_lyt = self.q_lyt_head(features)
        dv_in, cv_in = self.v_in_head(features)
        dv_lyt, cv_lyt = self.v_lyt_head(features)

        q_in = torch.sigmoid(
            torch.logit(q_in_0.clamp(1e-4, 1 - 1e-4))
            + self.lambda_q * torch.tanh(dq_in)
        )
        q_lyt = torch.sigmoid(
            torch.logit(q_lyt_0.clamp(1e-4, 1 - 1e-4))
            + self.lambda_q * torch.tanh(dq_lyt)
        )
        v_in = (
            v_in_0 * torch.exp(self.lambda_v * torch.tanh(dv_in))
        ).clamp(1e-3, 4.0)
        v_lyt = (
            v_lyt_0 * torch.exp(self.lambda_v * torch.tanh(dv_lyt))
        ).clamp(1e-3, 4.0)

        return {
            "q_in": q_in,
            "q_lyt": q_lyt,
            "v_in": v_in,
            "v_lyt": v_lyt,
            "contributions": {
                "q_in": cq_in,
                "q_lyt": cq_lyt,
                "v_in": cv_in,
                "v_lyt": cv_lyt,
            },
            "features": features,
        }


# Backward-compatible import name; the implementation is no longer a CNN.
ResidualCredibilityCalibrator = InterpretableCredibilityCalibrator


def edge_aware_refine(weight, guide, strength=0.35, beta=12.0, steps=3):
    """Unrolled differentiable diffusion with boundaries defined by ``guide``."""
    estimate = weight
    for _ in range(int(steps)):
        padded_w = F.pad(estimate, (1, 1, 1, 1), mode="replicate")
        padded_g = F.pad(guide, (1, 1, 1, 1), mode="replicate")
        neighbours_w = (
            padded_w[:, :, 1:-1, :-2],
            padded_w[:, :, 1:-1, 2:],
            padded_w[:, :, :-2, 1:-1],
            padded_w[:, :, 2:, 1:-1],
        )
        neighbours_g = (
            padded_g[:, :, 1:-1, :-2],
            padded_g[:, :, 1:-1, 2:],
            padded_g[:, :, :-2, 1:-1],
            padded_g[:, :, 2:, 1:-1],
        )
        affinities = tuple(
            torch.exp(-beta * (guide - neighbour).abs())
            for neighbour in neighbours_g
        )
        weighted_sum = sum(
            affinity * neighbour
            for affinity, neighbour in zip(affinities, neighbours_w)
        )
        numerator = weight + strength * weighted_sum
        denominator = 1.0 + strength * sum(affinities)
        estimate = numerator / denominator.clamp_min(1e-6)
    return estimate


def compute_T_lyt(
    q_in,
    v_in,
    q_lyt,
    v_lyt,
    tau_c=0.0,
    T_c=1.0,
    sigma_rho=3,
    guide=None,
):
    """Convert calibrated quality/variance into precision weights."""
    eps = 1e-6
    precision_in = q_in / (v_in + eps)
    precision_lyt = q_lyt / (v_lyt + eps)
    log_odds = torch.log(precision_lyt + eps) - torch.log(precision_in + eps)
    weight_lyt = torch.sigmoid(log_odds)
    if guide is None:
        weight_lyt = F.avg_pool2d(
            weight_lyt, 2 * sigma_rho + 1, 1, sigma_rho
        )
    else:
        weight_lyt = edge_aware_refine(weight_lyt, guide)

    total_log_precision = torch.log(precision_in + precision_lyt + eps)
    confidence_total = torch.sigmoid((total_log_precision - tau_c) / T_c)
    return weight_lyt, confidence_total


def extract_physical_features(phys, y_in, y_lyt, ir):
    """Extract named, normalised observables used by the additive heads."""
    with torch.no_grad():
        y_in = y_in.clamp(0.0, 1.0)
        y_lyt = y_lyt.clamp(0.0, 1.0)
        ir = ir.clamp(0.0, 1.0)
        T_base, C_base, q_in, v_in, q_lyt, v_lyt = phys(y_in, y_lyt, ir)
        _, k_in, _ = structure_tensor(y_in)
        _, k_lyt, _ = structure_tensor(y_lyt)
        _, v_art, energy_lyt, gain = lyt_noise_propagate(y_in, y_lyt)

        artifact_ratio = (v_art / (energy_lyt + 1e-6)).clamp(0.0, 1.0)
        gx_lyt, gy_lyt = scharr(y_lyt)
        gx_ir, gy_ir = scharr(ir)
        dot = gx_lyt * gx_ir + gy_lyt * gy_ir
        norm = torch.sqrt(
            (gx_lyt.square() + gy_lyt.square())
            * (gx_ir.square() + gy_ir.square()) + 1e-8
        )
        edge_agreement = ((dot / norm).clamp(-1.0, 1.0) + 1.0) * 0.5

        delta = (y_lyt - y_in).abs()
        darkness = (1.0 - y_in).clamp(0.0, 1.0)
        clipping_risk = (
            F.relu(y_lyt - 0.98) + F.relu(0.02 - y_lyt)
        ).mul(50.0).clamp(0.0, 1.0)
        noise_gain = (
            torch.log1p(gain) / torch.log(gain.new_tensor(9.0))
        ).clamp(0.0, 1.0)
        clean_gain = (
            darkness
            * (1.0 - artifact_ratio)
            * (delta / 0.5).clamp(0.0, 1.0)
        )

        features = {
            "q_in": q_in.clamp(0.0, 1.0),
            "q_lyt": q_lyt.clamp(0.0, 1.0),
            "v_in_norm": (
                torch.log1p(v_in) / torch.log(v_in.new_tensor(5.0))
            ).clamp(0.0, 1.0),
            "v_lyt_norm": (
                torch.log1p(v_lyt) / torch.log(v_lyt.new_tensor(5.0))
            ).clamp(0.0, 1.0),
            "k_in": k_in,
            "k_lyt": k_lyt,
            "artifact_ratio": artifact_ratio,
            "noise_gain": noise_gain,
            "darkness": darkness,
            "clipping_risk": clipping_risk,
            "edge_agreement": edge_agreement,
            "clean_gain": clean_gain,
        }

    return features, (q_in, v_in, q_lyt, v_lyt, T_base, C_base)


class ProbabilisticCalibratedFusion(nn.Module):
    def __init__(
        self,
        hidden=16,
        lambda_q=1.0,
        variance_scale=4.0,
        tau_c=0.0,
        T_c=1.0,
        knots=8,
    ):
        super().__init__()
        del hidden  # accepted for compatibility with existing configuration
        self.phys = ProbabilisticCredibilityFusion()
        self.phys.eval()
        for parameter in self.phys.parameters():
            parameter.requires_grad_(False)

        self.calibrator = InterpretableCredibilityCalibrator(
            lambda_q=lambda_q,
            variance_scale=variance_scale,
            knots=knots,
        )
        self.tau_c = tau_c
        self.T_c = T_c

    def forward(self, y_in, y_lyt, ir):
        features, physical = extract_physical_features(
            self.phys, y_in, y_lyt, ir
        )
        q_in_0, v_in_0, q_lyt_0, v_lyt_0, T_base, C_base = physical
        output = self.calibrator(
            features, q_in_0, q_lyt_0, v_in_0, v_lyt_0
        )
        weight_lyt, confidence_total = compute_T_lyt(
            output["q_in"],
            output["v_in"],
            output["q_lyt"],
            output["v_lyt"],
            tau_c=self.tau_c,
            T_c=self.T_c,
            guide=y_in,
        )
        output["precision_in"] = output["q_in"] / (output["v_in"] + 1e-6)
        output["precision_lyt"] = output["q_lyt"] / (output["v_lyt"] + 1e-6)
        output["weight_lyt"] = weight_lyt
        output["confidence_total"] = confidence_total
        return weight_lyt, confidence_total, output, T_base, C_base
