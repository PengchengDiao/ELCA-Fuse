"""Mechanism-interpretable chroma adaptation for RGB/infrared fusion.

The module never predicts chroma or hue with a free-form CNN.  It:
1. measures the luminance intervention and structural source ownership;
2. applies a bounded Hunt-inspired chroma-magnitude correction;
3. calibrates visible-source confidence with monotonic 1-D shape functions;
4. suppresses chroma changes across infrared-only edges with an unrolled,
   explicitly defined quadratic smoother;
5. preserves the visible chroma direction by construction.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def scharr(x: torch.Tensor):
    kernel_x = x.new_tensor([
        [-3.0, 0.0, 3.0],
        [-10.0, 0.0, 10.0],
        [-3.0, 0.0, 3.0],
    ]).view(1, 1, 3, 3) / 32.0
    kernel_y = kernel_x.transpose(-1, -2)
    padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
    return F.conv2d(padded, kernel_x), F.conv2d(padded, kernel_y)


def chroma_magnitude(cb: torch.Tensor, cr: torch.Tensor, eps=1e-6):
    u = cb - 0.5
    v = cr - 0.5
    return torch.sqrt(u.square() + v.square() + eps), u, v


class MonotonicShape1D(nn.Module):
    """Bounded piecewise-linear response with a guaranteed direction."""

    def __init__(self, knots=8, direction=1, max_scale=1.0):
        super().__init__()
        if direction not in (-1, 1):
            raise ValueError("direction must be -1 or 1")
        self.knots = int(knots)
        self.direction = float(direction)
        self.max_scale = float(max_scale)
        self.raw_increments = nn.Parameter(
            torch.full((self.knots - 1,), -4.0)
        )
        self.raw_scale = nn.Parameter(torch.tensor(-2.0))

    def knot_values(self):
        increments = F.softplus(self.raw_increments)
        values = torch.cat([
            increments.new_zeros(1),
            torch.cumsum(increments, dim=0),
        ])
        values = values / values[-1].clamp_min(1e-6)
        values = values - values.mean()
        scale = self.max_scale * torch.sigmoid(self.raw_scale)
        return self.direction * scale * values

    def forward(self, x):
        position = x.clamp(0.0, 1.0) * (self.knots - 1)
        left = position.floor().long().clamp(0, self.knots - 2)
        fraction = position - left.to(position.dtype)
        values = self.knot_values()
        return (
            values[left] * (1.0 - fraction)
            + values[left + 1] * fraction
        )


class SourceConfidenceCalibrator(nn.Module):
    """Auditable additive model for visible-source chroma confidence."""

    SPECIFICATIONS = {
        "visible_agreement": +1,
        "visible_edge_share": +1,
        "chroma_support": +1,
        "ir_only": -1,
        "visible_noise": -1,
        "clipping_risk": -1,
    }

    def __init__(self, knots=8):
        super().__init__()
        self.shapes = nn.ModuleDict({
            name: MonotonicShape1D(
                knots=knots,
                direction=direction,
                max_scale=0.8,
            )
            for name, direction in self.SPECIFICATIONS.items()
        })
        self.bias = nn.Parameter(torch.tensor(1.0))

    def forward(self, features: Dict[str, torch.Tensor]):
        contributions = {
            name: shape(features[name])
            for name, shape in self.shapes.items()
        }
        logit = self.bias.view(1, 1, 1, 1)
        for contribution in contributions.values():
            logit = logit + contribution
        return torch.sigmoid(logit), contributions


def structural_features(
    y_visible: torch.Tensor,
    y_fused: torch.Tensor,
    infrared: torch.Tensor,
    chroma: torch.Tensor,
):
    gx_v, gy_v = scharr(y_visible)
    gx_f, gy_f = scharr(y_fused)
    gx_i, gy_i = scharr(infrared)

    edge_v = torch.sqrt(gx_v.square() + gy_v.square() + 1e-8)
    edge_f = torch.sqrt(gx_f.square() + gy_f.square() + 1e-8)
    edge_i = torch.sqrt(gx_i.square() + gy_i.square() + 1e-8)

    dot_v = gx_f * gx_v + gy_f * gy_v
    dot_i = gx_f * gx_i + gy_f * gy_i
    agreement_v = (
        dot_v.abs() / (edge_f * edge_v + 1e-6)
    ).clamp(0.0, 1.0)
    agreement_i = (
        dot_i.abs() / (edge_f * edge_i + 1e-6)
    ).clamp(0.0, 1.0)

    visible_presence = edge_v / (edge_v + 0.02)
    infrared_presence = edge_i / (edge_i + 0.02)
    fused_presence = edge_f / (edge_f + 0.02)
    # A flat region has no competing infrared structure and is therefore
    # visible-source safe by default. Confidence falls only for IR-only edges.
    visible_edge_share = (
        1.0 - infrared_presence * (1.0 - visible_presence)
    ).clamp(0.0, 1.0)
    new_fused_structure = (
        F.relu(edge_f - edge_v) / (edge_f + 1e-4)
    ).clamp(0.0, 1.0)
    infrared_share = infrared_presence * (1.0 - visible_presence)
    ir_only = (
        new_fused_structure * infrared_share * agreement_i
    ).clamp(0.0, 1.0)

    local_mean = F.avg_pool2d(y_visible, 5, 1, 2)
    visible_noise = (
        (y_visible - local_mean).abs() / 0.12
    ).clamp(0.0, 1.0)
    clipping_risk = (
        F.relu(0.04 - y_fused) / 0.04
        + F.relu(y_fused - 0.96) / 0.04
    ).clamp(0.0, 1.0)
    chroma_support = (chroma / 0.25).clamp(0.0, 1.0)

    # No fused edge needs no ownership decision, hence confidence defaults to 1.
    visible_agreement = (
        (1.0 - fused_presence)
        + fused_presence * agreement_v * visible_presence
    ).clamp(0.0, 1.0)

    return {
        "visible_agreement": visible_agreement,
        "visible_edge_share": visible_edge_share,
        "chroma_support": chroma_support,
        "ir_only": ir_only,
        "visible_noise": visible_noise,
        "clipping_risk": clipping_risk,
        "edge_visible": edge_v,
        "edge_fused": edge_f,
        "edge_infrared": edge_i,
        "infrared_agreement": agreement_i,
    }


def edge_aware_projection(
    log_chroma_target: torch.Tensor,
    chroma_visible: torch.Tensor,
    ir_only: torch.Tensor,
    strength=0.45,
    beta=24.0,
    steps=3,
):
    """Suppress chroma discontinuities supported only by infrared structure."""
    estimate = log_chroma_target
    data = log_chroma_target
    for _ in range(int(steps)):
        pe = F.pad(estimate, (1, 1, 1, 1), mode="replicate")
        pc = F.pad(chroma_visible, (1, 1, 1, 1), mode="replicate")
        pi = F.pad(ir_only, (1, 1, 1, 1), mode="replicate")
        neighbours_e = (
            pe[:, :, 1:-1, :-2],
            pe[:, :, 1:-1, 2:],
            pe[:, :, :-2, 1:-1],
            pe[:, :, 2:, 1:-1],
        )
        neighbours_c = (
            pc[:, :, 1:-1, :-2],
            pc[:, :, 1:-1, 2:],
            pc[:, :, :-2, 1:-1],
            pc[:, :, 2:, 1:-1],
        )
        neighbours_i = (
            pi[:, :, 1:-1, :-2],
            pi[:, :, 1:-1, 2:],
            pi[:, :, :-2, 1:-1],
            pi[:, :, 2:, 1:-1],
        )
        affinities = tuple(
            0.5 * (ir_only + neighbour_i)
            * torch.exp(-beta * (chroma_visible - neighbour_c).abs())
            for neighbour_i, neighbour_c in zip(
                neighbours_i, neighbours_c
            )
        )
        weighted_sum = sum(
            affinity * neighbour
            for affinity, neighbour in zip(affinities, neighbours_e)
        )
        denominator = 1.0 + strength * sum(affinities)
        estimate = (
            data + strength * weighted_sum
        ) / denominator.clamp_min(1e-6)
    return estimate


class SAHCA(nn.Module):
    """Source-Aware Hunt Chroma Adaptation."""

    def __init__(
        self,
        knots=8,
        alpha_min=0.0,
        alpha_max=0.5,
        projection_steps=3,
    ):
        super().__init__()
        self.source_calibrator = SourceConfidenceCalibrator(knots=knots)
        self.raw_alpha = nn.Parameter(torch.tensor(0.0))
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.projection_steps = int(projection_steps)

    def hunt_alpha(self):
        return self.alpha_min + (
            self.alpha_max - self.alpha_min
        ) * torch.sigmoid(self.raw_alpha)

    def forward(self, Y_in, Y_out, Cb_in, Cr_in, IR):
        eps = 1e-5
        chroma_visible, u_visible, v_visible = chroma_magnitude(
            Cb_in, Cr_in, eps
        )
        features = structural_features(
            Y_in, Y_out, IR, chroma_visible
        )
        source_confidence, contributions = self.source_calibrator(
            features
        )

        log_ratio = (
            torch.log(Y_in.clamp_min(1e-3))
            - torch.log(Y_out.clamp_min(1e-3))
        ).clamp(math.log(0.2), math.log(5.0))
        alpha = self.hunt_alpha().view(1, 1, 1, 1)
        log_chroma_visible = torch.log(chroma_visible.clamp_min(eps))
        hunt_delta = alpha * log_ratio
        log_chroma_hunt = log_chroma_visible + hunt_delta
        log_chroma_target = (
            log_chroma_visible + source_confidence * hunt_delta
        )
        log_chroma_output = edge_aware_projection(
            log_chroma_target,
            chroma_visible,
            features["ir_only"],
            steps=self.projection_steps,
        )
        chroma_output = torch.exp(log_chroma_output)

        # Radial gamut projection preserves the visible chroma direction.
        direction_u = u_visible / chroma_visible.clamp_min(eps)
        direction_v = v_visible / chroma_visible.clamp_min(eps)
        max_u = 0.5 / direction_u.abs().clamp_min(eps)
        max_v = 0.5 / direction_v.abs().clamp_min(eps)
        max_chroma = torch.minimum(max_u, max_v)
        chroma_output = torch.minimum(chroma_output, max_chroma)

        cb_out = 0.5 + chroma_output * direction_u
        cr_out = 0.5 + chroma_output * direction_v

        aux = {
            "Y_in": Y_in,
            "Y_out": Y_out,
            "log_luminance_ratio": log_ratio,
            "hunt_alpha": alpha.expand_as(Y_in),
            "source_confidence": source_confidence,
            "chroma_original": chroma_visible,
            "chroma_hunt": torch.exp(log_chroma_hunt),
            "chroma_before_projection": torch.exp(log_chroma_target),
            "chroma_output": chroma_output,
            "feature_contributions": contributions,
            **features,
        }
        return cb_out.clamp(0.0, 1.0), cr_out.clamp(0.0, 1.0), aux


def spatial_gradient_magnitude(x: torch.Tensor):
    gx, gy = scharr(x)
    return torch.sqrt(gx.square() + gy.square() + 1e-8)


class SAHCALoss(nn.Module):
    """Counterfactual, source-aware objectives without a chroma CNN target."""

    def __init__(
        self,
        lambda_gate=2.0,
        lambda_identity=2.0,
        lambda_leakage=1.0,
        lambda_visible=0.5,
        lambda_cycle=0.25,
        lambda_alpha=0.1,
    ):
        super().__init__()
        self.lambda_gate = float(lambda_gate)
        self.lambda_identity = float(lambda_identity)
        self.lambda_leakage = float(lambda_leakage)
        self.lambda_visible = float(lambda_visible)
        self.lambda_cycle = float(lambda_cycle)
        self.lambda_alpha = float(lambda_alpha)

    def forward(
        self,
        cb_out,
        cr_out,
        cb_in,
        cr_in,
        aux,
        identity_cb=None,
        identity_cr=None,
        cycle_cb=None,
        cycle_cr=None,
    ):
        ir_only = aux["ir_only"].detach()
        gate_target = (
            aux["visible_agreement"].detach()
            * aux["visible_edge_share"].detach()
            * (1.0 - ir_only)
            * (1.0 - aux["visible_noise"].detach())
            * (1.0 - aux["clipping_risk"].detach())
            * (0.5 + 0.5 * aux["chroma_support"].detach())
        ).clamp(0.0, 1.0)
        loss_gate = F.mse_loss(aux["source_confidence"], gate_target)

        chroma_out, _, _ = chroma_magnitude(cb_out, cr_out)
        chroma_in, _, _ = chroma_magnitude(cb_in, cr_in)
        grad_out = spatial_gradient_magnitude(chroma_out)
        grad_in = spatial_gradient_magnitude(chroma_in)
        loss_leakage = (ir_only * grad_out).sum() / (
            ir_only.sum() + 1.0
        )
        visible_weight = 1.0 - ir_only
        loss_visible = (
            visible_weight * (grad_out - grad_in).abs()
        ).sum() / (visible_weight.sum() + 1.0)

        if identity_cb is None or identity_cr is None:
            loss_identity = cb_out.new_zeros(())
        else:
            loss_identity = (
                F.l1_loss(identity_cb, cb_in)
                + F.l1_loss(identity_cr, cr_in)
            )

        if cycle_cb is None or cycle_cr is None:
            loss_cycle = cb_out.new_zeros(())
        else:
            loss_cycle = (
                F.l1_loss(cycle_cb, cb_out)
                + F.l1_loss(cycle_cr, cr_out)
            )

        loss_alpha = (aux["hunt_alpha"].mean() - 0.25).square()
        total = (
            self.lambda_gate * loss_gate
            + self.lambda_identity * loss_identity
            + self.lambda_leakage * loss_leakage
            + self.lambda_visible * loss_visible
            + self.lambda_cycle * loss_cycle
            + self.lambda_alpha * loss_alpha
        )
        values = {
            "total": total.detach().item(),
            "gate": loss_gate.detach().item(),
            "identity": loss_identity.detach().item(),
            "leakage": loss_leakage.detach().item(),
            "visible": loss_visible.detach().item(),
            "cycle": loss_cycle.detach().item(),
            "alpha": aux["hunt_alpha"].mean().detach().item(),
            "source_confidence": (
                aux["source_confidence"].mean().detach().item()
            ),
        }
        return total, values
