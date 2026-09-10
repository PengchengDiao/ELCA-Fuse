"""Small deterministic invariance tests for SAHCA."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chroma_polar_interpretable import SAHCA


def main():
    torch.manual_seed(7)
    model = SAHCA()
    yin = torch.rand(2, 1, 32, 40)
    yout = torch.rand_like(yin)
    infrared = torch.rand_like(yin)
    cb = 0.3 + 0.4 * torch.rand_like(yin)
    cr = 0.3 + 0.4 * torch.rand_like(yin)

    cb_identity, cr_identity, _ = model(
        yin, yin, cb, cr, infrared
    )
    identity_error = (
        (cb_identity - cb).abs().mean()
        + (cr_identity - cr).abs().mean()
    )
    assert identity_error < 1e-6, identity_error.item()

    cb_out, cr_out, _ = model(yin, yout, cb, cr, infrared)
    u_in, v_in = cb - 0.5, cr - 0.5
    u_out, v_out = cb_out - 0.5, cr_out - 0.5
    cross_product = u_in * v_out - v_in * u_out
    assert cross_product.abs().mean() < 1e-6
    assert cb_out.min() >= 0 and cb_out.max() <= 1
    assert cr_out.min() >= 0 and cr_out.max() <= 1

    for name, shape in model.source_calibrator.shapes.items():
        x = torch.linspace(0, 1, 101).view(1, 1, 1, -1)
        differences = shape(x)[..., 1:] - shape(x)[..., :-1]
        direction = model.source_calibrator.SPECIFICATIONS[name]
        assert torch.all(direction * differences >= -1e-7), name

    loss = cb_out.mean() + cr_out.mean()
    loss.backward()
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    print(
        "SAHCA tests passed:",
        f"params={sum(p.numel() for p in model.parameters())}",
        f"identity={identity_error.item():.3e}",
        f"hue_cross={cross_product.abs().mean().item():.3e}",
    )


if __name__ == "__main__":
    main()
