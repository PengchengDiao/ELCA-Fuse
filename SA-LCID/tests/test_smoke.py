import unittest

import numpy as np

from sa_lcid import diagnose, reconstruct_with_fixed_luminance, relative_luminance
from sa_lcid.core import rotate_oklab_hue


def source_pair(size: int = 64):
    yy, xx = np.mgrid[0:size, 0:size]
    visible = np.empty((size, size, 3), dtype=np.float64)
    visible[..., 0] = 0.30 + 0.25 * xx / (size - 1)
    visible[..., 1] = 0.18 + 0.20 * yy / (size - 1)
    visible[..., 2] = 0.10 + 0.08 * xx / (size - 1)
    infrared = np.full((size, size), 0.15, dtype=np.float64)
    infrared[18:46, 28:36] = 0.90
    return visible, infrared


class SaLcidSmokeTest(unittest.TestCase):
    def test_fixed_luminance_reconstruction(self):
        visible, _ = source_pair()
        target = np.full(visible.shape[:2], 0.20, dtype=np.float64)
        reconstructed = reconstruct_with_fixed_luminance(visible, target)
        error = np.max(np.abs(relative_luminance(reconstructed) - target))
        self.assertLess(error, 2e-7)

    def test_identity_has_negligible_components(self):
        visible, infrared = source_pair()
        fixed_y = relative_luminance(visible)
        result = diagnose(visible, infrared, visible, fixed_luminance=fixed_y)
        self.assertLess(result["visible_hue_damage"], 1e-12)
        self.assertLess(result["ir_structure_chroma_leakage"], 1e-12)
        self.assertLess(result["high_frequency_chroma_noise"], 1e-12)
        self.assertLess(result["chroma_attenuation_penalty"], 1e-12)
        self.assertLess(result["chroma_overshoot_penalty"], 1e-12)
        self.assertLess(result["fixed_luminance_mae"], 1e-12)

    def test_hue_rotation_increases_hue_damage(self):
        visible, infrared = source_pair()
        fixed_y = relative_luminance(visible)
        rotated = rotate_oklab_hue(visible, 45.0)
        controlled = reconstruct_with_fixed_luminance(rotated, fixed_y)
        identity = diagnose(visible, infrared, visible, fixed_luminance=fixed_y)
        changed = diagnose(
            visible, infrared, controlled, fixed_luminance=fixed_y
        )
        self.assertGreater(
            changed["visible_hue_damage"], identity["visible_hue_damage"]
        )


if __name__ == "__main__":
    unittest.main()

