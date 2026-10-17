# SA-LCID evaluation

SA-LCID is a **Source-Aware Luminance--Chroma Intervention Diagnostic** for
registered infrared-visible image fusion. The protocol first reconstructs every method with the same declared luminance.It then measures five chroma failure modes:

1. visible-source hue damage;
2. chroma leakage on infrared-only structure;
3. excess high-frequency chroma noise;
4. chroma attenuation;
5. chroma overshoot.

All five components are **lower-is-better**. The reported
`provisional_artifact_index` is their unweighted RMS diagnostic. Component
values should be reported alongside that aggregate.

## Repository layout

```text
evaluation/sa_lcid/
├── pyproject.toml
├── requirements.txt
├── README.md
├── examples/
│   └── single_image.py
├── sa_lcid/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   └── core.py
└── tests/
    └── test_smoke.py
```

No dataset, fusion result, checkpoint, or generated evaluation table is stored
in this folder.

## Installation

From this directory:

```bash
python -m pip install -e .
```

The evaluator requires Python 3.9+, NumPy, Pillow, SciPy, and tqdm. It does not
require PyTorch or a GPU.

## Expected data layout

Visible, infrared, and method-output images must be registered and share the
same filename.

```text
dataset/
├── visible/
│   ├── 00001D.png
│   └── 00002N.png
└── infrared/
    ├── 00001D.png
    └── 00002N.png

fusion_results/
├── MethodA/
│   ├── 00001D.png
│   └── 00002N.png
├── MethodB/
│   ├── 00001D.png
│   └── 00002N.png
└── Ours/
    ├── 00001D.png
    └── 00002N.png
```

The method selected by `--fixed-luminance-method` supplies the common
linear-light luminance used to reconstruct every method before comparison.

## Batch evaluation

```bash
sa-lcid \
  --visible-dir dataset/visible \
  --infrared-dir dataset/infrared \
  --method-root fusion_results \
  --fixed-luminance-method Ours \
  --include-methods MethodA MethodB Ours \
  --output-dir results/sa_lcid
```

The equivalent command without installation is:

```bash
python -m sa_lcid \
  --visible-dir dataset/visible \
  --infrared-dir dataset/infrared \
  --method-root fusion_results \
  --fixed-luminance-method Ours \
  --output-dir results/sa_lcid
```

Useful options:

- `--analysis-max-side 384`: resize the long side for evaluation;
- `--limit 10`: smoke-test only the first ten common images;
- `--no-synthetic`: skip synthetic monotonicity checks;
- `--synthetic-samples 8`: number of source pairs used for those checks;
- `--day-suffix D --night-suffix N`: filename-based MSRS stratification.

Run `sa-lcid --help` for the complete interface.

## Outputs

| File | Contents |
|---|---|
| `per_image.csv` | Every image-method diagnostic |
| `summary.csv` | Full-set mean for each method |
| `summary_day_night.csv` | Day/night/other stratified means |
| `synthetic_monotonicity.json` | Controlled corruption checks |
| `run_config.json` | Paths, method list, seed, and metric constants |

The main CSV columns are:

- `visible_hue_damage`
- `ir_structure_chroma_leakage`
- `high_frequency_chroma_noise`
- `chroma_attenuation_penalty`
- `chroma_overshoot_penalty`
- `provisional_artifact_index`
- `fixed_luminance_mae`

`fixed_luminance_mae` is a protocol check and should be close to zero.

## Single-image API

```python
import numpy as np
from PIL import Image

from sa_lcid import diagnose, reconstruct_with_fixed_luminance, relative_luminance

visible = np.asarray(Image.open("visible.png").convert("RGB")) / 255.0
infrared = np.asarray(Image.open("infrared.png").convert("L")) / 255.0
fused = np.asarray(Image.open("fused.png").convert("RGB")) / 255.0

fixed_y = relative_luminance(fused)
controlled = reconstruct_with_fixed_luminance(fused, fixed_y)
result = diagnose(visible, infrared, controlled, fixed_luminance=fixed_y)

for name, value in result.items():
    if np.isscalar(value):
        print(f"{name}: {value:.8f}")
```

Pass `return_maps=True` to `diagnose` when diagnostic maps are needed for
visualization.

## Reproducibility checks

Run the included tests with the standard library runner:

```bash
python -m unittest discover -s tests -v
```

Synthetic monotonicity checks are enabled in the batch CLI by default. They
verify that increasing a controlled hue rotation, IR-only false colour,
chroma noise, attenuation, or overshoot increases the intended component.

## Interpretation

SA-LCID is a source-conditioned failure diagnostic under fixed luminance. It
does not require a color ground-truth image. Consequently, it should not be
presented as a universal perceptual color-quality score; use its separate
components to state which failure mode changed.

This module follows the Apache-2.0 license at the repository root.

