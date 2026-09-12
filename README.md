# ELCA-Fuse

Official implementation of **ELCA-Fuse: Evidence-Guided Luminance Selection
and Visible-Supported Chroma Adaptation for Low-Light Infrared--Visible Image
Fusion**.

ELCA-Fuse separates low-light fusion into two decisions: reliability-aware
luminance selection and visible-supported chroma adaptation. The repository
contains the TCDG fusion backbone, two monotonic reliability gates
(MICG-1/MICG-2), source-aware Hunt chroma adaptation (SAHCA), and deterministic
ICh refinement.

## Repository layout

```text
.
├── infer_fusion.py                 # end-to-end inference
├── fusion_model.py                 # complete luminance and chroma pipeline
├── train.py                        # TCDG fusion-backbone training
├── train_sahca.py                  # configurable SAHCA training launcher
├── credibility/
│   ├── models/                     # MICG reliability models
│   └── train/                      # MICG-1 and MICG-2 training
├── chroma_polar_interpretable/     # SAHCA and ICh refinement
├── third_party/LYT/                # LYT model definition and license
├── data/README.md                  # expected paired-image layout
├── checkpoints/README.md           # expected checkpoint names
├── scripts/check_release.py        # source-only release audit
└── requirements.txt
```

Some internal Python identifiers retain the earlier `credibility` and
`calibrated` names for checkpoint/API compatibility. In the paper and public
documentation, these components are described as reliability mappings.

## Installation

Python 3.9 or newer is recommended. Install the PyTorch build matching the
local CUDA runtime first, then install the remaining dependencies.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Data and checkpoints

Prepare paired visible/infrared images as described in
[`data/README.md`](data/README.md). The pipeline expects five checkpoint paths,
documented in [`checkpoints/README.md`](checkpoints/README.md); none is bundled
with this source-only release.

## Inference

From the repository root, run:

```bash
python infer_fusion.py \
  --visible data/MSRS/test/vi \
  --infrared data/MSRS/test/ir \
  --output outputs/MSRS \
  --fusion_weights checkpoints/fusion.pkl \
  --micg1_weights checkpoints/micg1.pkl \
  --micg2_weights checkpoints/micg2.pkl \
  --sahca_weights checkpoints/sahca.pkl \
  --lyt_weights checkpoints/lyt.pth \
  --device cuda
```

Use `--save_diagnostics` to export MICG and chroma-support maps. Run
`python infer_fusion.py --help` for all path, device, and image-size options.

## Training

The paper configuration uses full-resolution MSRS images (640x480), 200 epochs
for each added gating module, and the stage-specific batch sizes and learning
rates below.

```bash
# 1. Train the TCDG fusion backbone (omit --resize for native resolution).
python train.py \
  --data_root data/MSRS \
  --train_path checkpoints/tcdg_training \
  --epochs 200 \
  --shuffle

# 2. Train MICG-1: raw versus LYT-enhanced luminance.
python -m credibility.train.train_calibrator_v2 \
  --data_root data/MSRS \
  --lyt_weights checkpoints/lyt.pth \
  --ckpt_dir checkpoints/micg1_training \
  --epochs 200 --batch_size 1 --lr 1e-3 \
  --width 640 --height 480 --device cuda

# 3. Train MICG-2: selected visible luminance versus TCDG output.
python -m credibility.train.train_main_gate \
  --data-root data/MSRS \
  --fusion-weights checkpoints/fusion.pkl \
  --micg1-weights checkpoints/micg1.pkl \
  --lyt-weights checkpoints/lyt.pth \
  --epochs 200 --batch-size 8 --lr 5e-4 \
  --width 640 --height 480 --device cuda

# 4. Train SAHCA.
python train_sahca.py \
  --data-root data/MSRS \
  --fusion-weights checkpoints/fusion.pkl \
  --lyt-weights checkpoints/lyt.pth \
  --epochs 200 --batch-size 4 \
  --width 640 --height 480 --device cuda
```

Copy the selected outputs to the conventional filenames in
`checkpoints/README.md` before end-to-end inference. Generated checkpoints,
caches, logs, and images are ignored by Git.

## Checks

The following commands do not require trained weights:

```bash
python chroma_polar_interpretable/test_sahca.py
python scripts/check_release.py
```

The release audit checks Python syntax, required source files, forbidden private
artifacts, caches, datasets, and weight-like files.

## License and third-party code

The project source is distributed under the Apache-2.0 license. The vendored
LYT model definition is covered separately by the MIT license in
`third_party/LYT/LICENSE`; its pretrained weight must be obtained separately
from the [upstream LYT-Net project](https://github.com/albrateanu/LYT-Net).

Please add the final paper citation after publication. Remaining release checks
are tracked in [`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md).
