# Checkpoint paths

No model weights are distributed in this repository. End-to-end inference uses
the following conventional local paths:

| Path | Role | Produced or obtained from |
| --- | --- | --- |
| `checkpoints/fusion.pkl` | frozen TCDG fusion backbone | `train.py` |
| `checkpoints/micg1.pkl` | MICG-1 reliability mapping | `credibility.train.train_calibrator_v2` |
| `checkpoints/micg2.pkl` | MICG-2 reliability mapping | `credibility.train.train_main_gate` |
| `checkpoints/sahca.pkl` | SAHCA mapping | `train_sahca.py` |
| `checkpoints/lyt.pth` | pretrained LYT enhancer | upstream LYT-Net project |

The filenames are conventions rather than download links. You may store the
files elsewhere and pass explicit paths to `infer_fusion.py` and the training
commands. Do not commit checkpoints or other serialized model artifacts.
