# Data layout

Datasets are not included in this repository. Place paired visible and infrared
images under matching `vi` and `ir` directories. Filenames must match within
each pair.

```text
data/
├── MSRS/
│   ├── train/
│   │   ├── vi/                     # RGB visible images
│   │   └── ir/                     # grayscale infrared images
│   └── test/
│       ├── vi/
│       └── ir/
└── LLVIP/
    └── test/
        ├── vi/
        └── ir/
```

The released training commands use MSRS. LLVIP is used only for cross-dataset
evaluation without retraining. PNG, JPEG, BMP, and TIFF inputs are accepted by
the inference script.

Do not commit dataset images. The repository `.gitignore` keeps this README
while excluding all other contents of `data/`.
