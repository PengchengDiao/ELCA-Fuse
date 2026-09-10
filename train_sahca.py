"""Stable launcher for isolated SAHCA training.

Run this file from any working directory. It resolves project-relative paths,
checks the frozen fusion checkpoint before cache preparation, patches the
configuration in ``chroma_polar_interpretable.train``, and starts training.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# All paths in the original training configuration are project-relative.
os.chdir(PROJECT_ROOT)


# Keep a reproducible checkpoint for every epoch and render the same fixed
# day/night samples after training. Raw maps are also stored as compressed NPZ
# files so later statistics do not depend on 8-bit PNG quantisation.
SAVE_EVERY_EPOCH = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ELCA-Fuse SAHCA.")
    parser.add_argument("--data-root", default="data/MSRS")
    parser.add_argument("--fusion-weights", default="checkpoints/fusion.pkl")
    parser.add_argument("--lyt-weights", default="checkpoints/lyt.pth")
    parser.add_argument("--lyt-root", default="third_party/LYT")
    parser.add_argument(
        "--output-dir",
        default="chroma_polar_interpretable/checkpoints/sahca_v1",
    )
    parser.add_argument(
        "--cache-root",
        default="chroma_polar_interpretable/cache/sahca_640x480",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def require_file(path: Path, description: str) -> Path:
    if path.is_file():
        return path

    raise FileNotFoundError(
        f"{description} is required but was not found:\n  - {path}\n"
        "Train or obtain it separately, then pass its path on the command line."
    )


def install_epoch_checkpoint_archiver():
    original_save = torch.save

    def save_and_archive(obj, destination, *args, **kwargs):
        original_save(obj, destination, *args, **kwargs)
        if not SAVE_EVERY_EPOCH or not isinstance(
            destination, (str, os.PathLike)
        ):
            return

        destination_path = Path(destination)
        if destination_path.name != "last.pt" or not isinstance(obj, dict):
            return

        epoch = obj.get("epoch")
        if not isinstance(epoch, int):
            return

        epoch_dir = destination_path.parent / "epochs"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        epoch_path = epoch_dir / f"epoch_{epoch:03d}.pt"
        original_save(obj, epoch_path)
        print(f"Archived epoch checkpoint: {epoch_path}")

    torch.save = save_and_archive
    return original_save


def main() -> None:
    args = parse_args()
    fusion_weights = require_file(
        project_path(args.fusion_weights), "Frozen TCDG fusion checkpoint"
    )
    lyt_weights = require_file(
        project_path(args.lyt_weights), "Pretrained LYT checkpoint"
    )

    from chroma_polar_interpretable import train as train_module

    config = getattr(train_module, "CONFIG", None)
    if config is None:
        raise AttributeError(
            "chroma_polar_interpretable.train does not expose CONFIG. "
            "Restore the matching train.py source before training."
        )

    config.data_root = str(project_path(args.data_root))
    config.fusion_weights = str(fusion_weights)
    config.lyt_weights = str(lyt_weights)
    config.lyt_root = str(project_path(args.lyt_root))
    config.output_dir = str(project_path(args.output_dir))
    config.cache_root = str(project_path(args.cache_root))
    config.device = args.device
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.num_workers = args.num_workers
    config.width = args.width
    config.height = args.height
    config.seed = args.seed
    config.rebuild_cache = args.rebuild_cache
    config.use_cache = not args.no_cache
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Fusion weights: {fusion_weights}")
    print(f"LYT weights: {lyt_weights}")
    print(f"Data root: {config.data_root}")
    print(f"Output directory: {config.output_dir}")

    # SAHCA has a deterministic initialization. Saving epoch 0 makes the
    # learned chroma effect directly comparable with the untrained mechanism.
    from chroma_polar_interpretable import SAHCA

    initial_epoch_dir = Path(config.output_dir) / "epochs"
    initial_epoch_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": SAHCA().state_dict(),
            "epoch": 0,
            "metrics": {},
            "baseline": "deterministic_untrained_initialization",
        },
        initial_epoch_dir / "epoch_000.pt",
    )

    original_save = install_epoch_checkpoint_archiver()
    try:
        train_module.train()
    finally:
        torch.save = original_save

if __name__ == "__main__":
    main()
