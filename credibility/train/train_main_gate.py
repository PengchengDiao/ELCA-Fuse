"""Train MICG-2 on the actual ``y_en`` versus ``y_tcdg`` pair.

The LYT enhancer, MICG-1, TCDG backbone, and colour modules are frozen.  A
cache of the two luminance candidates is prepared once; subsequent epochs
update only the 108-parameter interpretable MICG-2 calibrator.

Default server command::

    python -m credibility.train.train_main_gate --device cuda --epochs 200
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from credibility.models.prob_fusion import scharr  # noqa: E402
from credibility.models.prob_fusion_calibrated import (  # noqa: E402
    ProbabilisticCalibratedFusion,
)
from fusion_model import FusionSerialBridge  # noqa: E402


RESAMPLE_BILINEAR = (
    Image.Resampling.BILINEAR
    if hasattr(Image, "Resampling")
    else Image.BILINEAR
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_checkpoint(path: Path) -> Dict[str, torch.Tensor]:
    try:
        state = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(str(path), map_location="cpu")
    if isinstance(state, dict):
        state = state.get("weight", state.get("state_dict", state))
    output = {}
    for key, value in state.items():
        clean_key = key[7:] if key.startswith("module.") else key
        output[clean_key] = value
    return output


def load_fusion_backbone(model: torch.nn.Module, path: Path) -> None:
    state = load_checkpoint(path)
    current = model.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in current and current[key].shape == value.shape
    }
    core_prefixes = ("encode.", "LIP.", "GIP.", "fc2.")
    core = {key for key in current if key.startswith(core_prefixes)}
    loaded_core = core.intersection(compatible)
    if loaded_core != core:
        missing = sorted(core.difference(loaded_core))
        raise RuntimeError(
            "Fusion checkpoint does not cover the complete TCDG backbone: "
            + ", ".join(missing[:10])
        )
    model.load_state_dict(compatible, strict=False)
    print(
        "Loaded TCDG backbone: "
        f"{len(loaded_core)}/{len(core)} tensors from {path}"
    )


def checkpoint_signature(path: Path) -> Dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def cache_directory(args) -> Path:
    lyt_weights = (
        ROOT.parent / "LYT-Net-main" / "PyTorch" / "best_model.pth"
    )
    signature = {
        "size": [int(args.width), int(args.height)],
        "fusion": checkpoint_signature(args.fusion_weights),
        "micg1": checkpoint_signature(args.micg1_weights),
        "lyt": checkpoint_signature(lyt_weights),
        "version": "micg2_candidates_v1",
    }
    encoded = json.dumps(signature, sort_keys=True).encode("utf-8")
    key = hashlib.sha256(encoded).hexdigest()[:12]
    path = args.cache_root / f"{args.width}x{args.height}_{key}"
    path.mkdir(parents=True, exist_ok=True)
    (path / "cache_signature.json").write_text(
        json.dumps(signature, indent=2), encoding="utf-8"
    )
    return path


class SourcePairSet(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root: Path,
        split: str,
        width: int,
        height: int,
        limit: int = 0,
    ) -> None:
        self.vi_dir = data_root / split / "vi"
        self.ir_dir = data_root / split / "ir"
        self.names = sorted(
            path.name
            for path in self.vi_dir.iterdir()
            if path.is_file() and (self.ir_dir / path.name).is_file()
        )
        if limit > 0:
            self.names = self.names[:limit]
        self.size = (int(width), int(height))

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int):
        name = self.names[index]
        visible = Image.open(self.vi_dir / name).convert("RGB")
        infrared = Image.open(self.ir_dir / name).convert("L")
        if visible.size != infrared.size:
            raise ValueError(f"Size mismatch for {name}")
        visible = visible.resize(self.size, RESAMPLE_BILINEAR)
        infrared = infrared.resize(self.size, RESAMPLE_BILINEAR)
        vi = torch.from_numpy(
            np.asarray(visible, dtype=np.float32).copy() / 255.0
        ).permute(2, 0, 1)
        ir = torch.from_numpy(
            np.asarray(infrared, dtype=np.float32).copy() / 255.0
        ).unsqueeze(0)
        return name, vi, ir


def prepare_cache(
    args,
    split: str,
    cache_root: Path,
    device: torch.device,
    limit: int,
) -> None:
    dataset = SourcePairSet(
        args.data_root, split, args.width, args.height, limit
    )
    destination = cache_root / split
    destination.mkdir(parents=True, exist_ok=True)
    missing = [name for name in dataset.names if not (destination / f"{Path(name).stem}.pt").is_file()]
    if not missing:
        print(f"Cache {split}: {len(dataset)} candidates already available")
        return

    bridge = FusionSerialBridge(
        use_lyt=True,
        lyt_weights=str(args.lyt_weights),
        lyt_root=str(args.lyt_root),
        use_credibility=True,
        credibility_weights=str(args.micg1_weights),
        use_polar_chroma=False,
        lyt_device=device,
    ).to(device)
    load_fusion_backbone(bridge, args.fusion_weights)
    bridge.eval()
    for parameter in bridge.parameters():
        parameter.requires_grad_(False)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.cache_batch_size,
        shuffle=False,
        num_workers=args.cache_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.cache_workers > 0,
    )
    bar = tqdm(loader, desc=f"Cache {split}", unit="batch", dynamic_ncols=True)
    with torch.no_grad():
        for names, visible, infrared in bar:
            needed = [
                index
                for index, name in enumerate(names)
                if not (destination / f"{Path(name).stem}.pt").is_file()
            ]
            if not needed:
                continue
            visible = visible.to(device, non_blocking=True)
            infrared = infrared.to(device, non_blocking=True)
            _, diagnostics = bridge(
                visible, infrared, return_diagnostics=True
            )
            for index in needed:
                name = names[index]
                torch.save(
                    {
                        "y_en": diagnostics["y_en"][index].detach().cpu().half(),
                        "y_tcdg": diagnostics["y_tcdg"][index].detach().cpu().half(),
                        "ir": infrared[index].detach().cpu().half(),
                    },
                    destination / f"{Path(name).stem}.pt",
                )
    del bridge
    if device.type == "cuda":
        torch.cuda.empty_cache()


class CachedCandidateSet(torch.utils.data.Dataset):
    def __init__(self, source: SourcePairSet, cache_root: Path, split: str):
        self.names = source.names
        self.directory = cache_root / split
        missing = [
            name for name in self.names
            if not (self.directory / f"{Path(name).stem}.pt").is_file()
        ]
        if missing:
            raise RuntimeError(
                f"Missing {len(missing)} cached {split} candidates"
            )

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int):
        name = self.names[index]
        path = self.directory / f"{Path(name).stem}.pt"
        try:
            item = torch.load(str(path), map_location="cpu", weights_only=True)
        except TypeError:
            item = torch.load(str(path), map_location="cpu")
        return (
            name,
            item["y_en"].float(),
            item["y_tcdg"].float(),
            item["ir"].float(),
        )


def gradient_magnitude(image: torch.Tensor) -> torch.Tensor:
    gx, gy = scharr(image)
    return torch.sqrt(gx.square() + gy.square() + 1e-8)


def main_gate_target(
    y_en: torch.Tensor,
    y_tcdg: torch.Tensor,
    infrared: torch.Tensor,
    physical_weight: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Build a source-aware target without a fused-image ground truth."""
    with torch.no_grad():
        grad_en = gradient_magnitude(y_en)
        grad_tcdg = gradient_magnitude(y_tcdg)
        grad_ir = gradient_magnitude(infrared)
        target_gradient = torch.maximum(grad_en, grad_ir)
        local_scale = F.avg_pool2d(
            target_gradient, kernel_size=7, stride=1, padding=3
        ) + 0.02

        error_en = (grad_en - target_gradient).abs() / local_scale
        error_tcdg = (grad_tcdg - target_gradient).abs() / local_scale
        candidate_advantage = (error_en - error_tcdg).clamp(-1.0, 1.0)
        candidate_win = torch.sigmoid(5.0 * candidate_advantage)

        ir_need = F.relu(grad_ir - grad_en) / (
            grad_ir + grad_en + 1e-4
        )
        ir_agreement = torch.exp(
            -(grad_tcdg - grad_ir).abs() / (grad_ir + 0.03)
        )
        task_target = (
            ir_need * ir_agreement + (1.0 - ir_need) * candidate_win
        )
        target = (
            0.35 * physical_weight.detach() + 0.65 * task_target
        ).clamp(0.0, 1.0)
        support = (
            ir_need + grad_en / (grad_en + 0.03)
        ).clamp(0.0, 1.0)
    return target, {
        "grad_en": grad_en,
        "grad_ir": grad_ir,
        "target_gradient": target_gradient,
        "ir_need": ir_need,
        "support": support,
    }


def compute_loss(
    gate: ProbabilisticCalibratedFusion,
    y_en: torch.Tensor,
    y_tcdg: torch.Tensor,
    infrared: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    weight, _, _, physical_weight, _ = gate(y_en, y_tcdg, infrared)
    target, auxiliary = main_gate_target(
        y_en, y_tcdg, infrared, physical_weight
    )
    y_fused = weight * y_tcdg + (1.0 - weight) * y_en
    grad_fused = gradient_magnitude(y_fused)

    loss_target = F.smooth_l1_loss(weight, target)
    loss_gradient = F.l1_loss(
        grad_fused, auxiliary["target_gradient"]
    )
    visible_weight = 1.0 - auxiliary["ir_need"]
    loss_visible = (
        visible_weight * (y_fused - y_en).abs()
    ).sum() / (visible_weight.sum() + 1.0)
    loss_overshoot = F.relu(
        grad_fused - auxiliary["target_gradient"]
    ).mean()
    unsupported_change = (
        (1.0 - auxiliary["support"])
        * weight
        * (y_tcdg - y_en).abs()
    ).mean()
    weight_gx, weight_gy = scharr(weight)
    weight_variation = weight_gx.abs() + weight_gy.abs()
    loss_smooth = (
        weight_variation
        * torch.exp(-12.0 * auxiliary["target_gradient"])
    ).mean()

    total = (
        loss_target
        + 0.25 * loss_gradient
        + 0.15 * loss_visible
        + 0.10 * loss_overshoot
        + 0.10 * unsupported_change
        + 0.02 * loss_smooth
    )
    return total, {
        "loss": total,
        "target": loss_target,
        "gradient": loss_gradient,
        "visible": loss_visible,
        "overshoot": loss_overshoot,
        "unsupported": unsupported_change,
        "smooth": loss_smooth,
        "weight_mean": weight.mean(),
        "target_mean": target.mean(),
    }


def run_epoch(
    gate: ProbabilisticCalibratedFusion,
    loader,
    device: torch.device,
    optimizer,
    description: str,
) -> Dict[str, float]:
    training = optimizer is not None
    gate.train(training)
    totals: Dict[str, float] = {}
    samples = 0
    bar = tqdm(loader, desc=description, unit="batch", dynamic_ncols=True)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for _, y_en, y_tcdg, infrared in bar:
            y_en = y_en.to(device, non_blocking=True)
            y_tcdg = y_tcdg.to(device, non_blocking=True)
            infrared = infrared.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss, terms = compute_loss(gate, y_en, y_tcdg, infrared)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    gate.calibrator.parameters(), max_norm=1.0
                )
                optimizer.step()
            batch_size = int(y_en.shape[0])
            samples += batch_size
            for key, value in terms.items():
                totals[key] = totals.get(key, 0.0) + float(
                    value.detach().item()
                ) * batch_size
            bar.set_postfix(
                loss=f"{totals['loss'] / samples:.5f}",
                weight=f"{totals['weight_mean'] / samples:.3f}",
                target=f"{totals['target_mean'] / samples:.3f}",
            )
    return {key: value / max(samples, 1) for key, value in totals.items()}


def save_checkpoint(
    path: Path,
    gate: ProbabilisticCalibratedFusion,
    optimizer,
    epoch: int,
    best_val: float,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    args,
) -> None:
    torch.save(
        {
            "role": "micg2_main_luminance_gate",
            "epoch": int(epoch),
            "state_dict": gate.calibrator.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": float(best_val),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "config": {
                "width": int(args.width),
                "height": int(args.height),
                "learning_rate": float(args.lr),
                "seed": int(args.seed),
                "fusion_weights": str(args.fusion_weights),
                "micg1_weights": str(args.micg1_weights),
            },
        },
        str(path),
    )


def write_history(path: Path, history: Iterable[Dict[str, object]]) -> None:
    rows = list(history)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the standalone MICG-2 main luminance gate."
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "MSRS")
    parser.add_argument("--fusion-weights", type=Path, default=ROOT / "checkpoints" / "fusion.pkl")
    parser.add_argument("--micg1-weights", type=Path, default=ROOT / "checkpoints" / "micg1.pkl")
    parser.add_argument("--lyt-weights", type=Path, default=ROOT / "checkpoints" / "lyt.pth")
    parser.add_argument("--lyt-root", type=Path, default=ROOT / "third_party" / "LYT")
    parser.add_argument("--cache-root", type=Path, default=ROOT / "credibility" / "cache" / "micg2")
    parser.add_argument("--ckpt-dir", type=Path, default=ROOT / "credibility" / "checkpoints" / "micg2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cache-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cache-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(str(ROOT))
    seed_everything(args.seed)
    if not args.fusion_weights.is_file():
        raise FileNotFoundError(args.fusion_weights)
    if not args.micg1_weights.is_file():
        raise FileNotFoundError(args.micg1_weights)
    if not args.lyt_weights.is_file():
        raise FileNotFoundError(args.lyt_weights)
    device = torch.device(
        args.device if args.device.startswith("cpu") or torch.cuda.is_available()
        else "cpu"
    )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    cache_root = cache_directory(args)
    prepare_cache(
        args, "train", cache_root, device, args.train_limit
    )
    prepare_cache(
        args, "test", cache_root, device, args.val_limit
    )
    print("Candidate cache:", cache_root)
    if args.prepare_only:
        return

    train_source = SourcePairSet(
        args.data_root,
        "train",
        args.width,
        args.height,
        args.train_limit,
    )
    val_source = SourcePairSet(
        args.data_root,
        "test",
        args.width,
        args.height,
        args.val_limit,
    )
    train_set = CachedCandidateSet(train_source, cache_root, "train")
    val_set = CachedCandidateSet(val_source, cache_root, "test")
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_set, shuffle=True, generator=generator, **loader_options
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, shuffle=False, **loader_options
    )

    gate = ProbabilisticCalibratedFusion().to(device)
    initial = load_checkpoint(args.micg1_weights)
    compatible = {
        key: value
        for key, value in initial.items()
        if key in gate.calibrator.state_dict()
        and gate.calibrator.state_dict()[key].shape == value.shape
    }
    gate.calibrator.load_state_dict(compatible, strict=True)
    for parameter in gate.phys.parameters():
        parameter.requires_grad_(False)
    trainable = list(gate.calibrator.parameters())
    print(
        f"Train:{len(train_set)} Val:{len(val_set)} "
        f"resolution:{args.width}x{args.height} device:{device}"
    )
    print(
        "MICG-2 trainable parameters:",
        sum(parameter.numel() for parameter in trainable),
    )

    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    history: List[Dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            gate,
            train_loader,
            device,
            optimizer,
            f"MICG-2 Train {epoch:03d}/{args.epochs:03d}",
        )
        val_metrics = run_epoch(
            gate,
            val_loader,
            device,
            None,
            f"MICG-2 Val   {epoch:03d}/{args.epochs:03d}",
        )
        improved = val_metrics["loss"] < best_val
        if improved:
            best_val = val_metrics["loss"]
        record: Dict[str, object] = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_weight_mean": train_metrics["weight_mean"],
            "val_weight_mean": val_metrics["weight_mean"],
            "train_target_mean": train_metrics["target_mean"],
            "val_target_mean": val_metrics["target_mean"],
            "best": int(improved),
        }
        history.append(record)
        save_checkpoint(
            args.ckpt_dir / "micg2_last.pkl",
            gate,
            optimizer,
            epoch,
            best_val,
            train_metrics,
            val_metrics,
            args,
        )
        if improved:
            save_checkpoint(
                args.ckpt_dir / "micg2_best.pkl",
                gate,
                optimizer,
                epoch,
                best_val,
                train_metrics,
                val_metrics,
                args,
            )
        write_history(args.ckpt_dir / "history.csv", history)
        print(
            f"Epoch {epoch:03d}: train={train_metrics['loss']:.6f} "
            f"val={val_metrics['loss']:.6f} best={best_val:.6f} "
            f"{'[saved best]' if improved else ''}"
        )

    print("Best MICG-2:", (args.ckpt_dir / "micg2_best.pkl").resolve())
    print("Last MICG-2:", (args.ckpt_dir / "micg2_last.pkl").resolve())


if __name__ == "__main__":
    main()
