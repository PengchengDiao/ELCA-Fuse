"""Train SAHCA in isolation; existing chroma_polar code is never modified.

Edit the CONFIG block below directly.  No terminal configuration is required.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from chroma_polar_interpretable import SAHCA, SAHCALoss
from fusion_model import FusionSerialBridge, rgb_to_ycbcr


@dataclass
class TrainConfig:
    data_root: str = "data/MSRS"
    fusion_weights: str = "checkpoints/fusion.pkl"
    lyt_weights: str = "checkpoints/lyt.pth"
    lyt_root: str = "third_party/LYT"
    output_dir: str = (
        "chroma_polar_interpretable/checkpoints/sahca_v1"
    )
    cache_root: str = "chroma_polar_interpretable/cache/sahca_640x480"
    use_cache: bool = True
    rebuild_cache: bool = False
    width: int = 640
    height: int = 480
    batch_size: int = 4
    num_workers: int = 4
    epochs: int = 200
    learning_rate: float = 2e-3
    weight_decay: float = 1e-5
    device: str = "cuda"
    seed: int = 42
    train_limit: Optional[int] = None
    val_limit: Optional[int] = None


CONFIG = TrainConfig()
RESAMPLE_BILINEAR = (
    Image.Resampling.BILINEAR
    if hasattr(Image, "Resampling")
    else Image.BILINEAR
)


class PairedMSRS(Dataset):
    def __init__(
        self,
        root,
        split,
        width,
        height,
        limit=None,
        augment=False,
    ):
        self.vi_dir = Path(root) / split / "vi"
        self.ir_dir = Path(root) / split / "ir"
        self.names = sorted(
            path.name for path in self.vi_dir.glob("*.png")
            if (self.ir_dir / path.name).is_file()
        )
        if limit is not None:
            self.names = self.names[: int(limit)]
        self.size = (int(width), int(height))
        self.augment = bool(augment)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        vi = Image.open(self.vi_dir / name).convert("RGB")
        ir = Image.open(self.ir_dir / name).convert("L")
        if vi.size != ir.size:
            raise ValueError(
                f"Size mismatch for {name}: vi={vi.size}, ir={ir.size}"
            )
        vi = vi.resize(self.size, RESAMPLE_BILINEAR)
        ir = ir.resize(self.size, RESAMPLE_BILINEAR)
        vi_array = np.asarray(vi, dtype=np.float32) / 255.0
        ir_array = np.asarray(ir, dtype=np.float32) / 255.0
        vi_tensor = torch.from_numpy(vi_array).permute(2, 0, 1)
        ir_tensor = torch.from_numpy(ir_array).unsqueeze(0)
        if self.augment and torch.rand(()) < 0.5:
            vi_tensor = vi_tensor.flip(-1)
            ir_tensor = ir_tensor.flip(-1)
        return vi_tensor, ir_tensor, name


class CachedSAHCASet(Dataset):
    def __init__(self, cache_root, split, augment=False, limit=None):
        self.cache_dir = Path(cache_root) / split
        self.files = sorted(self.cache_dir.glob("*.pt"))
        if limit is not None:
            self.files = self.files[: int(limit)]
        self.augment = bool(augment)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        try:
            item = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            item = torch.load(path, map_location="cpu")
        tensors = [
            item[key].float()
            for key in ("yin", "yfused", "cb", "cr", "infrared")
        ]
        if self.augment and torch.rand(()) < 0.5:
            tensors = [tensor.flip(-1) for tensor in tensors]
        return (*tensors, item["name"])


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_state(path):
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        state = state.get("weight", state.get("state_dict", state))
    return {
        key.removeprefix("module."): value
        for key, value in state.items()
    }


def build_frozen_backbone(config, device):
    backbone = FusionSerialBridge(
        use_lyt=True,
        lyt_weights=config.lyt_weights,
        lyt_root=config.lyt_root,
        use_credibility=False,
        use_polar_chroma=False,
        lyt_device=device,
    )
    state = load_state(config.fusion_weights)
    current = backbone.state_dict()
    compatible = {
        key: value for key, value in state.items()
        if key in current and current[key].shape == value.shape
    }
    backbone.load_state_dict(compatible, strict=False)
    backbone.to(device).eval()
    backbone.requires_grad_(False)
    print(
        f"Frozen fusion backbone: loaded {len(compatible)}/{len(state)} tensors"
    )
    return backbone


@torch.no_grad()
def prepare_cache_split(config, split, device):
    cache_dir = Path(config.cache_root) / split
    marker = cache_dir / "complete.json"
    fusion_path = Path(config.fusion_weights)
    fusion_stat = fusion_path.stat()
    fusion_signature = {
        "path": fusion_path.as_posix(),
        "size": fusion_stat.st_size,
        "mtime_ns": fusion_stat.st_mtime_ns,
    }
    split_limit = (
        config.train_limit if split == "train" else config.val_limit
    )
    raw_set = PairedMSRS(
        config.data_root,
        split,
        config.width,
        config.height,
        limit=split_limit,
        augment=False,
    )
    if marker.is_file() and not config.rebuild_cache:
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        if (
            metadata.get("images") == len(raw_set)
            and metadata.get("width") == config.width
            and metadata.get("height") == config.height
            and metadata.get("fusion_signature") == fusion_signature
        ):
            print(f"Reuse {split} cache: {cache_dir}")
            return

    cache_dir.mkdir(parents=True, exist_ok=True)
    backbone = build_frozen_backbone(config, device)
    loader = DataLoader(
        raw_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
    )
    for visible, infrared, names in tqdm(
        loader,
        desc=f"Cache {split}",
        unit="batch",
        dynamic_ncols=True,
    ):
        visible = visible.to(device, non_blocking=True)
        infrared = infrared.to(device, non_blocking=True)
        yin, yfused = final_luminance(
            backbone, visible, infrared
        )
        _, cb, cr = rgb_to_ycbcr(visible)
        for index, name in enumerate(names):
            torch.save({
                "name": name,
                "yin": yin[index].half().cpu(),
                "yfused": yfused[index].half().cpu(),
                "cb": cb[index].half().cpu(),
                "cr": cr[index].half().cpu(),
                "infrared": infrared[index].half().cpu(),
            }, cache_dir / f"{Path(name).stem}.pt")
    marker.write_text(json.dumps({
        "images": len(raw_set),
        "width": config.width,
        "height": config.height,
        "fusion_signature": fusion_signature,
    }, indent=2), encoding="utf-8")
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()


@torch.no_grad()
def final_luminance(backbone, visible, infrared):
    _, diagnostics = backbone(
        visible, infrared, return_diagnostics=True
    )
    return (
        diagnostics["y_in"],
        diagnostics["y_fused"],
    )


def counterfactual_luminance(y_visible, infrared):
    batch = y_visible.shape[0]
    gamma = torch.empty(
        batch, 1, 1, 1, device=y_visible.device
    ).uniform_(0.72, 1.35)
    exposure = y_visible.clamp_min(1e-4).pow(gamma)
    infrared_base = F.avg_pool2d(infrared, 9, 1, 4)
    infrared_detail = infrared - infrared_base
    injection = torch.empty(
        batch, 1, 1, 1, device=y_visible.device
    ).uniform_(0.10, 0.35)
    return (exposure + injection * infrared_detail).clamp(0.0, 1.0)


def run_batch(
    model,
    criterion,
    yin,
    yfused,
    cb,
    cr,
    infrared,
):
    with torch.no_grad():
        ycounter = counterfactual_luminance(yin, infrared)
    # Real fusion and controlled IR-only intervention share one batched pass.
    yin_both = torch.cat([yin, yin], dim=0)
    yout_both = torch.cat([yfused, ycounter], dim=0)
    cb_both = torch.cat([cb, cb], dim=0)
    cr_both = torch.cat([cr, cr], dim=0)
    infrared_both = torch.cat([infrared, infrared], dim=0)
    cb_out, cr_out, aux = model(
        yin_both,
        yout_both,
        cb_both,
        cr_both,
        infrared_both,
    )
    return criterion(
        cb_out,
        cr_out,
        cb_both,
        cr_both,
        aux,
    )


def run_epoch(
    model,
    criterion,
    backbone,
    loader,
    device,
    epoch,
    epochs,
    optimizer=None,
):
    training = optimizer is not None
    model.train(training)
    totals = {}
    samples = 0
    prefix = "Train" if training else "Val  "
    bar = tqdm(
        loader,
        desc=f"{prefix} {epoch:03d}/{epochs:03d}",
        unit="batch",
        dynamic_ncols=True,
    )
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for yin, yfused, cb, cr, infrared, _ in bar:
            yin = yin.to(device, non_blocking=True)
            yfused = yfused.to(device, non_blocking=True)
            cb = cb.to(device, non_blocking=True)
            cr = cr.to(device, non_blocking=True)
            infrared = infrared.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss, metrics = run_batch(
                model,
                criterion,
                yin,
                yfused,
                cb,
                cr,
                infrared,
            )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )
                optimizer.step()

            batch_size = yin.shape[0]
            samples += batch_size
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value * batch_size
            bar.set_postfix(
                loss=f"{totals['total'] / samples:.4f}",
                gate=f"{totals['gate'] / samples:.4f}",
                leak=f"{totals['leakage'] / samples:.4f}",
                alpha=f"{metrics['alpha']:.3f}",
            )
    return {
        key: value / max(samples, 1)
        for key, value in totals.items()
    }


def save_checkpoint(path, model, optimizer, epoch, metrics, config):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "metrics": metrics,
        "config": asdict(config),
        "architecture": "SAHCA-v1",
    }, path)


def train(config=CONFIG):
    seed_everything(config.seed)
    device = torch.device(
        config.device
        if config.device == "cpu" or torch.cuda.is_available()
        else "cpu"
    )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2),
        encoding="utf-8",
    )

    if not config.use_cache:
        raise ValueError(
            "SAHCA v1 training requires the fixed-feature cache. "
            "Set use_cache=True."
        )
    prepare_cache_split(config, "train", device)
    prepare_cache_split(config, "test", device)
    train_set = CachedSAHCASet(
        config.cache_root,
        "train",
        augment=True,
        limit=config.train_limit,
    )
    val_set = CachedSAHCASet(
        config.cache_root,
        "test",
        augment=False,
        limit=config.val_limit,
    )
    loader_args = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": config.num_workers > 0,
    }
    train_loader = DataLoader(
        train_set, shuffle=True, **loader_args
    )
    val_loader = DataLoader(
        val_set, shuffle=False, **loader_args
    )

    model = SAHCA().to(device)
    criterion = SAHCALoss().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.epochs, 1)
    )
    parameter_count = sum(p.numel() for p in model.parameters())
    print(
        f"Train:{len(train_set)} Val:{len(val_set)} "
        f"resolution:{config.width}x{config.height} "
        f"params:{parameter_count} device:{device}"
    )

    best_loss = float("inf")
    history = []
    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(
            model,
            criterion,
            None,
            train_loader,
            device,
            epoch,
            config.epochs,
            optimizer=optimizer,
        )
        val_metrics = run_epoch(
            model,
            criterion,
            None,
            val_loader,
            device,
            epoch,
            config.epochs,
        )
        scheduler.step()
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(record)
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            epoch,
            val_metrics,
            config,
        )
        if val_metrics["total"] < best_loss:
            best_loss = val_metrics["total"]
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                val_metrics,
                config,
            )
            print(
                f"[BEST] epoch={epoch} val={best_loss:.6f}"
            )

    print(
        f"Training complete. best_val={best_loss:.6f} "
        f"output={output_dir}"
    )


if __name__ == "__main__":
    train()
