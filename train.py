"""Train the ELCA-Fuse luminance fusion backbone."""
import argparse
import copy
import csv
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # ✅ FIX: needed for smooth_l1_loss
import torch.optim as optim
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from fusion_model import FusionSerialBridge as Net1
from loss import Fusionloss

# =========================
# Utils: RGB -> Y / CbCr
# =========================
def rgb_to_y(rgb: torch.Tensor):
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def rgb_to_cbcr(rgb: torch.Tensor):
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return cb, cr


def ycbcr_to_rgb(y, cb, cr):
    cb = cb - 0.5
    cr = cr - 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return torch.cat([r, g, b], dim=1).clamp(0, 1)


def exposure_loss(y: torch.Tensor, target: float = 0.6, patch: int = 16):
    """
    Simple exposure control loss:
      - average luminance per patch should approach target.
    y: (B,1,H,W) in [0,1]
    """
    pool = torch.nn.AvgPool2d(kernel_size=patch, stride=patch)
    y_p = pool(y)
    return (y_p - target).abs().mean()


def tv_loss(x):
    # Robust TV loss: avoid division by zero if H/W too small
    if x.size(2) < 2 and x.size(3) < 2:
        return x.new_tensor(0.0)
    batch_size = x.size(0)
    h_tv = (x[:, :, 1:, :] - x[:, :, :-1, :]).pow(2).sum()
    w_tv = (x[:, :, :, 1:] - x[:, :, :, :-1]).pow(2).sum()
    count_h = x[:, :, 1:, :].numel()
    count_w = x[:, :, :, 1:].numel()
    out = 0.0
    if count_h > 0:
        out += h_tv / count_h
    if count_w > 0:
        out += w_tv / count_w
    return 2.0 * out / batch_size


def edge_aware_tv(w, guide, alpha=10.0):
    """
    Edge-aware TV for w guided by guide (e.g., y_in).
    w, guide: (B,1,H,W)
    """
    gx = (guide[:, :, :, 1:] - guide[:, :, :, :-1]).abs()
    gy = (guide[:, :, 1:, :] - guide[:, :, :-1, :]).abs()
    wx = torch.exp(-alpha * gx)
    wy = torch.exp(-alpha * gy)

    dw = (w[:, :, :, 1:] - w[:, :, :, :-1]).abs()
    dh = (w[:, :, 1:, :] - w[:, :, :-1, :]).abs()
    return (wx * dw).mean() + (wy * dh).mean()


def module_grad_norm(module: nn.Module) -> float:
    total = None
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        grad_sq = parameter.grad.detach().float().pow(2).sum()
        total = grad_sq if total is None else total + grad_sq
    return 0.0 if total is None else total.sqrt().item()


def region_stats_w_deltas(w, dcb, dcr, y_in, bright_thr=0.35, dark_thr=0.30):
    """
    统计：亮区/暗区的 w 均值、|dcb| 均值、|dcr| 均值（按像素统计）
    bright: y_in >= bright_thr
    dark:   y_in <  dark_thr
    """
    bright = (y_in >= bright_thr).float()
    dark = (y_in < dark_thr).float()

    def masked_mean(x, m):
        denom = m.sum()
        if denom.item() < 1.0:
            return x.new_tensor(0.0), x.new_tensor(0.0)
        return (x * m).sum() / (denom + 1e-6), denom

    w_b, cnt_b = masked_mean(w, bright)
    w_d, cnt_d = masked_mean(w, dark)

    adcb = dcb.abs()
    adcr = dcr.abs()
    dcb_b, _ = masked_mean(adcb, bright)
    dcb_d, _ = masked_mean(adcb, dark)
    dcr_b, _ = masked_mean(adcr, bright)
    dcr_d, _ = masked_mean(adcr, dark)

    total = y_in.numel()
    bright_ratio = cnt_b / max(total, 1)
    dark_ratio = cnt_d / max(total, 1)

    return {
        "w_bright": w_b,
        "w_dark": w_d,
        "dcb_bright": dcb_b,
        "dcb_dark": dcb_d,
        "dcr_bright": dcr_b,
        "dcr_dark": dcr_d,
        "bright_ratio": bright_ratio,
        "dark_ratio": dark_ratio,
        "cnt_bright": cnt_b,
        "cnt_dark": cnt_d,
    }


# =========================
# Dataset: Paired VI/IR by filename
# =========================
class PairedMSRS(torch.utils.data.Dataset):
    """
    将 VI / IR 按文件名对齐后打包返回 (vi, ir)
    这样 DataLoader 可以 shuffle=True 而不会错位
    """

    def __init__(
        self,
        root_vi: str,
        root_ir: str,
        transform_vi=None,
        transform_ir=None,
        exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
    ):
        self.root_vi = root_vi
        self.root_ir = root_ir
        self.transform_vi = transform_vi
        self.transform_ir = transform_ir
        exts = tuple(e.lower() for e in exts)

        if not os.path.isdir(root_vi):
            raise FileNotFoundError(f"VI directory not found: {root_vi}")
        if not os.path.isdir(root_ir):
            raise FileNotFoundError(f"IR directory not found: {root_ir}")

        vi_map = {}
        for n in os.listdir(root_vi):
            p = os.path.join(root_vi, n)
            if os.path.isfile(p) and os.path.splitext(n)[1].lower() in exts:
                key = os.path.splitext(os.path.basename(n))[0]
                vi_map[key] = p

        ir_map = {}
        for n in os.listdir(root_ir):
            p = os.path.join(root_ir, n)
            if os.path.isfile(p) and os.path.splitext(n)[1].lower() in exts:
                key = os.path.splitext(os.path.basename(n))[0]
                ir_map[key] = p

        self.keys = sorted(set(vi_map.keys()) & set(ir_map.keys()))
        if not self.keys:
            raise RuntimeError("No paired samples found (VI/IR filename intersection empty).")

        self.vi_paths = [vi_map[k] for k in self.keys]
        self.ir_paths = [ir_map[k] for k in self.keys]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        vi = Image.open(self.vi_paths[idx]).convert("RGB")
        ir = Image.open(self.ir_paths[idx]).convert("L")
        if self.transform_vi is not None:
            vi = self.transform_vi(vi)
        if self.transform_ir is not None:
            ir = self.transform_ir(ir)
        return vi, ir


# =========================
# Debug Visualization
# =========================
def save_debug_images(save_dir, epoch, vi, y_fused, cb_out, cr_out, w, dcb, dcr,
                      y_in, y_lyt, y_mix, gY, polar=False):
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    num_images = min(5, vi.size(0))
    for idx in range(num_images):
        w_np = w[idx, 0].detach().cpu().numpy()
        dcb_np = dcb[idx, 0].detach().cpu().numpy()
        dcr_np = dcr[idx, 0].detach().cpu().numpy()
        limit = max(abs(dcb_np).max(), abs(dcr_np).max(), 0.01)
        cb_in, cr_in = rgb_to_cbcr(vi)
        rgb_baseline = ycbcr_to_rgb(y_fused, cb_in, cr_in)
        rgb_ours = ycbcr_to_rgb(y_fused, cb_out, cr_out)

        # combined figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        vi_np = vi[idx].permute(1, 2, 0).detach().cpu().numpy()
        axes[0, 0].imshow(vi_np)
        axes[0, 0].set_title(f"Input VI (Img {idx})")
        axes[0, 0].axis("off")

        rgb_base_np = rgb_baseline[idx].permute(1, 2, 0).detach().cpu().numpy()
        axes[0, 1].imshow(rgb_base_np)
        axes[0, 1].set_title("Baseline (Y_fused + CbCr_in)")
        axes[0, 1].axis("off")

        rgb_ours_np = rgb_ours[idx].permute(1, 2, 0).detach().cpu().numpy()
        axes[0, 2].imshow(rgb_ours_np)
        axes[0, 2].set_title("Ours (Y_fused + CbCr_out)")
        axes[0, 2].axis("off")

        w_vmin, w_vmax = (0.01, 3.0) if polar else (0.0, 1.0)
        im_w = axes[1, 0].imshow(w_np, cmap="gray", vmin=w_vmin, vmax=w_vmax)
        axes[1, 0].set_title("Polar alpha" if polar else "Gate w (White=LYT)")
        axes[1, 0].axis("off")
        fig.colorbar(im_w, ax=axes[1, 0], fraction=0.046, pad=0.04)

        im_dcb = axes[1, 1].imshow(dcb_np, cmap="seismic", vmin=-limit, vmax=limit)
        axes[1, 1].set_title(f"Delta Cb (Range: ±{limit:.3f})")
        axes[1, 1].axis("off")
        fig.colorbar(im_dcb, ax=axes[1, 1], fraction=0.046, pad=0.04)

        im_dcr = axes[1, 2].imshow(dcr_np, cmap="seismic", vmin=-limit, vmax=limit)
        axes[1, 2].set_title(f"Delta Cr (Range: ±{limit:.3f})")
        axes[1, 2].axis("off")
        fig.colorbar(im_dcr, ax=axes[1, 2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"epoch_{epoch}_img_{idx}_combined.png"))
        plt.close(fig)

        # =========================
        # Y gate visualization
        # =========================
        y_in_np = y_in[idx, 0].detach().cpu().numpy()
        y_lyt_np = y_lyt[idx, 0].detach().cpu().numpy()
        y_mix_np = y_mix[idx, 0].detach().cpu().numpy()
        gY_np = gY[idx, 0].detach().cpu().numpy()

        dy_lyt = (y_lyt_np - y_in_np)
        dy_mix = (y_mix_np - y_in_np)
        lim = max(float(np.abs(dy_lyt).max()), float(np.abs(dy_mix).max()), 1e-3)

        fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))

        axes2[0, 0].imshow(y_in_np, cmap="gray", vmin=0, vmax=1)
        axes2[0, 0].set_title("Y_in (original)")
        axes2[0, 0].axis("off")

        axes2[0, 1].imshow(y_lyt_np, cmap="gray", vmin=0, vmax=1)
        axes2[0, 1].set_title("Y_lyt (LYT)")
        axes2[0, 1].axis("off")

        axes2[0, 2].imshow(y_mix_np, cmap="gray", vmin=0, vmax=1)
        axes2[0, 2].set_title("Y_mix (gated)")
        axes2[0, 2].axis("off")

        im_g = axes2[1, 0].imshow(gY_np, cmap="gray", vmin=0, vmax=1)
        axes2[1, 0].set_title("gY (White=LYT on Y)")
        axes2[1, 0].axis("off")
        fig2.colorbar(im_g, ax=axes2[1, 0], fraction=0.046, pad=0.04)

        im_d1 = axes2[1, 1].imshow(dy_lyt, cmap="seismic", vmin=-lim, vmax=lim)
        axes2[1, 1].set_title(f"ΔY_lyt = Y_lyt - Y_in (±{lim:.3f})")
        axes2[1, 1].axis("off")
        fig2.colorbar(im_d1, ax=axes2[1, 1], fraction=0.046, pad=0.04)

        im_d2 = axes2[1, 2].imshow(dy_mix, cmap="seismic", vmin=-lim, vmax=lim)
        axes2[1, 2].set_title(f"ΔY_mix = Y_mix - Y_in (±{lim:.3f})")
        axes2[1, 2].axis("off")
        fig2.colorbar(im_d2, ax=axes2[1, 2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"epoch_{epoch}_img_{idx}_Ygate_combined.png"))
        plt.close(fig2)


def save_polar_maps_grid(save_dir, epoch, polar_aux):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    os.makedirs(save_dir, exist_ok=True)

    alpha = polar_aux["alpha_map"][0, 0].detach().float().cpu().numpy()
    log_ratio = polar_aux["log_ratio_map"][0, 0].detach().float().cpu().numpy()
    effective_scale = np.exp(alpha * log_ratio)
    c_orig = polar_aux["C_orig"][0, 0].detach().float().cpu().numpy()
    c_base = polar_aux["C_base"][0, 0].detach().float().cpu().numpy()
    delta_log_c = polar_aux["delta_log_C"][0, 0].detach().float().cpu().numpy()
    c_pred = polar_aux["C_pred"][0, 0].detach().float().cpu().numpy()

    alpha_vmax = max(0.05, float(np.percentile(alpha, 99.0)))
    log_limit = max(0.1, float(np.percentile(np.abs(log_ratio), 99.0)))
    delta_limit = max(0.01, float(np.percentile(np.abs(delta_log_c), 99.0)))
    scale_vmin = float(np.percentile(effective_scale, 1.0))
    scale_vmax = float(np.percentile(effective_scale, 99.0))
    scale_vmin = min(scale_vmin, 0.999)
    scale_vmax = max(scale_vmax, 1.001)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    def draw(axis, data, title, cmap, vmin=None, vmax=None, norm=None):
        image = axis.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
        axis.set_title(title)
        axis.axis("off")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    draw(
        axes[0, 0], alpha,
        f"alpha map\nmean={alpha.mean():.4f}, std={alpha.std():.4f}",
        "viridis", vmin=0.01, vmax=alpha_vmax,
    )
    draw(
        axes[0, 1], log_ratio,
        f"log ratio map\nmean={log_ratio.mean():.4f}",
        "coolwarm", vmin=-log_limit, vmax=log_limit,
    )
    draw(
        axes[0, 2], effective_scale,
        f"effective scale\nmean={effective_scale.mean():.4f}",
        "coolwarm", norm=TwoSlopeNorm(vmin=scale_vmin, vcenter=1.0, vmax=scale_vmax),
    )
    draw(axes[0, 3], c_orig, "C_orig", "viridis", vmin=0.0, vmax=0.5)
    draw(axes[1, 0], c_base, "C_base", "viridis", vmin=0.0, vmax=0.5)
    draw(
        axes[1, 1], delta_log_c,
        f"delta_log_C\nmean|x|={np.abs(delta_log_c).mean():.4f}",
        "seismic", vmin=-delta_limit, vmax=delta_limit,
    )
    draw(axes[1, 2], c_pred, "C_pred", "viridis", vmin=0.0, vmax=0.5)

    axes[1, 3].axis("off")
    summary = (
        f"Epoch: {epoch}\n\n"
        f"alpha min/mean/max:\n{alpha.min():.5f} / {alpha.mean():.5f} / {alpha.max():.5f}\n\n"
        f"log_ratio min/mean/max:\n{log_ratio.min():.4f} / {log_ratio.mean():.4f} / {log_ratio.max():.4f}\n\n"
        f"scale min/mean/max:\n{effective_scale.min():.4f} / {effective_scale.mean():.4f} / {effective_scale.max():.4f}\n\n"
        f"C_orig/base/pred mean:\n{c_orig.mean():.4f} / {c_base.mean():.4f} / {c_pred.mean():.4f}"
    )
    axes[1, 3].text(0.03, 0.97, summary, va="top", ha="left", fontsize=12, family="monospace")

    fig.suptitle("Polar chroma intermediate maps", fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(save_dir, f"epoch_{epoch:04d}_polar_maps.png"), dpi=170)
    plt.close(fig)


# =========================
# Train
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)  # 你可改大，比如 1e-4
    parser.add_argument("--train_path", type=str, default="./checkpoints/stage1_msrs_y_gate_V2")
    parser.add_argument("--data_root", type=str, default="./data/MSRS/")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--shuffle", action="store_true", help="Enable shuffle for train loader (recommended).")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Set 0 to disable grad clipping.")
    parser.add_argument("--resize", type=int, nargs=2, default=None, metavar=('H', 'W'),
                        help="Resize training images to H W, e.g. --resize 120 160")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path, e.g. weight_50.pkl")
    opt = parser.parse_args()

    use_polar_chroma = False
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    polar_loss_fn = None
    os.makedirs(opt.train_path, exist_ok=True)

    # -------------------------
    # Dataset
    # -------------------------
    if opt.resize:
        to_tensor = transforms.Compose([
            transforms.Resize(tuple(opt.resize)),
            transforms.ToTensor(),
        ])
    else:
        to_tensor = transforms.Compose([transforms.ToTensor()])

    train_root_vi = os.path.join(opt.data_root, "train/vi")
    train_root_ir = os.path.join(opt.data_root, "train/ir")
    val_root_vi = os.path.join(opt.data_root, "test/vi")
    val_root_ir = os.path.join(opt.data_root, "test/ir")

    train_set = PairedMSRS(train_root_vi, train_root_ir, transform_vi=to_tensor, transform_ir=to_tensor)
    val_set = PairedMSRS(val_root_vi, val_root_ir, transform_vi=to_tensor, transform_ir=to_tensor)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=opt.batch_size,
        shuffle=True,  # paired dataset safe to shuffle
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    train_iters = len(train_loader)
    val_iters = len(val_loader)

    print("Train_Image_Number:", len(train_set))
    print("Val_Image_Number:", len(val_set))
    print("Train iters:", train_iters, "Val iters:", val_iters)

    # -------------------------
    # Model / Loss / Optim
    # -------------------------
    Net = Net1(use_lyt=True, use_polar_chroma=use_polar_chroma).to(device)
    optimizer = optim.Adam([p for p in Net.parameters() if p.requires_grad], lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [1000], gamma=0.1)

    L1Loss = nn.L1Loss()
    Lg_loss = Fusionloss().to(device)

    start_epoch = 0
    best_loss = 1e9
    best_epoch = 0

    if opt.resume:
        ckpt_path = os.path.join(opt.train_path, opt.resume) if not os.path.isabs(opt.resume) else opt.resume
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            Net.load_state_dict(ckpt["weight"] if "weight" in ckpt else ckpt, strict=False)
            optimizer.load_state_dict(ckpt.get("optimizer_state_dict", optimizer.state_dict()))
            scheduler.load_state_dict(ckpt.get("scheduler_state_dict", scheduler.state_dict()))
            start_epoch = ckpt.get("epoch", 0)
            best_loss = ckpt.get("best_loss", 1e9)
            best_epoch = ckpt.get("best_epoch", start_epoch)
            best_weights_val = ckpt.get("weight")
            print(f"[RESUME] {ckpt_path}  epoch={start_epoch}  best_loss={best_loss:.6f}")
        else:
            print(f"[WARN] resume checkpoint not found: {ckpt_path}")

    best_weights = copy.deepcopy(Net.state_dict())

    # -------------------------
    # Loss weights
    # -------------------------
    lambda_gate = 0.005
    lambda_res = 0.5
    lambda_tv = 0.05
    lambda_dark = 0.02
    lambda_wtv = 1e-5

    # Y gate losses
    lambda_exp = 0.10
    lambda_tvgy = 0.002  # ✅ 降 10 倍：0.02 -> 0.002
    exp_target = 0.60
    exp_patch = 16

    train_start_time = time.time()
    print("============ Training Begins ===============")

    for epoch in range(start_epoch, opt.epochs):
        Net.train()

        train_loss = 0.0
        loss_fuse_acc = 0.0
        loss_gate_acc = 0.0
        loss_res_acc = 0.0
        loss_tv_acc = 0.0
        loss_dark_acc = 0.0
        loss_exp_acc = 0.0
        loss_tvgy_acc = 0.0
        loss_wtv_acc = 0.0
        train_polar_hunt = 0.0
        train_polar_delta = 0.0
        train_polar_grad = 0.0
        train_polar_tv = 0.0
        train_alpha_grad_norm = 0.0
        train_residual_grad_norm = 0.0

        train_abs_dcb_acc = 0.0
        train_abs_dcr_acc = 0.0
        train_max_abs_dcb = 0.0
        train_max_abs_dcr = 0.0

        # w stats + histogram
        train_w_sum = 0.0
        train_w_sumsq = 0.0
        train_w_count = 0
        hist_max = 3.0 if use_polar_chroma else 1.0
        hist_bins = np.linspace(0.0, hist_max, num=51)
        train_hist_counts = np.zeros(50, dtype=np.int64)
        val_hist_counts = np.zeros(50, dtype=np.int64)

        train_enable_sum = 0.0
        train_enable_count = 0

        train_dcb_true_sum = 0.0
        train_dcb_true_cnt = 0
        train_dcb_false_sum = 0.0
        train_dcb_false_cnt = 0
        train_dcr_true_sum = 0.0
        train_dcr_true_cnt = 0
        train_dcr_false_sum = 0.0
        train_dcr_false_cnt = 0

        print(f"\nEpoch {epoch+1}/{opt.epochs}")

        running_true = 0.0
        running_total = 0.0

        for step, (vi, ir) in enumerate(train_loader):
            vi = vi.to(device, non_blocking=True)
            ir = ir.to(device, non_blocking=True)

            optimizer.zero_grad()

            out = Net(
                vi, ir,
                return_y=True, return_ymix=True, return_ylyt=True,
                return_yin=True, return_cbcr=True, return_rgb=True,
                return_gate=True, return_deltas=True,
                return_enable_lyt=True, return_meanY=True, return_ygate=True,
                return_polar=use_polar_chroma,
            )
            if use_polar_chroma:
                y_fused, y_en, y_mix, y_lyt, gY, y_in, cb_out, cr_out, rgb_out, w, dcb, dcr, enable_lyt, meanY, polar_aux = out
            else:
                y_fused, y_en, y_mix, y_lyt, gY, y_in, cb_out, cr_out, rgb_out, w, dcb, dcr, enable_lyt, meanY = out

            # (A) fusion brightness loss
            loss1 = 0.5 * L1Loss(y_en, y_fused) + 0.5 * L1Loss(ir, y_fused)
            Lg = Lg_loss(y_en, ir, y_fused)
            loss_fuse = 0.2 * loss1 + 0.8 * Lg

            # (B) chroma regularizers
            loss_exp = exposure_loss(y_en, target=exp_target, patch=exp_patch)
            loss_tv_gy = tv_loss(gY)

            if not use_polar_chroma:
                bright_mask = (y_in >= 0.35).float()
                loss_gate = (bright_mask * w).sum() / (bright_mask.sum() + 1e-6)

                loss_res = dcb.abs().mean() + dcr.abs().mean()
                loss_tv_v = tv_loss(cb_out) + tv_loss(cr_out)

                # dark guidance (soft + clamp)
                w_target = (1 - y_in).clamp(0.0, 0.85)
                dark_mask = (y_in < 0.3).float()
                diff = F.smooth_l1_loss(w, w_target.detach(), reduction="none")
                loss_dark = (dark_mask * diff).sum() / (dark_mask.sum() + 1e-6)

                # anti-saturation
                loss_gY_sat = (gY * (1.0 - gY)).mean()
                lambda_gysat = 0.001

                # edge-aware smoothness on w
                loss_wtv = edge_aware_tv(w, guide=y_in)

                loss = (
                    loss_fuse
                    + lambda_gate * loss_gate
                    + lambda_res * loss_res
                    + lambda_tv * loss_tv_v
                    + lambda_dark * loss_dark
                    + lambda_exp * loss_exp
                    + lambda_tvgy * loss_tv_gy
                    + lambda_wtv * loss_wtv
                )
                loss = loss - lambda_gysat * loss_gY_sat
            else:
                cb_in, cr_in = rgb_to_cbcr(vi)
                # LYT reference for chroma
                with torch.no_grad():
                    vis_lyt = Net.lyt_enhancer(vi)
                cb_ref, cr_ref = rgb_to_cbcr(vis_lyt)
                polar_loss, polar_dict = polar_loss_fn(
                    Cb_out=cb_out, Cr_out=cr_out,
                    Cb_in=cb_in, Cr_in=cr_in,
                    aux=polar_aux,
                    Cb_ref=cb_ref, Cr_ref=cr_ref)
                loss = loss_fuse + lambda_exp * loss_exp + lambda_tvgy * loss_tv_gy + polar_loss

            loss.backward()

            if use_polar_chroma:
                alpha_grad_norm = module_grad_norm(Net.chroma_head.alpha_predictor)
                residual_grad_norm = module_grad_norm(Net.chroma_head.residual_net)
                train_alpha_grad_norm += alpha_grad_norm
                train_residual_grad_norm += residual_grad_norm

            # ✅ optional: grad clipping (helps when you increase lr)
            if opt.grad_clip and opt.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(Net.parameters(), opt.grad_clip)

            optimizer.step()

            # stats
            train_loss += loss.item()
            loss_fuse_acc += loss_fuse.item()
            loss_exp_acc += loss_exp.item()
            loss_tvgy_acc += loss_tv_gy.item()
            if not use_polar_chroma:
                loss_gate_acc += loss_gate.item()
                loss_res_acc += loss_res.item()
                loss_tv_acc += loss_tv_v.item()
                loss_dark_acc += loss_dark.item()
                loss_wtv_acc += loss_wtv.item()
            else:
                train_polar_hunt += polar_dict['L_hunt']
                train_polar_delta += polar_dict['L_delta']
                train_polar_grad += polar_dict['L_grad']
                train_polar_tv += polar_dict['L_tv']

            train_abs_dcb_acc += dcb.abs().mean().item()
            train_abs_dcr_acc += dcr.abs().mean().item()
            train_max_abs_dcb = max(train_max_abs_dcb, dcb.abs().max().item())
            train_max_abs_dcr = max(train_max_abs_dcr, dcr.abs().max().item())

            abs_dcb_img = dcb.abs().mean(dim=(2, 3)).squeeze(1)
            abs_dcr_img = dcr.abs().mean(dim=(2, 3)).squeeze(1)
            enable_mask = enable_lyt.reshape(-1).bool()
            if enable_mask.any():
                train_dcb_true_sum += abs_dcb_img[enable_mask].sum().item()
                train_dcb_true_cnt += int(enable_mask.sum().item())
                train_dcr_true_sum += abs_dcr_img[enable_mask].sum().item()
                train_dcr_true_cnt += int(enable_mask.sum().item())
            disable_mask = ~enable_mask
            if disable_mask.any():
                train_dcb_false_sum += abs_dcb_img[disable_mask].sum().item()
                train_dcb_false_cnt += int(disable_mask.sum().item())
                train_dcr_false_sum += abs_dcr_img[disable_mask].sum().item()
                train_dcr_false_cnt += int(disable_mask.sum().item())

            train_w_sum += w.sum().item()
            train_w_sumsq += (w * w).sum().item()
            train_w_count += w.numel()
            w_np = w.detach().cpu().numpy()
            counts, _ = np.histogram(w_np, bins=hist_bins)
            train_hist_counts += counts

            train_enable_sum += enable_lyt.float().sum().item()
            train_enable_count += enable_lyt.numel()

            running_true += enable_lyt.float().sum().item()
            running_total += enable_lyt.numel()

            if step % opt.log_interval == 0:
                batch_ratio = enable_lyt.float().mean().item()
                running_ratio = running_true / max(running_total, 1.0)
                mean_meanY = meanY.mean().item()

                # optional: saturation ratio for w
                w_sat = ((w < 0.05) | (w > 0.95)).float().mean().item()

                if use_polar_chroma:
                    print(
                        f"  step {step:04d} loss={loss.item():.4f}  "
                        f"ref={polar_dict.get('L_ref',0):.4f}  "
                        f"dlogC={polar_dict['L_delta_C']:.4f}  dh={polar_dict['L_delta_h']:.4f}  "
                        f"gradCb={polar_dict['L_grad_cb']:.4f}  gradCr={polar_dict['L_grad_cr']:.4f}  "
                        f"tv={polar_dict['L_tv']:.4f}  "
                        f"alpha={w.mean().item():.4f}+/-{w.std().item():.4f}  "
                        f"|dlogC|={dcb.abs().mean().item():.5f}  |dh|={dcr.abs().mean().item():.5f}  "
                        f"grad_alpha={alpha_grad_norm:.3e}  grad_res={residual_grad_norm:.3e}  "
                        f"lyt={batch_ratio:.2f}"
                    )
                else:
                    print(
                        f"step {step:04d}/{train_iters} | "
                        f"loss={loss.item():.4f} "
                        f"(fuse={loss_fuse.item():.4f}, gate={loss_gate.item():.4f}, res={loss_res.item():.4f}, "
                        f"tv={loss_tv_v.item():.4f}, dark={loss_dark.item():.4f}, exp={loss_exp.item():.4f}, "
                        f"tvgy={loss_tv_gy.item():.4f}, wtv={loss_wtv.item():.6f}) | "
                        f"w_mean={w.mean().item():.3f} w_sat={w_sat:.3f} gY_mean={gY.mean().item():.3f} | "
                        f"|dcb|={dcb.abs().mean().item():.4f} |dcr|={dcr.abs().mean().item():.4f} | "
                        f"meanY={mean_meanY:.3f} enable_lyt_ratio(batch/run)={batch_ratio:.3f}/{running_ratio:.3f}"
                    )

        train_epoch_loss = train_loss / max(train_iters, 1)
        train_enable_ratio = (train_enable_sum / train_enable_count) if train_enable_count > 0 else 0.0
        train_mean_abs_dcb_true = train_dcb_true_sum / train_dcb_true_cnt if train_dcb_true_cnt > 0 else 0.0
        train_mean_abs_dcb_false = train_dcb_false_sum / train_dcb_false_cnt if train_dcb_false_cnt > 0 else 0.0
        train_mean_abs_dcr_true = train_dcr_true_sum / train_dcr_true_cnt if train_dcr_true_cnt > 0 else 0.0
        train_mean_abs_dcr_false = train_dcr_false_sum / train_dcr_false_cnt if train_dcr_false_cnt > 0 else 0.0
        if train_w_count > 0:
            train_w_mean = train_w_sum / train_w_count
            train_w_var = max(0.0, train_w_sumsq / train_w_count - train_w_mean * train_w_mean)
            train_w_std = math.sqrt(train_w_var)
        else:
            train_w_mean, train_w_std = 0.0, 0.0
        if use_polar_chroma:
            print(f"  train loss={train_epoch_loss:.6f}  fuse={loss_fuse_acc/train_iters:.6f}  "
                  f"hunt={train_polar_hunt/train_iters:.4f}  "
                  f"delta={train_polar_delta/train_iters:.4f}  "
                  f"grad={train_polar_grad/train_iters:.4f}  "
                  f"ptv={train_polar_tv/train_iters:.4f}  "
                  f"alpha={train_w_mean:.4f}+/-{train_w_std:.4f}  "
                  f"|dlogC|={train_abs_dcb_acc/train_iters:.5f}  "
                  f"|dh|={train_abs_dcr_acc/train_iters:.5f}  "
                  f"grad_alpha={train_alpha_grad_norm/train_iters:.3e}  "
                  f"grad_res={train_residual_grad_norm/train_iters:.3e}  "
                  f"lyt={train_enable_ratio:.3f}")
        else:
            print(f"Train loss: {train_epoch_loss:.6f}")
            print(f"  fuse: {loss_fuse_acc/train_iters:.6f}")
            print(f"  gate: {loss_gate_acc/train_iters:.6f}")
            print(f"  res: {loss_res_acc/train_iters:.6f}")
            print(f"  tv: {loss_tv_acc/train_iters:.6f}")
            print(f"  dark: {loss_dark_acc/train_iters:.6f}")
            print(f"  exp: {loss_exp_acc/train_iters:.6f}")
            print(f"  tvgy: {loss_tvgy_acc/train_iters:.6f}")
            print(f"  wtv: {loss_wtv_acc/train_iters:.6f}")
            print(f"  |dcb|: {train_abs_dcb_acc/train_iters:.6f}")
            print(f"  |dcr|: {train_abs_dcr_acc/train_iters:.6f}")
            print(f"  max|dcb|: {train_max_abs_dcb:.6f}")
            print(f"  max|dcr|: {train_max_abs_dcr:.6f}")

        if not use_polar_chroma:
            print(f"  |dcb| (enable_lyt=T/F): {train_mean_abs_dcb_true:.6f} / {train_mean_abs_dcb_false:.6f}")
            print(f"  |dcr| (enable_lyt=T/F): {train_mean_abs_dcr_true:.6f} / {train_mean_abs_dcr_false:.6f}")
            print(f"  w mean/std (train): {train_w_mean:.6f} / {train_w_std:.6f}")
            print(f"  enable_lyt_ratio (train): {train_enable_ratio:.6f}")

        # =========================
        # Validation
        # =========================
        Net.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_abs_dcb_acc = 0.0
            val_abs_dcr_acc = 0.0
            val_max_abs_dcb = 0.0
            val_max_abs_dcr = 0.0
            val_exp_acc = 0.0
            val_tvgy_acc = 0.0
            val_wtv_acc = 0.0
            val_polar_hunt_v = 0.0
            val_polar_delta_v = 0.0
            val_polar_grad_v = 0.0
            val_polar_tv_v = 0.0

            val_dcb_true_sum = 0.0
            val_dcb_true_cnt = 0
            val_dcb_false_sum = 0.0
            val_dcb_false_cnt = 0
            val_dcr_true_sum = 0.0
            val_dcr_true_cnt = 0
            val_dcr_false_sum = 0.0
            val_dcr_false_cnt = 0

            val_w_sum = 0.0
            val_w_sumsq = 0.0
            val_w_count = 0

            val_enable_sum = 0.0
            val_enable_count = 0

            # region accumulators (pixel-level)
            val_w_bright_sum = 0.0
            val_w_dark_sum = 0.0
            val_dcb_bright_sum = 0.0
            val_dcb_dark_sum = 0.0
            val_dcr_bright_sum = 0.0
            val_dcr_dark_sum = 0.0
            val_bright_cnt = 0.0
            val_dark_cnt = 0.0
            val_total_pixels = 0.0

            last_batch_cache = None

            for step, (vi, ir) in enumerate(val_loader):
                vi = vi.to(device, non_blocking=True)
                ir = ir.to(device, non_blocking=True)

                out = Net(
                    vi, ir,
                    return_y=True, return_ymix=True, return_ylyt=True,
                    return_yin=True, return_cbcr=True, return_rgb=True,
                    return_gate=True, return_deltas=True,
                    return_enable_lyt=True, return_meanY=True, return_ygate=True,
                    return_polar=use_polar_chroma,
                )
                if use_polar_chroma:
                    y_fused, y_en, y_mix, y_lyt, gY, y_in, cb_out, cr_out, rgb_out, w, dcb, dcr, enable_lyt, meanY, polar_aux = out
                else:
                    y_fused, y_en, y_mix, y_lyt, gY, y_in, cb_out, cr_out, rgb_out, w, dcb, dcr, enable_lyt, meanY = out

                loss1 = 0.5 * L1Loss(y_en, y_fused) + 0.5 * L1Loss(ir, y_fused)
                Lg = Lg_loss(y_en, ir, y_fused)
                loss_fuse = 0.2 * loss1 + 0.8 * Lg

                loss_exp = exposure_loss(y_en, target=exp_target, patch=exp_patch)
                loss_tv_gy = tv_loss(gY)
                if not use_polar_chroma:
                    bright_mask = (y_in >= 0.35).float()
                    loss_gate = (bright_mask * w).sum() / (bright_mask.sum() + 1e-6)
                    loss_res = dcb.abs().mean() + dcr.abs().mean()
                    loss_tv_v = tv_loss(cb_out) + tv_loss(cr_out)
                    w_target = (1 - y_in).clamp(0.0, 0.85)
                    dark_mask = (y_in < 0.3).float()
                    diff = F.smooth_l1_loss(w, w_target.detach(), reduction="none")
                    loss_dark = (dark_mask * diff).sum() / (dark_mask.sum() + 1e-6)
                    loss_wtv = edge_aware_tv(w, guide=y_in)
                    loss = (loss_fuse + lambda_gate * loss_gate + lambda_res * loss_res
                           + lambda_tv * loss_tv_v + lambda_dark * loss_dark
                           + lambda_exp * loss_exp + lambda_tvgy * loss_tv_gy + lambda_wtv * loss_wtv)
                    val_loss += loss.item()
                    val_exp_acc += loss_exp.item()
                    val_tvgy_acc += loss_tv_gy.item()
                    val_wtv_acc += loss_wtv.item()
                    val_abs_dcb_acc += dcb.abs().mean().item()
                    val_abs_dcr_acc += dcr.abs().mean().item()
                    val_max_abs_dcb = max(val_max_abs_dcb, dcb.abs().max().item())
                    val_max_abs_dcr = max(val_max_abs_dcr, dcr.abs().max().item())
                    abs_dcb_img = dcb.abs().mean(dim=(2,3)).squeeze(1)
                    abs_dcr_img = dcr.abs().mean(dim=(2,3)).squeeze(1)
                    enable_mask = enable_lyt.reshape(-1).bool()
                    if enable_mask.any():
                        val_dcb_true_sum += abs_dcb_img[enable_mask].sum().item()
                        val_dcb_true_cnt += int(enable_mask.sum().item())
                        val_dcr_true_sum += abs_dcr_img[enable_mask].sum().item()
                        val_dcr_true_cnt += int(enable_mask.sum().item())
                    disable_mask = ~enable_mask
                    if disable_mask.any():
                        val_dcb_false_sum += abs_dcb_img[disable_mask].sum().item()
                        val_dcb_false_cnt += int(disable_mask.sum().item())
                        val_dcr_false_sum += abs_dcr_img[disable_mask].sum().item()
                        val_dcr_false_cnt += int(disable_mask.sum().item())
                    val_w_sum += w.sum().item()
                    val_w_sumsq += (w*w).sum().item()
                    val_w_count += w.numel()
                    val_enable_sum += enable_lyt.float().sum().item()
                    val_enable_count += enable_lyt.numel()
                    st = region_stats_w_deltas(w, dcb, dcr, y_in, bright_thr=0.35, dark_thr=0.30)
                    val_bright_cnt += float(st["cnt_bright"].item())
                    val_dark_cnt += float(st["cnt_dark"].item())
                    val_total_pixels += float(y_in.numel())
                    val_w_bright_sum += float(st["w_bright"].item())*float(st["cnt_bright"].item())
                    val_w_dark_sum += float(st["w_dark"].item())*float(st["cnt_dark"].item())
                    val_dcb_bright_sum += float(st["dcb_bright"].item())*float(st["cnt_bright"].item())
                    val_dcb_dark_sum += float(st["dcb_dark"].item())*float(st["cnt_dark"].item())
                    val_dcr_bright_sum += float(st["dcr_bright"].item())*float(st["cnt_bright"].item())
                    val_dcr_dark_sum += float(st["dcr_dark"].item())*float(st["cnt_dark"].item())
                    last_batch_cache = (vi, y_fused, cb_out, cr_out, w, dcb, dcr, y_in, y_lyt, y_mix, gY)
                else:
                    cb_in_v, cr_in_v = rgb_to_cbcr(vi)
                    with torch.no_grad():
                        vis_lyt_v = Net.lyt_enhancer(vi)
                    cb_ref_v, cr_ref_v = rgb_to_cbcr(vis_lyt_v)
                    polar_loss_v, polar_dict_v = polar_loss_fn(
                        Cb_out=cb_out, Cr_out=cr_out, Cb_in=cb_in_v, Cr_in=cr_in_v,
                        aux=polar_aux, Cb_ref=cb_ref_v, Cr_ref=cr_ref_v)
                    loss = loss_fuse + lambda_exp * loss_exp + lambda_tvgy * loss_tv_gy + polar_loss_v
                    val_loss += loss.item()
                    val_exp_acc += loss_exp.item()
                    val_tvgy_acc += loss_tv_gy.item()
                    val_enable_sum += enable_lyt.float().sum().item()
                    val_enable_count += enable_lyt.numel()
                    val_polar_hunt_v += polar_dict_v["L_hunt"]
                    val_polar_delta_v += polar_dict_v["L_delta"]
                    val_polar_grad_v += polar_dict_v["L_grad"]
                    val_polar_tv_v += polar_dict_v["L_tv"]
                    val_abs_dcb_acc += dcb.abs().mean().item()
                    val_abs_dcr_acc += dcr.abs().mean().item()
                    val_max_abs_dcb = max(val_max_abs_dcb, dcb.abs().max().item())
                    val_max_abs_dcr = max(val_max_abs_dcr, dcr.abs().max().item())
                    abs_dcb_img = dcb.abs().mean(dim=(2, 3)).squeeze(1)
                    abs_dcr_img = dcr.abs().mean(dim=(2, 3)).squeeze(1)
                    enable_mask = enable_lyt.reshape(-1).bool()
                    if enable_mask.any():
                        val_dcb_true_sum += abs_dcb_img[enable_mask].sum().item()
                        val_dcb_true_cnt += int(enable_mask.sum().item())
                        val_dcr_true_sum += abs_dcr_img[enable_mask].sum().item()
                        val_dcr_true_cnt += int(enable_mask.sum().item())
                    disable_mask = ~enable_mask
                    if disable_mask.any():
                        val_dcb_false_sum += abs_dcb_img[disable_mask].sum().item()
                        val_dcb_false_cnt += int(disable_mask.sum().item())
                        val_dcr_false_sum += abs_dcr_img[disable_mask].sum().item()
                        val_dcr_false_cnt += int(disable_mask.sum().item())
                    val_w_sum += w.sum().item()
                    val_w_sumsq += (w * w).sum().item()
                    val_w_count += w.numel()
                    st = region_stats_w_deltas(w, dcb, dcr, y_in, bright_thr=0.35, dark_thr=0.30)
                    val_bright_cnt += float(st["cnt_bright"].item())
                    val_dark_cnt += float(st["cnt_dark"].item())
                    val_total_pixels += float(y_in.numel())
                    val_w_bright_sum += float(st["w_bright"].item()) * float(st["cnt_bright"].item())
                    val_w_dark_sum += float(st["w_dark"].item()) * float(st["cnt_dark"].item())
                    val_dcb_bright_sum += float(st["dcb_bright"].item()) * float(st["cnt_bright"].item())
                    val_dcb_dark_sum += float(st["dcb_dark"].item()) * float(st["cnt_dark"].item())
                    val_dcr_bright_sum += float(st["dcr_bright"].item()) * float(st["cnt_bright"].item())
                    val_dcr_dark_sum += float(st["dcr_dark"].item()) * float(st["cnt_dark"].item())
                    last_batch_cache = (vi, y_fused, cb_out, cr_out, w, dcb, dcr, y_in, y_lyt, y_mix, gY)

            val_epoch_loss = val_loss / max(val_iters, 1)
            val_enable_ratio = (val_enable_sum / val_enable_count) if val_enable_count > 0 else 0.0
            val_mean_abs_dcb = val_abs_dcb_acc / max(val_iters, 1)
            val_mean_abs_dcr = val_abs_dcr_acc / max(val_iters, 1)
            val_mean_abs_dcb_true = val_dcb_true_sum / val_dcb_true_cnt if val_dcb_true_cnt > 0 else 0.0
            val_mean_abs_dcb_false = val_dcb_false_sum / val_dcb_false_cnt if val_dcb_false_cnt > 0 else 0.0
            val_mean_abs_dcr_true = val_dcr_true_sum / val_dcr_true_cnt if val_dcr_true_cnt > 0 else 0.0
            val_mean_abs_dcr_false = val_dcr_false_sum / val_dcr_false_cnt if val_dcr_false_cnt > 0 else 0.0
            if val_w_count > 0:
                val_w_mean = val_w_sum / val_w_count
                val_w_var = max(0.0, val_w_sumsq / val_w_count - val_w_mean * val_w_mean)
                val_w_std = math.sqrt(val_w_var)
            else:
                val_w_mean, val_w_std = 0.0, 0.0
            bright_ratio = val_bright_cnt / max(val_total_pixels, 1.0)
            dark_ratio = val_dark_cnt / max(val_total_pixels, 1.0)
            w_bright_mean = val_w_bright_sum / max(val_bright_cnt, 1.0)
            w_dark_mean = val_w_dark_sum / max(val_dark_cnt, 1.0)
            dcb_bright_mean = val_dcb_bright_sum / max(val_bright_cnt, 1.0)
            dcb_dark_mean = val_dcb_dark_sum / max(val_dark_cnt, 1.0)
            dcr_bright_mean = val_dcr_bright_sum / max(val_bright_cnt, 1.0)
            dcr_dark_mean = val_dcr_dark_sum / max(val_dark_cnt, 1.0)
            if use_polar_chroma:
                print(f"  val   loss={val_epoch_loss:.6f}  "
                      f"hunt={val_polar_hunt_v/val_iters:.4f}  "
                      f"delta={val_polar_delta_v/val_iters:.4f}  "
                      f"grad={val_polar_grad_v/val_iters:.4f}  "
                      f"ptv={val_polar_tv_v/val_iters:.4f}  "
                      f"alpha={val_w_mean:.4f}+/-{val_w_std:.4f}  "
                      f"|dlogC|={val_mean_abs_dcb:.5f}  "
                      f"|dh|={val_mean_abs_dcr:.5f}  "
                      f"lyt={val_enable_ratio:.3f}")
            else:
                print(f"Val loss: {val_epoch_loss:.6f}")
                print(f"Val exp: {val_exp_acc/val_iters:.6f}")
                print(f"Val tvgy: {val_tvgy_acc/val_iters:.6f}")
                if val_wtv_acc:
                    print(f"Val wtv: {val_wtv_acc/val_iters:.6f}")
                print(f"Val |dcb| mean: {val_mean_abs_dcb:.6f}")
                print(f"Val |dcr| mean: {val_mean_abs_dcr:.6f}")
                print(f"Val w mean/std (val): {val_w_mean:.6f} / {val_w_std:.6f}")
                print(f"Val enable_lyt_ratio (val): {val_enable_ratio:.6f}")

            # save debug images using fixed reference 00004N
            ref_vi = Image.open(os.path.join(opt.data_root, 'test/vi/00004N.png')).convert('RGB')
            ref_ir = Image.open(os.path.join(opt.data_root, 'test/ir/00004N.png')).convert('L')
            ref_vi_t = torch.from_numpy(np.array(ref_vi, np.float32)/255).permute(2,0,1).unsqueeze(0).to(device)
            ref_ir_t = torch.from_numpy(np.array(ref_ir, np.float32)/255).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                was_training = Net.training
                Net.train()
                ref_outputs = Net(
                    ref_vi_t, ref_ir_t,
                    return_y=True, return_ymix=True, return_ylyt=True, return_yin=True,
                    return_cbcr=True, return_ygate=True,
                    return_gate=True, return_deltas=True,
                    return_polar=use_polar_chroma)
                if not was_training:
                    Net.eval()
                if use_polar_chroma:
                    (ref_y_f, ref_y_en, ref_y_mix, ref_y_lyt, ref_gY, ref_y_in,
                     ref_cb_out, ref_cr_out, ref_w, ref_dcb, ref_dcr,
                     ref_polar_aux) = ref_outputs
                else:
                    (ref_y_f, ref_y_en, ref_y_mix, ref_y_lyt, ref_gY, ref_y_in,
                     ref_cb_out, ref_cr_out, ref_w, ref_dcb, ref_dcr) = ref_outputs
                save_debug_images(
                    os.path.join(opt.train_path, "vis"), epoch + 1,
                    ref_vi_t, ref_y_f, ref_cb_out, ref_cr_out,
                    ref_w, ref_dcb, ref_dcr,
                    ref_y_in, ref_y_lyt, ref_y_mix, ref_gY,
                    polar=use_polar_chroma,
                )
                if use_polar_chroma:
                    save_polar_maps_grid(
                        os.path.join(opt.train_path, "vis"), epoch + 1, ref_polar_aux
                    )

            # CSV
            candidates = [
                os.path.join(opt.train_path, "metrics.csv"),
                os.path.join(opt.train_path, "metrics_v2.csv"),
                os.path.join(opt.train_path, "metrics_v3.csv"),
                os.path.join(opt.train_path, "metrics_v4.csv"),
            ]
            metrics_path = None
            for c in candidates:
                if not os.path.exists(c):
                    metrics_path = c
                    break
            if metrics_path is None:
                metrics_path = os.path.join(opt.train_path, "metrics_v5.csv")

            write_header = not os.path.exists(metrics_path)
            try:
                with open(metrics_path, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    if write_header:
                        writer.writerow(
                            [
                                "epoch",
                                "train_loss",
                                "val_loss",
                                "train_mean_abs_dcb",
                                "train_mean_abs_dcr",
                                "val_mean_abs_dcb",
                                "val_mean_abs_dcr",
                                "train_max_abs_dcb",
                                "train_max_abs_dcr",
                                "val_max_abs_dcb",
                                "val_max_abs_dcr",
                                "train_mean_abs_dcb_lyt_true",
                                "train_mean_abs_dcb_lyt_false",
                                "train_mean_abs_dcr_lyt_true",
                                "train_mean_abs_dcr_lyt_false",
                                "val_mean_abs_dcb_lyt_true",
                                "val_mean_abs_dcb_lyt_false",
                                "val_mean_abs_dcr_lyt_true",
                                "val_mean_abs_dcr_lyt_false",
                                "train_w_mean",
                                "train_w_std",
                                "val_w_mean",
                                "val_w_std",
                                "train_enable_ratio",
                                "val_enable_ratio",
                                # region stats (val pixel-level)
                                "val_bright_ratio",
                                "val_dark_ratio",
                                "val_w_bright_mean",
                                "val_w_dark_mean",
                                "val_abs_dcb_bright_mean",
                                "val_abs_dcb_dark_mean",
                                "val_abs_dcr_bright_mean",
                                "val_abs_dcr_dark_mean",
                            ]
                        )

                    writer.writerow(
                        [
                            epoch + 1,
                            train_epoch_loss,
                            val_epoch_loss,
                            train_abs_dcb_acc / train_iters,
                            train_abs_dcr_acc / train_iters,
                            val_mean_abs_dcb,
                            val_mean_abs_dcr,
                            train_max_abs_dcb,
                            train_max_abs_dcr,
                            val_max_abs_dcb,
                            val_max_abs_dcr,
                            train_mean_abs_dcb_true,
                            train_mean_abs_dcb_false,
                            train_mean_abs_dcr_true,
                            train_mean_abs_dcr_false,
                            val_mean_abs_dcb_true,
                            val_mean_abs_dcb_false,
                            val_mean_abs_dcr_true,
                            val_mean_abs_dcr_false,
                            train_w_mean,
                            train_w_std,
                            val_w_mean,
                            val_w_std,
                            train_enable_ratio,
                            val_enable_ratio,
                            float(bright_ratio),
                            float(dark_ratio),
                            float(w_bright_mean),
                            float(w_dark_mean),
                            float(dcb_bright_mean),
                            float(dcb_dark_mean),
                            float(dcr_bright_mean),
                            float(dcr_dark_mean),
                        ]
                    )
            except Exception as e:
                print(f"[WARN] Failed to write metrics.csv: {e}")

        # =========================
        # Save best / checkpoints
        # =========================
        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            best_epoch = epoch + 1
            best_weights = copy.deepcopy(Net.state_dict())
            torch.save(
                {
                    "weight": best_weights,
                    "epoch": best_epoch,
                    "loss": train_epoch_loss,
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(opt.train_path, "best_weight.pkl"),
            )
            print(f"[BEST] epoch={best_epoch} val_loss={best_loss:.6f}")

        if (epoch + 1) % 50 == 0:
            weights = copy.deepcopy(Net.state_dict())
            torch.save(
                {
                    "weight": weights,
                    "epoch": epoch + 1,
                    "loss": train_epoch_loss,
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(opt.train_path, f"weight_{epoch+1}.pkl"),
            )
            print(f"[SAVE] weight_{epoch+1}.pkl")

        scheduler.step()

        train_hours = (time.time() - train_start_time) / 3600
        print(f"Elapsed: {train_hours:.4f} hours | best_epoch={best_epoch}")

    print("============ Training Finished ===============")
    print(f"Best epoch: {best_epoch}, best val loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()
    os.system("/usr/bin/shutdown")
