"""Train the MICG-1 reliability mapper for raw and LYT luminance.

The deterministic quality/variance proxies remain fixed; only the monotonic
mapping functions are optimized. The defaults reproduce the paper's MSRS
training resolution and schedule.
"""

import sys, os, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch, torch.nn.functional as F, numpy as np
from PIL import Image
from tqdm import tqdm
from fusion_model import PretrainedLYTWithFeatures, rgb_to_ycbcr
from credibility.models.prob_fusion_calibrated import (
    ProbabilisticCalibratedFusion,
)

RESAMPLE_BILINEAR = (
    Image.Resampling.BILINEAR
    if hasattr(Image, 'Resampling')
    else Image.BILINEAR
)


class PairedPhysicsSet(torch.utils.data.Dataset):
    def __init__(self, data_root, split='train', num=None, image_size=(640, 480)):
        vi_dir = os.path.join(data_root, split, 'vi')
        self.names = sorted([f for f in os.listdir(vi_dir) if f.endswith('.png')])
        if num: self.names = self.names[:num]
        self.data_root, self.split = data_root, split
        self.image_size = tuple(int(value) for value in image_size)

    def __len__(self): return len(self.names)
    def __getitem__(self, idx):
        name = self.names[idx]
        vi = Image.open(os.path.join(self.data_root, self.split, 'vi', name)).convert('RGB')
        ir = Image.open(os.path.join(self.data_root, self.split, 'ir', name)).convert('L')
        if vi.size != ir.size:
            raise ValueError(
                f'Visible/infrared size mismatch for {name}: '
                f'visible={vi.size}, infrared={ir.size}'
            )
        vi = vi.resize(self.image_size, RESAMPLE_BILINEAR)
        ir = ir.resize(self.image_size, RESAMPLE_BILINEAR)
        vi_t = torch.from_numpy(np.array(vi,np.float32)/255).permute(2,0,1)
        ir_t = torch.from_numpy(np.array(ir,np.float32)/255).unsqueeze(0)
        return vi_t, ir_t


@torch.no_grad()
def prepare_luminance(vi, lyt):
    """Generate raw/LYT luminance on the same device as the input batch."""
    vi_lyt = lyt(vi)
    y_in, _, _ = rgb_to_ycbcr(vi)
    y_lyt = (
        0.299 * vi_lyt[:, 0:1]
        + 0.587 * vi_lyt[:, 1:2]
        + 0.114 * vi_lyt[:, 2:3]
    )
    return y_in, y_lyt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='Training batch size.',
    )
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument(
        '--ckpt_dir', default='./credibility/checkpoints'
    )
    parser.add_argument('--data_root', default='./data/MSRS')
    parser.add_argument('--lyt_weights', default='./checkpoints/lyt.pth')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    opt = parser.parse_args()
    os.makedirs(os.path.join(opt.ckpt_dir, 'calib_v2_maps'), exist_ok=True)
    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')

    image_size = (opt.width, opt.height)
    train_set = PairedPhysicsSet(
        opt.data_root, 'train', num=None, image_size=image_size
    )
    val_set = PairedPhysicsSet(
        opt.data_root, 'test', num=None, image_size=image_size
    )
    loader_args = dict(
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=device.type == 'cuda',
        persistent_workers=opt.num_workers > 0,
    )
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    print(
        f'Train:{len(train_set)} Val:{len(val_set)} '
        f'resolution:{opt.width}x{opt.height} '
        f'batch_size:{opt.batch_size} workers:{opt.num_workers} device:{device}'
    )

    lyt = PretrainedLYTWithFeatures(
        weights_path=opt.lyt_weights,
        device=device,
    ).to(device).eval()
    net = ProbabilisticCalibratedFusion(hidden=16).to(device)
    calib_params = list(net.calibrator.parameters())
    print(f'Calibrator params: {sum(p.numel() for p in calib_params)}')
    optim = torch.optim.Adam(calib_params, lr=opt.lr)
    best_v = 1e9

    for epoch in range(opt.epochs):
        net.train()
        tr_loss = 0
        train_bar = tqdm(
            train_loader,
            desc=f'Train {epoch + 1:03d}/{opt.epochs:03d}',
            unit='batch',
            dynamic_ncols=True,
        )
        for batch_index, (vi, ir) in enumerate(train_bar, 1):
            vi = vi.to(device, non_blocking=True)
            ir = ir.to(device, non_blocking=True)
            y_in, y_lyt = prepare_luminance(vi, lyt)
            optim.zero_grad(set_to_none=True)
            T_lyt, C_total, out, T_base, C_base = net(y_in, y_lyt, ir)

            # ── 人工干预损失 (三项验收) ──
            # 1. 干预正确: 暗平坦区 T↓, 暗边区 T↑
            from credibility.models.prob_fusion import scharr
            gx, gy = scharr(y_in.detach()); g = (gx.abs() + gy.abs()).detach()
            dark_mask = (y_in < 0.15).float().detach()
            edge_mask = (g > g.mean()).float().detach() * dark_mask   # 暗边界
            flat_mask = (g < g.mean()).float().detach() * dark_mask   # 暗平坦

            L_flat = (T_lyt * flat_mask).sum() / (flat_mask.sum() + 1)   # 暗平坦→T↓
            L_edge = ((1 - T_lyt) * edge_mask).sum() / (edge_mask.sum() + 1)  # 暗边界→T↑

            # 2. 有效贡献: Q_lyt占比高区 → T 应高
            q_target = T_base.detach()
            L_quality = F.mse_loss(T_lyt, q_target)

            # 3. 噪声验收: T本身噪声 ≤ 原图噪声
            Te_x, Te_y = scharr(T_lyt); T_edge = (Te_x.abs() + Te_y.abs())
            gxi, gyi = scharr(y_in.detach()); g_in = (gxi.abs() + gyi.abs())
            L_T_noise = F.mse_loss(T_edge, g_in * 0.5)

            loss = L_quality + 0.3 * L_flat + 0.3 * L_edge + 0.05 * L_T_noise
            loss.backward()
            optim.step()
            tr_loss += loss.item()
            train_bar.set_postfix(
                loss=f'{tr_loss / batch_index:.4f}',
                quality=f'{L_quality.item():.4f}',
                flat=f'{L_flat.item():.4f}',
                edge=f'{L_edge.item():.4f}',
                noise=f'{L_T_noise.item():.4f}',
            )
        tr_loss /= max(len(train_loader), 1)

        # val
        net.eval()
        vl = 0
        with torch.no_grad():
            val_bar = tqdm(
                val_loader,
                desc=f'Val   {epoch + 1:03d}/{opt.epochs:03d}',
                unit='batch',
                dynamic_ncols=True,
            )
            for batch_index, (vi, ir) in enumerate(val_bar, 1):
                vi = vi.to(device, non_blocking=True)
                ir = ir.to(device, non_blocking=True)
                y_in, y_lyt = prepare_luminance(vi, lyt)
                T, C, out, Tb, _ = net(y_in, y_lyt, ir)
                target = Tb
                vl += F.mse_loss(T, target).item()
                val_bar.set_postfix(val=f'{vl / batch_index:.4f}')
        vl /= max(len(val_loader), 1)
        print(f'E{epoch+1:3d}  train={tr_loss:.4f}  val={vl:.4f}')

        # 每10轮保存验收图
        if (epoch+1) % 10 == 0 or vl < best_v:
            ref_name = val_set.names[0]
            ref_vi = Image.open(Path(opt.data_root) / 'test' / 'vi' / ref_name).convert('RGB')
            ref_ir = Image.open(Path(opt.data_root) / 'test' / 'ir' / ref_name).convert('L')
            ref_vi = ref_vi.resize(image_size, RESAMPLE_BILINEAR)
            ref_ir = ref_ir.resize(image_size, RESAMPLE_BILINEAR)
            vt = torch.from_numpy(np.array(ref_vi,np.float32)/255).permute(2,0,1).unsqueeze(0).to(device)
            it = torch.from_numpy(np.array(ref_ir,np.float32)/255).unsqueeze(0).unsqueeze(0).to(device)
            rv = lyt(vt)
            yi = (0.299*vt[:,0]+0.587*vt[:,1]+0.114*vt[:,2]).unsqueeze(1)
            yl = (0.299*rv[:,0]+0.587*rv[:,1]+0.114*rv[:,2]).unsqueeze(1)
            with torch.no_grad():
                Tf, Cf, _, Tb, _ = net(yi, yl, it)

            scale = lambda x: (np.clip(x,0,1)*255).astype(np.uint8)
            import cv2
            cv2.imwrite(f'{opt.ckpt_dir}/calib_v2_maps/Tcalib_e{epoch+1:03d}.png', scale(Tf[0,0].cpu().numpy()))
            ymix_calib = Tf * yl + (1 - Tf) * yi
            ymix_base = Tb * yl + (1 - Tb) * yi
            diff = (ymix_calib - ymix_base).abs()[0,0].cpu().numpy()
            cv2.imwrite(f'{opt.ckpt_dir}/calib_v2_maps/Ydiff_e{epoch+1:03d}.png', scale(diff))

        if vl < best_v:
            best_v = vl
            torch.save(net.calibrator.state_dict(), f'{opt.ckpt_dir}/micg_best.pkl')

        torch.save(net.calibrator.state_dict(), f'{opt.ckpt_dir}/micg_last.pkl')

    print(f'Done. Best val={best_v:.4f}')


if __name__ == '__main__':
    main()
