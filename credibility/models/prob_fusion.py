"""
Probabilistic Credibility Fusion — 无学习版本 (严格按文档实现)
==============================================================
基于 Poisson–Gaussian SNR、结构张量、逆方差权重

输出:
  T_lyt     — LYT 纹理最终权重 ∈ [0,1]
  W_ir      — IR 结构最终权重 ∈ [0,1]
  Q_in      — 原图Y纹理质量 ∈ [0,1]
  Q_lyt     — LYT Y纹理质量 ∈ [0,1]
  V_in      — 原图纹理质量方差
  V_lyt     — LYT纹理质量方差
  C_tex_total — 总纹理可信度 ∈ [0,1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════
# 基础算子
# ══════════════════════════════════════════════════

def local_mean(x, k=7):
    return F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)


def local_var(x, k=7):
    mu = local_mean(x, k)
    return local_mean(x * x, k) - mu * mu


def local_cov(x, y, k=7):
    mu_x = local_mean(x, k)
    mu_y = local_mean(y, k)
    return local_mean(x * y, k) - mu_x * mu_y


# ══════════════════════════════════════════════════
# Scharr + Laplacian (固定滤波器)
# ══════════════════════════════════════════════════

def scharr(x):
    """Scharr X, Y → 结构响应"""
    kx = torch.tensor([[3,10,3],[0,0,0],[-3,-10,-3]], device=x.device, dtype=x.dtype).view(1,1,3,3) / 32.0
    ky = torch.tensor([[3,0,-3],[10,0,-10],[3,0,-3]], device=x.device, dtype=x.dtype).view(1,1,3,3) / 32.0
    gx = F.conv2d(F.pad(x, (1,1,1,1), 'replicate'), kx)
    gy = F.conv2d(F.pad(x, (1,1,1,1), 'replicate'), ky)
    return gx, gy


def laplacian(x):
    k = torch.tensor([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    return F.conv2d(F.pad(x, (1,1,1,1), 'replicate'), k)


def filter_norm_sq(kernel):
    return (kernel ** 2).sum()


# 预计算常用滤波器的 L2 范数 (固定, 无需每次重建)
_LAPLACIAN_KERNEL = torch.tensor([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=torch.float32).view(1,1,3,3)
_LAP_NORM_SQ = float((_LAPLACIAN_KERNEL ** 2).sum())

_SCHARR_KERNEL_3x3 = torch.tensor([[3,10,3],[0,0,0],[-3,-10,-3]], dtype=torch.float32).view(1,1,3,3) / 32.0
_SCHARR_NORM_SQ = float((_SCHARR_KERNEL_3x3 ** 2).sum())


# ══════════════════════════════════════════════════
# 4. 线性化
# ══════════════════════════════════════════════════

def linearize(y, gamma=2.2, eps=1e-6):
    return (y.clamp(0, 1) + eps).pow(gamma)


# ══════════════════════════════════════════════════
# 5. Poisson–Gaussian 噪声方差
# ══════════════════════════════════════════════════

def pg_noise_variance(x, a=0.005, b=1e-4, k=7):
    mu = local_mean(x, k)
    return a * mu + b


# ══════════════════════════════════════════════════
# 6. 滤波响应 SNR
# ══════════════════════════════════════════════════

def texture_snr(x, a=0.005, b=1e-4, k=7, rho=3, tau=1.0, T=0.5):
    """纹理 SNR → 映射到 [0,1]"""
    V_n = pg_noise_variance(x, a, b, k)
    L = laplacian(x)
    V_L = V_n * _LAP_NORM_SQ
    E_obs = local_mean(L * L, 2 * rho + 1)
    E_true = F.relu(E_obs - V_L)
    snr = E_true / (V_L + 1e-6)
    return torch.sigmoid((torch.log1p(snr) - tau) / T)


def edge_snr(x, a=0.005, b=1e-4, k=7):
    """边缘 SNR → 结构响应显著性"""
    V_n = pg_noise_variance(x, a, b, k)
    gx, gy = scharr(x)
    R = torch.sqrt(gx * gx + gy * gy + 1e-6)
    V_scharr = V_n * _SCHARR_NORM_SQ
    snr = R / (torch.sqrt(V_scharr) + 1e-6)
    return torch.sigmoid(torch.log1p(snr) - 1.0)


# ══════════════════════════════════════════════════
# 7. LYT 噪声传播 + 伪影方差
# ══════════════════════════════════════════════════

def lyt_noise_propagate(y_in, y_lyt, a=0.005, b=1e-4, k=7, rho=3, Gmax=8.0):
    """LYT 局部增益、传播噪声、伪影方差"""
    cov = local_cov(y_in, y_lyt, k)
    var_in = local_var(y_in, k)
    G = (cov / (var_in + 1e-6)).clamp(0, Gmax)
    B = local_mean(y_lyt, k) - G * local_mean(y_in, k)
    # 传播噪声
    V_in = pg_noise_variance(y_in, a, b, k)
    V_prop = G * G * V_in
    # 伪影残差
    r_art = y_lyt - (G * y_in + B)
    V_art = local_mean(r_art * r_art, 2 * rho + 1)
    V_lyt = V_prop + V_art
    E_obs_lyt = local_mean(laplacian(y_lyt) ** 2, 2 * rho + 1)
    return V_lyt, V_art, E_obs_lyt, G


# ══════════════════════════════════════════════════
# 8. 结构张量 + 方向一致性
# ══════════════════════════════════════════════════

def structure_tensor(x, rho=5):
    """返回 S(结构强度), K(方向一致性), M_S(结构门控)"""
    gx, gy = scharr(x)
    A = local_mean(gx * gx, 2 * rho + 1)
    B = local_mean(gx * gy, 2 * rho + 1)
    C = local_mean(gy * gy, 2 * rho + 1)
    disc = torch.sqrt((A - C) ** 2 + 4 * B * B + 1e-6)
    lam1 = (A + C + disc) / 2
    lam2 = (A + C - disc) / 2
    S = lam1 + lam2
    K = torch.clamp((lam1 - lam2) / (S + 1e-4), 0.0, 1.0)
    V_g = pg_noise_variance(x, 0.005, 1e-4, 2 * rho + 1)
    M_S = torch.sigmoid((torch.log1p(S / (V_g + 1e-6)) - 1.0) / 0.5)
    return S, K, M_S


# ══════════════════════════════════════════════════
# 10-11. 物理质量均值 + 方差
# ══════════════════════════════════════════════════

def quality_tex_original(y_in, a=0.005, b=1e-4, alpha_k=0.3):
    """原图Y纹理质量 Q_in + V_in"""
    C_snr = texture_snr(y_in, a, b)
    _, K, _ = structure_tensor(y_in)
    C_dir = K
    Q = C_snr * (alpha_k + (1 - alpha_k) * C_dir)

    V_L = pg_noise_variance(y_in, a, b, 7) * _LAP_NORM_SQ
    E_obs = local_mean(laplacian(y_in) ** 2, 7)
    E_true = F.relu(E_obs - V_L)
    alpha_v, beta_v, vmin = 0.5, 0.5, 1e-3
    V = (alpha_v * V_L / (E_true + 1e-6) + beta_v * (1 - C_dir) + vmin).clamp(1e-3, 100)
    return Q, V


def quality_tex_lyt(y_in, y_lyt, a=0.005, b=1e-4, alpha_k=0.3, gamma_a=2.0):
    """LYT纹理质量 Q_lyt + V_lyt"""
    C_snr = texture_snr(y_lyt, a, b)
    _, K, _ = structure_tensor(y_lyt)
    C_dir = K
    V_lyt_total, V_art, E_obs_lyt, _ = lyt_noise_propagate(y_in, y_lyt, a, b)
    P_art = torch.exp(-gamma_a * V_art / (E_obs_lyt + 1e-6))
    Q = C_snr * (alpha_k + (1 - alpha_k) * C_dir) * P_art

    V_L = pg_noise_variance(y_lyt, a, b, 7) * _LAP_NORM_SQ
    E_obs = local_mean(laplacian(y_lyt) ** 2, 7)
    E_true = F.relu(E_obs - V_L)
    alpha_v, beta_v, gamma_v, vmin = 0.5, 0.5, 1.0, 1e-3
    V = (alpha_v * V_L / (E_true + 1e-6) + beta_v * (1 - C_dir)
         + gamma_v * V_art / (E_obs_lyt + 1e-6) + vmin).clamp(1e-3, 100)
    return Q, V


def quality_struct_ir(ir, a=0.005, b=1e-4):
    """IR 结构质量"""
    C_snr = edge_snr(ir, a, b)
    _, K, _ = structure_tensor(ir)
    Q = C_snr * K
    S, _, _ = structure_tensor(ir)
    V_g = pg_noise_variance(ir, a, b, 7) * _SCHARR_NORM_SQ
    alpha_v, beta_v, vmin = 0.5, 0.5, 1e-3
    V = (alpha_v * V_g / (S + 1e-6) + beta_v * (1 - K) + vmin).clamp(1e-3, 100)
    return Q, V


# ══════════════════════════════════════════════════
# 12. 逆方差融合
# ══════════════════════════════════════════════════

def inverse_variance_blend(Q_in, V_in, Q_lyt, V_lyt):
    """LYT 纹理权重 + 总可信度"""
    Pi_in = Q_in / (V_in + 1e-6)
    Pi_lyt = Q_lyt / (V_lyt + 1e-6)
    T_lyt = Pi_lyt / (Pi_in + Pi_lyt + 1e-6)
    C_tex_total = 1 - torch.exp(-Pi_in - Pi_lyt)
    return T_lyt, C_tex_total


# ══════════════════════════════════════════════════
# 主模块
# ══════════════════════════════════════════════════

class ProbabilisticCredibilityFusion(nn.Module):
    """无学习概率可信度融合"""

    def __init__(self, a=0.005, b=1e-4, gamma_a=2.0):
        super().__init__()
        self.a = a
        self.b = b
        self.gamma_a = gamma_a

    def forward(self, y_in, y_lyt, ir):
        """
        y_in, y_lyt, ir: (B,1,H,W) ∈ [0,1]
        Returns:
            T_lyt        — LYT纹理权重 [0,1]
            C_tex_total  — 纹理总可信度 [0,1]
            Q_in, V_in   — 原图纹理质量+方差
            Q_lyt, V_lyt — LYT纹理质量+方差
        """
        # 质量 (直接用 [0,1] 范围, 跳过线性化—MSRS是PNG已做gamma)
        Q_in, V_in = quality_tex_original(y_in, self.a, self.b)
        Q_lyt, V_lyt = quality_tex_lyt(y_in, y_lyt, self.a, self.b, gamma_a=self.gamma_a)

        # 逆方差融合
        T_lyt, C_tex_total = inverse_variance_blend(Q_in, V_in, Q_lyt, V_lyt)

        return T_lyt, C_tex_total, Q_in, V_in, Q_lyt, V_lyt
