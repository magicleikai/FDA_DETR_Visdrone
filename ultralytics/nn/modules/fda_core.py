import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np

__all__ = ['WaveletFPN_Downsample', 'gaussian_nwd', 'sinkhorn_knopp_match']


class HighFrequencyAttention(nn.Module):
    """
    轻量级通道注意力机制，用于放大高频分量的特征能量 。
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        # 确保 reduction 不会大于 channels
        mid_channels = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class WaveletFPN_Downsample(nn.Module):
    """
    基于离散小波变换(DWT)的频域解耦特征金字塔下采样层。
    适配 Ultralytics 原生 c1(输入), c2(输出) 传参规范。
    """

    def __init__(self, c1, c2):
        super().__init__()
        self.in_channels = c1

        # 1. 提取 Haar 小波滤波器系数
        wavelet = pywt.Wavelet('haar')
        dec_hi = torch.tensor(wavelet.dec_hi[::-1], dtype=torch.float32)
        dec_lo = torch.tensor(wavelet.dec_lo[::-1], dtype=torch.float32)

        # 2. 构造 2D 滤波器矩阵 (LL, LH, HL, HH)
        filter_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        filter_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        filter_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        filter_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)

        # 3. 堆叠为 [4, 1, 2, 2] 的卷积核，并适配输入通道 c1
        filters = torch.stack([filter_ll, filter_lh, filter_hl, filter_hh], dim=0)
        filters = filters.unsqueeze(1).repeat(c1, 1, 1, 1)

        # 4. 注册为 Buffer，冻结权重
        self.register_buffer('wavelet_filters', filters)

        # 5. 高频分量通道注意力
        self.high_freq_attn = HighFrequencyAttention(c1 * 3)

        # 6. 维度对齐层 (强行映射到配置表中要求的输出通道 c2)
        self.channel_align = nn.Conv2d(c1 * 4, c2, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        # x: [B, C, H, W]
        out = F.conv2d(x, self.wavelet_filters, stride=2, padding=0, groups=self.in_channels)

        B, C4, H_out, W_out = out.shape
        C = C4 // 4
        out = out.view(B, C, 4, H_out, W_out)

        LL = out[:, :, 0, :, :]
        high_freqs = out[:, :, 1:, :, :].contiguous().view(B, C * 3, H_out, W_out)

        high_freqs_enhanced = self.high_freq_attn(high_freqs)
        fused_features = torch.cat([LL, high_freqs_enhanced], dim=1)

        return self.act(self.bn(self.channel_align(fused_features)))


# ==========================================
# 匹配层优化：NWD 与 Sinkhorn
# ==========================================

def gaussian_nwd(pred_bboxes, gt_bboxes, pairwise=True):
    """
    针对 VisDrone 归一化目标的平滑自适应 NWD 计算。
    """
    cx1, cy1, w1, h1 = pred_bboxes.unbind(-1)
    cx2, cy2, w2, h2 = gt_bboxes.unbind(-1)

    if pairwise:
        wasserstein_sq = (cx1.unsqueeze(1) - cx2.unsqueeze(0)) ** 2 + \
                         (cy1.unsqueeze(1) - cy2.unsqueeze(0)) ** 2 + \
                         ((w1.unsqueeze(1) - w2.unsqueeze(0)) ** 2 + \
                          (h1.unsqueeze(1) - h2.unsqueeze(0)) ** 2) / 4.0

        gt_area = (w2 * h2).unsqueeze(0)
    else:
        wasserstein_sq = (cx1 - cx2) ** 2 + \
                         (cy1 - cy2) ** 2 + \
                         ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4.0

        gt_area = w2 * h2

    # 🚀 绝杀平滑：分段式自适应常数！
    # 相比于直接乘以 5.0，使用平方根可以极大地平滑常数的波动范围
    # 面积在 0.0001 (极小) 时，常数为 sqrt(0.0001) = 0.01 (不会过于严苛)
    # 面积在 0.04 (较大) 时，常数为 sqrt(0.04) = 0.2 (相当于大常数)
    # 最后再加上 0.02 的底座，保证绝对不会出现 0 分黑洞
    dynamic_constant = torch.sqrt(torch.clamp(gt_area, min=1e-5)) + 0.02

    nwd = torch.exp(-wasserstein_sq / dynamic_constant)
    return nwd

def sinkhorn_knopp_match(cost_matrix, high_freq_energy=None, epsilon=0.05, iterations=3):
    """
    高频能量加权的最优传输匹配 (Freq-Sinkhorn) 。
    利用熵正则化求解器生成“软性”匹配概率矩阵，缓解硬匹配的震荡 。

    cost_matrix: [N_queries, M_gt] 代价矩阵
    high_freq_energy: [M_gt] 真实目标框区域的高频能量先验
    """
    N, M = cost_matrix.shape
    if N == 0 or M == 0:
        return torch.empty((N, M), device=cost_matrix.device)

    # 如果提供了高频能量，则增加对微小/难检 GT 的先验权重偏好
    if high_freq_energy is not None:
        # 高频能量越高，v_prior 越大，强制 Sinkhorn 分配更多资源
        v_prior = 1.0 + high_freq_energy
    else:
        v_prior = torch.ones(M, device=cost_matrix.device)

    v_prior = v_prior / v_prior.sum()  # 归一化分布
    u_prior = torch.ones(N, device=cost_matrix.device) / N

    # 熵正则化代价转移矩阵
    K = torch.exp(-cost_matrix / epsilon)

    u = u_prior
    v = v_prior

    # 快速矩阵迭代求解
    for _ in range(iterations):
        u = u_prior / (K @ v + 1e-8)
        v = v_prior / (K.t() @ u + 1e-8)

    optimal_plan = u.unsqueeze(1) * K * v.unsqueeze(0)
    return optimal_plan