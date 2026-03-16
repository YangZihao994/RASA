import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ─────────────────────────────────────────────
# 基础构建块
# ─────────────────────────────────────────────

class ConvBnReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBnReLU(ch, ch),
            ConvBnReLU(ch, ch),
        )
        self.skip = nn.Identity()

    def forward(self, x):
        return self.block(x) + self.skip(x)


class DownBlock(nn.Module):
    """下采样块：stride=2卷积 + ResBlock"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = ConvBnReLU(in_ch, out_ch, kernel=3, stride=2, padding=1)
        self.res = ResBlock(out_ch)

    def forward(self, x):
        return self.res(self.down(x))


class UpBlock(nn.Module):
    """上采样块：转置卷积 + skip connection + ResBlock"""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            ConvBnReLU(out_ch + skip_ch, out_ch),
            ResBlock(out_ch),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # 处理尺寸不整除的情况
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear',
                              align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ─────────────────────────────────────────────
# 编码器
# ─────────────────────────────────────────────

class Encoder3D(nn.Module):
    """
    共享3D UNet编码器。
    输入：[B, 4, H, W, D]（4序列concat）
    输出：瓶颈特征 F_bot 及各层skip features
    """
    def __init__(self, in_channels: int = 4, base_ch: int = 32, depth: int = 4):
        super().__init__()
        self.depth = depth

        # stem
        self.stem = nn.Sequential(
            ConvBnReLU(in_channels, base_ch),
            ResBlock(base_ch),
        )

        # 下采样层
        self.downs = nn.ModuleList()
        ch = base_ch
        for _ in range(depth):
            self.downs.append(DownBlock(ch, ch * 2))
            ch *= 2

        self.bottleneck_ch = ch  # = base_ch * 2^depth

    def forward(self, x):
        """
        Returns:
            F_bot: [B, bottleneck_ch, h, w, d]
            skips: list of [B, C_i, H_i, W_i, D_i], index 0 = stem output
        """
        skips = []
        out = self.stem(x)
        skips.append(out)

        for down in self.downs:
            out = down(out)
            skips.append(out)

        F_bot = skips.pop()  # 最深层作为瓶颈特征，不作为skip
        return F_bot, skips


# ─────────────────────────────────────────────
# 序列分离投影层
# ─────────────────────────────────────────────

class SeqProjections(nn.Module):
    """
    将瓶颈特征分解为4个序列特定特征。
    F_bot [B, C, h, w, d] → {z_s: [B, C/4, h, w, d]}
    使用独立的1×1×1卷积，参数训练后冻结。
    """
    def __init__(self, in_ch: int, sequences: List[str]):
        super().__init__()
        assert in_ch % len(sequences) == 0
        self.seq_ch = in_ch // len(sequences)
        self.sequences = sequences
        self.projs = nn.ModuleDict({
            s: nn.Conv3d(in_ch, self.seq_ch, kernel_size=1, bias=False)
            for s in sequences
        })

    def forward(self, F_bot):
        return {s: self.projs[s](F_bot) for s in self.sequences}


# ─────────────────────────────────────────────
# 解码器
# ─────────────────────────────────────────────

class Decoder3D(nn.Module):
    """
    共享3D UNet解码器。
    输入：z_fused [B, C, h, w, d] + skips
    输出：分割logits [B, num_classes, H, W, D]
    """
    def __init__(self, bottleneck_ch: int, base_ch: int, depth: int,
                 num_classes: int):
        super().__init__()
        self.depth = depth

        self.ups = nn.ModuleList()
        ch = bottleneck_ch
        for i in range(depth):
            skip_ch = ch // 2
            out_ch = ch // 2
            self.ups.append(UpBlock(ch, skip_ch, out_ch))
            ch = out_ch

        # 最终预测头
        self.head = nn.Conv3d(ch, num_classes, kernel_size=1)

    def forward(self, z_fused, skips):
        """
        skips: list，索引0对应最浅层（stem输出），从深到浅使用
        """
        x = z_fused
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)
        return self.head(x)
