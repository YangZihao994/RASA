import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class UnimodalHead(nn.Module):
    """
    单序列轻量分割头。
    输入：z_s [B, C/4, h, w, d]（瓶颈层分辨率）
    输出：logits [B, num_classes, H, W, D]（原始分辨率）

    使用简单的上采样+卷积，不引入额外的解码器复杂度。
    TTA时冻结，只用于计算单模态预测分布。
    """
    def __init__(self, seq_ch: int, num_classes: int, scale_factor: int = 16):
        super().__init__()
        self.scale_factor = scale_factor  # 瓶颈层相对于原图的下采样倍数

        self.conv = nn.Sequential(
            nn.Conv3d(seq_ch, seq_ch * 2, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(seq_ch * 2, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(seq_ch * 2, num_classes, kernel_size=1),
        )

    def forward(self, z_s: torch.Tensor,
                target_size: tuple = None) -> torch.Tensor:
        """
        Args:
            z_s: [B, C/4, h, w, d]
            target_size: 目标空间尺寸 (H, W, D)，用于上采样对齐
        Returns:
            logits: [B, num_classes, H, W, D]
        """
        x = self.conv(z_s)
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='trilinear',
                              align_corners=False)
        return x


class UnimodalHeads(nn.Module):
    """
    4个序列各自独立的预测头。
    训练时作为辅助任务监督，TTA时冻结。
    """
    def __init__(self, seq_ch: int, num_classes: int,
                 sequences: List[str] = None, scale_factor: int = 16):
        super().__init__()
        if sequences is None:
            sequences = ['T1', 'T1ce', 'T2', 'FLAIR']
        self.sequences = sequences
        self.heads = nn.ModuleDict({
            s: UnimodalHead(seq_ch, num_classes, scale_factor)
            for s in sequences
        })

    def forward(self, z_dict: Dict[str, torch.Tensor],
                target_size: tuple = None) -> Dict[str, torch.Tensor]:
        """
        Returns:
            {seq: logits [B, num_classes, H, W, D]}
        """
        return {
            s: self.heads[s](z_dict[s], target_size)
            for s in self.sequences
        }
