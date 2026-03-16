import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

from .unet3d import Encoder3D, Decoder3D
from .fusion import CrossSeqTransformer
from .reconstruction import BidirectionalRecNets, RECONSTRUCTION_PAIRS
from .heads import UnimodalHeads

class MultiSeqTTANet(nn.Module):
    def __init__(self, in_channels: int = 4, num_classes: int = 4,
                 base_ch: int = 32, depth: int = 4,
                 num_heads: int = 8, num_attn_layers: int = 2,
                 sequences: List[str] = None):
        super().__init__()
        if sequences is None:
            sequences = ['t1n', 't1c', 't2w', 't2f']
        self.sequences = sequences
        self.num_classes = num_classes

        # 【核心修复1】：编码器通道设为1。四个模态独立通过共享权重的编码器，实现绝对物理隔离
        self.encoder = Encoder3D(in_channels=1, base_ch=base_ch, depth=depth)
        bottleneck_ch = self.encoder.bottleneck_ch  
        seq_ch = bottleneck_ch 

        self.fusion = CrossSeqTransformer(
            seq_ch=seq_ch,
            num_heads=num_heads,
            num_layers=num_attn_layers,
            sequences=sequences,
        )

        self.decoder = Decoder3D(bottleneck_ch, base_ch, depth, num_classes)

        scale_factor = 2 ** depth
        self.uni_heads = UnimodalHeads(seq_ch, num_classes, sequences, scale_factor)
        self.rec_nets = BidirectionalRecNets(seq_ch, sequences)

        self.seq_ch = seq_ch
        self.bottleneck_ch = bottleneck_ch

    def forward(self, x: torch.Tensor,
                r_dict: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        B, S, H, W, D = x.shape
        target_size = x.shape[2:]

        # 折叠 Batch 与 Sequence 维度，独立抽取特征
        x_in = x.view(B * S, 1, H, W, D)
        F_bot_all, skips_all = self.encoder(x_in)

        # 切分特征为独立字典
        C_bot = F_bot_all.shape[1]
        F_bot_sep = F_bot_all.view(B, S, C_bot, *F_bot_all.shape[2:])
        z_dict = {seq: F_bot_sep[:, i] for i, seq in enumerate(self.sequences)}

        z_norm, rec_errors = self.rec_nets(z_dict)
        
        # 这里的 z_fused 输出的是 2048 维拼接特征
        z_fused = self.fusion(z_dict, r_dict)

        # 调用统一下游处理（压缩瓶颈特征并解码）
        logits = self.decode_from_fused(z_fused, skips_all, target_size, r_dict)
        
        uni_logits = self.uni_heads(z_dict, target_size)

        return {
            'logits': logits,
            'uni_logits': uni_logits,
            'rec_errors': rec_errors,
            'z_norm': z_norm,
            'z_dict': z_dict,
        }

    def encode_and_project(self, x: torch.Tensor):
        B, S, H, W, D = x.shape
        x_in = x.view(B * S, 1, H, W, D)
        F_bot_all, skips_all = self.encoder(x_in)
        
        C_bot = F_bot_all.shape[1]
        F_bot_sep = F_bot_all.view(B, S, C_bot, *F_bot_all.shape[2:])
        z_dict = {seq: F_bot_sep[:, i] for i, seq in enumerate(self.sequences)}
        
        z_norm = self.rec_nets.normalize(z_dict)
        return F_bot_sep, skips_all, z_dict, z_norm

    def decode_from_fused(self, z_fused: torch.Tensor, skips_all: list,
                          target_size: tuple, r_dict: Dict[str, float] = None) -> torch.Tensor:
        """统一的自适应解码方法：压缩瓶颈层特征 + 过滤 Skip Connections"""
        B = z_fused.shape[0]
        S = len(self.sequences)
        fused_skips = []
        
        # 计算可靠性权重
        if r_dict is None:
            weights = torch.ones(S, device=z_fused.device) / S
        else:
            weights = torch.tensor([r_dict[s] for s in self.sequences], device=z_fused.device)
            
        weights = weights / weights.sum()
        weights_z = weights.view(1, S, 1, 1, 1, 1)
        weights_skips = weights.view(1, S, 1, 1, 1, 1)

        # 【核心修复】：将 2048 维特征折叠为 4 个 512 维特征，并进行加权求和融合
        if z_fused.shape[1] == self.seq_ch * S:
            z_fused_sep = z_fused.view(B, S, self.seq_ch, *z_fused.shape[2:])
            z_fused_final = (z_fused_sep * weights_z).sum(dim=1)
        else:
            z_fused_final = z_fused

        # 同样对 Skip Connections 进行加权求和融合
        for skip in skips_all:
            C_skip = skip.shape[1]
            skip_sep = skip.view(B, S, C_skip, *skip.shape[2:])
            skip_fused = (skip_sep * weights_skips).sum(dim=1)
            fused_skips.append(skip_fused)

        logits = self.decoder(z_fused_final, fused_skips)
        return F.interpolate(logits, size=target_size, mode='trilinear', align_corners=False)

    def freeze_for_tta(self):
        for p in self.parameters():
            p.requires_grad_(False)
        for p in self.fusion.tta_parameters():
            p.requires_grad_(True)

    def get_tta_optimizer(self, lr: float = 1e-4):
        self.freeze_for_tta()
        tta_params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.Adam(tta_params, lr=lr)

    def reset_fusion_weights(self, checkpoint_state: Dict):
        fusion_keys = {k: v for k, v in checkpoint_state.items() if 'fusion' in k}
        self.load_state_dict(fusion_keys, strict=False)