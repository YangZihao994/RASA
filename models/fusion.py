import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class CrossSeqAttentionLayer(nn.Module):
    """
    单层跨序列多头自注意力。
    TTA期间更新 W_Q, W_K, W_V 及缩放因子 beta。
    """
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.W_Q = nn.Linear(dim, dim, bias=False)
        self.W_K = nn.Linear(dim, dim, bias=False)
        self.W_V = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

        # RAF缩放因子 β，初始化为1.0，TTA时随Q/K/V一同更新
        self.beta = nn.Parameter(torch.ones(1))

        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor,
                R: Optional[torch.Tensor] = None,
                eps: float = 1e-8) -> torch.Tensor:
        """
        Args:
            tokens: [B, S*N, dim]
            R:      [S*N, S*N] 可靠性矩阵（可选，TTA时传入）
            eps:    log(R+eps) 的数值稳定项
        Returns:
            tokens: [B, S*N, dim]（残差连接后）
        """
        B, L, D = tokens.shape
        H = self.num_heads
        Dh = self.head_dim

        residual = tokens

        Q = self.W_Q(tokens).reshape(B, L, H, Dh).transpose(1, 2)  # [B,H,L,Dh]
        K = self.W_K(tokens).reshape(B, L, H, Dh).transpose(1, 2)
        V = self.W_V(tokens).reshape(B, L, H, Dh).transpose(1, 2)

        # 基础注意力分数
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B,H,L,L]

        # RAF：叠加可靠性偏置 β·log(R+ε)
        if R is not None:
            # R: [L, L] → [1, 1, L, L]
            log_R = torch.log(R.clamp(min=eps)).unsqueeze(0).unsqueeze(0)
            attn = attn + self.beta * log_R

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)  # [B,H,L,Dh]
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.proj(out)

        # 残差 + LayerNorm
        return self.norm(out + residual)


class CrossSeqTransformer(nn.Module):
    """
    跨序列Transformer融合层。
    将4个序列的特征token化后做跨序列自注意力，最终融合为统一特征。

    TTA期间仅更新：W_Q, W_K, W_V（在每个CrossSeqAttentionLayer中）和 beta。
    """
    def __init__(self, seq_ch: int, num_heads: int = 8,
                 num_layers: int = 2, sequences=None):
        super().__init__()
        if sequences is None:
            sequences = ['T1', 'T1ce', 'T2', 'FLAIR']
        self.sequences = sequences
        self.S = len(sequences)
        self.seq_ch = seq_ch

        # 序列位置编码（区分不同序列）
        self.seq_embed = nn.Embedding(self.S, seq_ch)

        # 注意力层堆叠
        self.layers = nn.ModuleList([
            CrossSeqAttentionLayer(seq_ch, num_heads)
            for _ in range(num_layers)
        ])

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(seq_ch, seq_ch * 4),
            nn.GELU(),
            nn.Linear(seq_ch * 4, seq_ch),
        )
        self.norm_ffn = nn.LayerNorm(seq_ch)

    def build_reliability_matrix(self, r_dict: Dict[str, float],
                                  N: int, device: torch.device) -> torch.Tensor:
        """
        构造token级可靠性矩阵 R ∈ [S*N, S*N]。
        R[i,j] = r_{s(i)} * r_{s(j)}
        """
        r_vec = torch.tensor(
            [r_dict[s] for s in self.sequences],
            dtype=torch.float32, device=device
        )  # [S]
        # 扩展到token级别
        r_token = r_vec.repeat_interleave(N)   # [S*N]
        R = torch.outer(r_token, r_token)       # [S*N, S*N]
        return R

    def forward(self, z_dict: Dict[str, torch.Tensor],
                r_dict: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """
        Args:
            z_dict: {seq: [B, C/4, h, w, d]}
            r_dict: {seq: float}，可靠性分数（TTA时传入）
        Returns:
            z_fused: [B, C, h, w, d]（C = S * seq_ch）
        """
        B = next(iter(z_dict.values())).shape[0]
        device = next(iter(z_dict.values())).device
        spatial_shape = next(iter(z_dict.values())).shape[2:]
        N = 1
        for s in spatial_shape:
            N *= s

        # 展平并加序列位置编码
        tokens_list = []
        for i, s in enumerate(self.sequences):
            z_flat = z_dict[s].flatten(2).transpose(1, 2)  # [B, N, C/4]
            z_flat = z_flat + self.seq_embed.weight[i]      # 广播加位置编码
            tokens_list.append(z_flat)
        tokens = torch.cat(tokens_list, dim=1)              # [B, S*N, C/4]

        # 可靠性矩阵（TTA时传入）
        R = None
        if r_dict is not None:
            R = self.build_reliability_matrix(r_dict, N, device)

        # 跨序列注意力
        for layer in self.layers:
            tokens = layer(tokens, R)

        # FFN
        tokens = self.norm_ffn(self.ffn(tokens) + tokens)

        # 重组为空间特征并拼接
        SN = self.S * N
        tokens = tokens.reshape(B, self.S, N, self.seq_ch)
        z_list = []
        for i in range(self.S):
            z_i = tokens[:, i]                             # [B, N, C/4]
            z_i = z_i.transpose(1, 2).reshape(B, self.seq_ch, *spatial_shape)
            z_list.append(z_i)
        z_fused = torch.cat(z_list, dim=1)                 # [B, C, h, w, d]

        return z_fused

    def tta_parameters(self):
        """返回TTA期间需要更新的参数（Q/K/V 和 beta）"""
        params = []
        for layer in self.layers:
            params += list(layer.W_Q.parameters())
            params += list(layer.W_K.parameters())
            params += list(layer.W_V.parameters())
            params.append(layer.beta)
        return params
