import torch
import torch.nn as nn
from typing import Dict, List, Tuple


class BottleneckMLP(nn.Module):
    """
    轻量Bottleneck MLP，用1×1×1卷积实现。
    结构：in_ch → hidden_ch → hidden_ch → out_ch
    参数量：in_ch*hidden_ch + hidden_ch^2 + hidden_ch*out_ch
    """
    def __init__(self, in_ch: int, out_ch: int, hidden_ch: int = None):
        super().__init__()
        if hidden_ch is None:
            hidden_ch = max(in_ch, out_ch) * 2
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, hidden_ch, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_ch, hidden_ch, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_ch, out_ch, kernel_size=1, bias=False),
        )

    def forward(self, x):
        return self.net(x)


# 重建对定义：(src, tgt)
# 注意 T2F 表示 concat(t2w, t2f) → t1c 的方向已拆解为独立对
RECONSTRUCTION_PAIRS: List[Tuple[str, str]] = [
    ('t1c', 't2w'),
    ('t2w',   't1c'),
    ('t1c', 't1n'),
    ('t1n',   't1c'),
    ('t2w',   't2f'),
    ('t2f','t2w'),
]


class BidirectionalRecNets(nn.Module):
    """
    6对双向重建网络，在归一化特征空间（LayerNorm后）操作。
    训练完成后冻结，测试时只做前向传播。
    """
    def __init__(self, seq_ch: int, sequences: List[str] = None):
        super().__init__()
        if sequences is None:
            sequences = ['t1n', 't1c', 't2w', 't2f']
        self.sequences = sequences
        self.seq_ch = seq_ch
        hidden_ch = seq_ch * 2  # C/4 → C/2 → C/2 → C/4

        self.nets = nn.ModuleDict()
        for src, tgt in RECONSTRUCTION_PAIRS:
            key = f'{src}_to_{tgt}'
            self.nets[key] = BottleneckMLP(seq_ch, seq_ch, hidden_ch)

        # LayerNorm：对每个序列的特征独立归一化（沿通道维度）
        # 在forward中对各序列特征做归一化，保证跨序列误差可比
        self.layer_norms = nn.ModuleDict({
            s: nn.GroupNorm(1, seq_ch)  # GroupNorm(1,C) ≡ LayerNorm on spatial
            for s in sequences
        })

    def normalize(self, z_dict: Dict[str, torch.Tensor]
                  ) -> Dict[str, torch.Tensor]:
        """对各序列特征做归一化，返回归一化后的特征字典"""
        return {s: self.layer_norms[s](z_dict[s]) for s in self.sequences}

    def reconstruct(self, z_norm_dict: Dict[str, torch.Tensor],
                    src: str, tgt: str) -> torch.Tensor:
        """单对重建"""
        key = f'{src}_to_{tgt}'
        return self.nets[key](z_norm_dict[src])

    def compute_all_errors(
        self,
        z_norm_dict: Dict[str, torch.Tensor],
    ) -> Dict[Tuple[str, str], torch.Tensor]:
        """
        计算所有重建对的L2误差（标量）。
        返回：{(src, tgt): scalar_tensor}
        """
        errors = {}
        for src, tgt in RECONSTRUCTION_PAIRS:
            z_hat = self.reconstruct(z_norm_dict, src, tgt)
            err = (z_norm_dict[tgt] - z_hat).pow(2).mean()
            errors[(src, tgt)] = err
        return errors

    def forward(self, z_dict: Dict[str, torch.Tensor]):
        """
        训练时调用：返回归一化特征和所有重建对误差。
        """
        z_norm = self.normalize(z_dict)
        errors = self.compute_all_errors(z_norm)
        return z_norm, errors


class BaselineErrorRegistry:
    """
    保存源域验证集上每个重建对的基准误差 μ_{i→j}。
    训练结束后在验证集上统计，测试时用于计算相对误差。
    """
    def __init__(self):
        self.mu: Dict[Tuple[str, str], float] = {}

    def update(self, errors: Dict[Tuple[str, str], float]):
        """累积误差（用于在验证集上averaging）"""
        for k, v in errors.items():
            if k not in self.mu:
                self.mu[k] = []
            self.mu[k].append(v)

    def finalize(self):
        """将列表转为均值"""
        self.mu = {k: sum(v) / len(v) for k, v in self.mu.items()}

    def relative_errors(
        self,
        errors: Dict[Tuple[str, str], torch.Tensor],
        eps: float = 1e-8
    ) -> Dict[Tuple[str, str], torch.Tensor]:
        """
        计算相对误差：ε_{i→j} = err_{i→j} / μ_{i→j}
        使目标域误差以源域基准为单位，不同重建对量纲统一。
        """
        rel = {}
        for k, err in errors.items():
            mu = self.mu.get(k, 1.0)
            rel[k] = err / (mu + eps)
        return rel

    def save(self, path: str):
        import json
        # 键格式与网络字典保持一致：'T1ce_to_T2'
        serializable = {f'{k[0]}_to_{k[1]}': v for k, v in self.mu.items()}
        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2)

    def load(self, path: str):
        import json
        with open(path) as f:
            data = json.load(f)
        self.mu = {}
        for src, tgt in RECONSTRUCTION_PAIRS:
            key = f'{src}_to_{tgt}'
            if key in data:
                self.mu[(src, tgt)] = data[key]