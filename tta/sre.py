import torch
from typing import Dict, Tuple
from models.reconstruction import RECONSTRUCTION_PAIRS, BaselineErrorRegistry


class SequenceReliabilityEstimator:
    """
    动态序列可靠性估计（SRE）。

    基于双向相对重建误差估计各序列可靠性：
    1. 计算所有重建对的相对L2误差（相对于源域基准μ_{i→j}）
    2. 以序列为节点双向累加误差 E_s
    3. r_s^raw = 1/(1+E_s)，温度归一化得到 r_s
    4. s* = argmax r_s
    """

    def __init__(self, baseline_registry: BaselineErrorRegistry,
                 sequences=None, eps: float = 1e-6):
        if sequences is None:
            sequences = ['T1', 'T1ce', 'T2', 'FLAIR']
        self.sequences = sequences
        self.baseline = baseline_registry
        self.eps = eps

    @torch.no_grad()
    def estimate(
        self,
        z_norm_dict: Dict[str, torch.Tensor],
        rec_nets,
    ) -> Tuple[Dict[str, float], str]:
        """
        Args:
            z_norm_dict: {seq: [B, C/4, h, w, d]}，LayerNorm后的特征
            rec_nets:    BidirectionalRecNets 实例

        Returns:
            r_dict: {seq: float}，归一化可靠性分数，和为1
            anchor: str，当前batch的锚点序列
        """
        # Step 1：计算所有重建对的绝对误差
        raw_errors = rec_nets.compute_all_errors(z_norm_dict)

        # Step 2：转为相对误差（除以源域基准μ）
        rel_errors = self.baseline.relative_errors(raw_errors)

        # Step 3：以序列为节点双向累加
        E = {s: 0.0 for s in self.sequences}
        for (src, tgt), err in rel_errors.items():
            val = err.item() if torch.is_tensor(err) else err
            E[src] += val
            E[tgt] += val

        # Step 4：原始可靠性分数
        r_raw = {s: 1.0 / (1.0 + E[s]) for s in self.sequences}

        # Step 5：温度归一化
        r_vals = torch.tensor([r_raw[s] for s in self.sequences],
                               dtype=torch.float32)
        tau = r_vals.std().item() + self.eps
        r_softmax = torch.softmax(r_vals / tau, dim=0)
        r_dict = {s: r_softmax[i].item()
                  for i, s in enumerate(self.sequences)}

        # Step 6：动态锚点
        anchor = max(r_dict, key=r_dict.get)

        return r_dict, anchor

    def entropy_filter(self, p_multi: torch.Tensor,
                       num_classes: int,
                       ratio: float = 0.4) -> bool:
        """
        样本筛选：H_multi(x) > γ_m 则跳过（返回True表示跳过）。
        γ_m = ratio * ln(C)
        """
        import math
        gamma_m = ratio * math.log(num_classes)
        H = -(p_multi * torch.log(p_multi.clamp(min=1e-8))).sum(dim=1).mean()
        return H.item() > gamma_m, H.item()
