import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List, Tuple
import copy
import logging

from models.model import MultiSeqTTANet
from models.reconstruction import BaselineErrorRegistry
from losses.losses import tta_loss
from .sre import SequenceReliabilityEstimator

logger = logging.getLogger(__name__)


class TTAEngine:
    """
    测试时自适应引擎。

    实现流程：
    1. 冻结除融合层Q/K/V外的所有参数
    2. 对每个batch：
       a. 提取特征（无梯度）
       b. SRE：估计序列可靠性和锚点
       c. 熵筛选：高熵样本跳过更新
       d. 计算TTA损失（L_entropy + L_DSIS + L_hier）
       e. 更新融合层参数
    """

    def __init__(self,
                 model: MultiSeqTTANet,
                 baseline_registry: BaselineErrorRegistry,
                 tta_config,
                 device: torch.device,
                 sequences: List[str] = None):
        self.model = model
        self.device = device
        self.config = tta_config
        self.sequences = sequences or ['T1', 'T1ce', 'T2', 'FLAIR']

        # 保存初始权重，用于重置
        self.init_state = copy.deepcopy(model.state_dict())

        # SRE
        self.sre = SequenceReliabilityEstimator(
            baseline_registry, sequences
        )

        # 设置TTA优化器
        self.optimizer = model.get_tta_optimizer(lr=tta_config.lr)

    def reset(self):
        """重置模型到训练完毕的初始状态（多次实验时使用）"""
        self.model.load_state_dict(self.init_state)
        self.optimizer = self.model.get_tta_optimizer(lr=self.config.lr)
        logger.info("TTA model reset to initial checkpoint state.")

    @torch.no_grad()
    def _extract_features(self, x: torch.Tensor):
        """无梯度地提取特征（SRE和筛选阶段）"""
        F_bot, skips, z_dict, z_norm = self.model.encode_and_project(x)

        # 用当前融合层（无梯度）得到多模态预测，用于熵筛选
        z_fused = self.model.fusion(z_dict, r_dict=None)
        logits = self.model.decode_from_fused(z_fused, skips, x.shape[2:])
        p_multi = F.softmax(logits, dim=1)

        return F_bot, skips, z_dict, z_norm, p_multi

    def adapt_and_predict(self, x: torch.Tensor) -> Dict:
        """
        对单个batch执行TTA并返回预测。

        Args:
            x: [B, 4, H, W, D]
        Returns:
            {
                'pred':    [B, H, W, D]  最终分割预测（argmax）
                'prob':    [B, C, H, W, D]  概率图
                'r_dict':  {seq: float}  可靠性分数
                'anchor':  str  锚点序列
                'skipped': bool  是否跳过了参数更新
                'entropy': float  多模态熵值
            }
        """
        x = x.to(self.device)
        target_size = x.shape[2:]

        # ── Step 1：无梯度特征提取 ────────────────────────────────
        F_bot, skips, z_dict, z_norm, p_multi_init = self._extract_features(x)

        # ── Step 2：SRE — 动态可靠性估计 ─────────────────────────
        r_dict, anchor = self.sre.estimate(z_norm, self.model.rec_nets)

        # ── Step 3：熵筛选 ────────────────────────────────────────
        skip_update, H_val = self.sre.entropy_filter(
            p_multi_init, self.model.num_classes,
            ratio=self.config.entropy_threshold_ratio
        )

        if skip_update:
            pred = p_multi_init.argmax(dim=1)
            return {
                'pred': pred, 'prob': p_multi_init,
                'r_dict': r_dict, 'anchor': anchor,
                'skipped': True, 'entropy': H_val,
            }

        # ── Step 4：带梯度的前向（TTA更新） ──────────────────────
        self.optimizer.zero_grad()

        # 融合（RAF：传入可靠性矩阵）
        z_fused = self.model.fusion(z_dict, r_dict)
        logits = self.model.decode_from_fused(z_fused, skips, target_size, r_dict=r_dict)
        p_multi = F.softmax(logits, dim=1)

        # 单模态预测（冻结头，无梯度）
        with torch.no_grad():
            uni_logits = self.model.uni_heads(z_dict, target_size)
        p_uni = {s: F.softmax(uni_logits[s], dim=1).detach()
                 for s in self.sequences}

        # ── Step 5：TTA损失 ───────────────────────────────────────
        losses = tta_loss(
            p_multi, p_uni, r_dict, anchor,
            lambda_dsis=self.config.lambda_dsis,
            lambda_hier=self.config.lambda_hier,
        )

        # ── Step 6：梯度裁剪 + 更新 ──────────────────────────────
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            max_norm=self.config.max_grad_norm,
        )

        # 异常梯度检测（防止在异常样本上过拟合）
        grad_norm = self._compute_grad_norm()
        if self._is_abnormal_gradient(grad_norm):
            self.optimizer.zero_grad()
            logger.warning(
                f"Abnormal gradient norm {grad_norm:.4f}, skipping update."
            )
            with torch.no_grad():
                pred = p_multi.argmax(dim=1)
            return {
                'pred': pred, 'prob': p_multi.detach(),
                'r_dict': r_dict, 'anchor': anchor,
                'skipped': True, 'entropy': H_val,
                'losses': {k: v.item() for k, v in losses.items()},
            }

        self.optimizer.step()

        with torch.no_grad():
            pred = p_multi.argmax(dim=1)

        return {
            'pred': pred,
            'prob': p_multi.detach(),
            'r_dict': r_dict,
            'anchor': anchor,
            'skipped': False,
            'entropy': H_val,
            'losses': {k: v.item() for k, v in losses.items()},
        }

    def _compute_grad_norm(self) -> float:
        total_norm = 0.0
        for p in self.model.parameters():
            if p.requires_grad and p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def _is_abnormal_gradient(self, grad_norm: float) -> bool:
        """
        基于历史中位数的异常梯度检测。
        若当前梯度范数超过历史中位数的10倍，视为异常。
        """
        if not hasattr(self, '_grad_history'):
            self._grad_history = []
        self._grad_history.append(grad_norm)

        if len(self._grad_history) < 5:
            return False  # 历史太少，不判断

        median = float(np.median(self._grad_history))
        return grad_norm > 10 * median

    def run(self, dataloader, desc: str = "TTA") -> List[Dict]:
        """
        对整个测试集执行TTA，返回所有batch的结果。

        Args:
            dataloader: 返回 (x, meta) 或只返回 x 的DataLoader
        Returns:
            results: list of dicts，每个元素对应一个batch
        """
        self.model.train()  # 保持BN在eval模式但允许梯度
        self.model.encoder.eval()
        self.model.seq_projs.eval()
        self.model.decoder.eval()
        self.model.uni_heads.eval()
        self.model.rec_nets.eval()
        # 只有fusion层保持train模式

        results = []
        anchor_counts = {s: 0 for s in self.sequences}
        skip_count = 0

        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)):
                x = batch[0]
                meta = batch[1] if len(batch) > 1 else {}
            else:
                x, meta = batch, {}

            result = self.adapt_and_predict(x)
            result['meta'] = meta
            results.append(result)

            # 统计
            anchor_counts[result['anchor']] += 1
            if result['skipped']:
                skip_count += 1

            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"[{desc}] {batch_idx+1}/{len(dataloader)} | "
                    f"anchor={result['anchor']} | "
                    f"H={result['entropy']:.3f} | "
                    f"skipped={result['skipped']}"
                )

        # 汇总统计
        total = len(results)
        logger.info(f"\n[{desc}] Anchor statistics:")
        for s, cnt in anchor_counts.items():
            logger.info(f"  {s}: {cnt}/{total} ({100*cnt/total:.1f}%)")
        logger.info(f"  Skipped: {skip_count}/{total} "
                    f"({100*skip_count/total:.1f}%)")

        return results
