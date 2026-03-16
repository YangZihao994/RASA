"""
消融实验脚本：系统地验证各个模块的贡献。

每个配置对应消融表中的一行：
  ① 完整方法
  ② 硬编码T1ce锚点（去掉动态SRE）
  ③ 硬编码T1锚点
  ④ 绝对误差替换相对误差（去掉基准归一化）
  ⑤ 去掉LayerNorm
  ⑥ 对称KL替换有向DSIS
  ⑦ 去掉L_hier
  ⑧ 去掉RAF（固定注意力，β=0）
  ⑨ 更新BN层替换融合层

Usage:
    python scripts/ablation.py --checkpoint ./checkpoints/best_model.pth
"""

import os
import sys
import json
import logging
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import Config
from models.model import MultiSeqTTANet
from models.reconstruction import BaselineErrorRegistry, RECONSTRUCTION_PAIRS
from data.dataset import BraTSDataset
from torch.utils.data import DataLoader
from losses.losses import tta_loss, entropy_loss, hierarchy_loss
from scripts.metrics import MetricTracker

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 各消融配置的TTA step实现
# ─────────────────────────────────────────────

class AblationTTAStep:
    """
    每种消融配置对应一个step函数。
    所有step接口相同：(model, x, registry, config, sequences) → result
    """

    @staticmethod
    def _base_step(model, x, registry, config, sequences,
                   use_dynamic_anchor=True,
                   fixed_anchor=None,
                   use_relative_error=True,
                   use_layernorm=True,
                   use_directed_dsis=True,
                   use_hier=True,
                   use_raf=True,
                   update_bn=False):
        """
        通用step函数，通过flags控制消融。
        """
        device = x.device
        target_size = x.shape[2:]

        # 特征提取
        with torch.no_grad():
            F_bot, skips, z_dict, z_norm = model.encode_and_project(x)

            # LayerNorm消融
            if not use_layernorm:
                z_norm = z_dict  # 不归一化

            # 可靠性估计
            raw_errors = model.rec_nets.compute_all_errors(z_norm)

            if use_relative_error and registry and registry.mu:
                rel_errors = registry.relative_errors(raw_errors)
            else:
                rel_errors = raw_errors  # 绝对误差

            E = {s: 0.0 for s in sequences}
            for (src, tgt), err in rel_errors.items():
                val = err.item() if torch.is_tensor(err) else err
                E[src] += val
                E[tgt] += val

            r_raw = {s: 1.0 / (1.0 + E[s]) for s in sequences}
            r_vals = torch.tensor([r_raw[s] for s in sequences])
            tau = r_vals.std().item() + 1e-6
            r_softmax = torch.softmax(r_vals / tau, dim=0)
            r_dict = {s: r_softmax[i].item() for i, s in enumerate(sequences)}

            if use_dynamic_anchor:
                anchor = max(r_dict, key=r_dict.get)
            else:
                anchor = fixed_anchor

            # RAF消融：β=0 表示不使用可靠性偏置
            if not use_raf:
                for layer in model.fusion.layers:
                    layer.beta.data.zero_()

            z_fused = model.fusion(z_dict, r_dict if use_raf else None)
            logits = model.decode_from_fused(z_fused, skips, target_size)
            p_multi_check = F.softmax(logits, dim=1)

            # 熵筛选
            H = -(p_multi_check * torch.log(p_multi_check.clamp(min=1e-8))
                  ).sum(dim=1).mean().item()
            import math
            if H > 0.4 * math.log(model.num_classes):
                return {'pred': p_multi_check.argmax(dim=1),
                        'prob': p_multi_check, 'anchor': anchor,
                        'skipped': True, 'entropy': H, 'r_dict': r_dict}

        # 带梯度的更新
        optimizer = model.get_tta_optimizer(lr=config.tta.lr)
        optimizer.zero_grad()

        z_fused = model.fusion(z_dict, r_dict if use_raf else None)
        logits = model.decode_from_fused(z_fused, skips, target_size)
        p_multi = F.softmax(logits, dim=1)

        with torch.no_grad():
            uni_logits = model.uni_heads(z_dict, target_size)
        p_uni = {s: F.softmax(uni_logits[s], dim=1).detach()
                 for s in sequences}

        # 有向 vs 对称 DSIS
        if use_directed_dsis:
            from losses.losses import dsis_loss
            L_dsis = dsis_loss(p_uni, p_multi, r_dict, anchor)
        else:
            # 对称MIS（SuMi风格）
            L_dsis = _symmetric_mis(p_uni, p_multi, sequences)

        L_ent = entropy_loss(p_multi)
        L_h = hierarchy_loss(p_uni, p_multi) if use_hier else torch.tensor(0.0)
        loss = L_ent + config.tta.lambda_dsis * L_dsis + config.tta.lambda_hier * L_h
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=config.tta.max_grad_norm,
        )
        optimizer.step()

        with torch.no_grad():
            pred = p_multi.argmax(dim=1)

        return {'pred': pred, 'prob': p_multi.detach(),
                'anchor': anchor, 'skipped': False,
                'entropy': H, 'r_dict': r_dict}

    # ── 各具体配置 ─────────────────────────────────────────────

    @classmethod
    def full(cls, model, x, registry, config, sequences):
        """① 完整方法"""
        return cls._base_step(model, x, registry, config, sequences)

    @classmethod
    def fixed_T1ce(cls, model, x, registry, config, sequences):
        """② 硬编码T1ce锚点"""
        return cls._base_step(model, x, registry, config, sequences,
                               use_dynamic_anchor=False, fixed_anchor='T1ce')

    @classmethod
    def fixed_T1(cls, model, x, registry, config, sequences):
        """③ 硬编码T1锚点"""
        return cls._base_step(model, x, registry, config, sequences,
                               use_dynamic_anchor=False, fixed_anchor='T1')

    @classmethod
    def absolute_error(cls, model, x, registry, config, sequences):
        """④ 绝对误差（去掉基准归一化）"""
        return cls._base_step(model, x, registry, config, sequences,
                               use_relative_error=False)

    @classmethod
    def no_layernorm(cls, model, x, registry, config, sequences):
        """⑤ 去掉LayerNorm"""
        return cls._base_step(model, x, registry, config, sequences,
                               use_layernorm=False)

    @classmethod
    def symmetric_dsis(cls, model, x, registry, config, sequences):
        """⑥ 对称KL（SuMi MIS风格）"""
        return cls._base_step(model, x, registry, config, sequences,
                               use_directed_dsis=False)

    @classmethod
    def no_hier(cls, model, x, registry, config, sequences):
        """⑦ 去掉层级约束"""
        return cls._base_step(model, x, registry, config, sequences,
                               use_hier=False)

    @classmethod
    def no_raf(cls, model, x, registry, config, sequences):
        """⑧ 去掉RAF（β=0）"""
        return cls._base_step(model, x, registry, config, sequences,
                               use_raf=False)


def _symmetric_mis(p_uni, p_multi, sequences):
    """对称MIS：SuMi风格，各模态地位平等"""
    loss = torch.tensor(0.0, device=p_multi.device)
    for s in sequences:
        target = 0.5 * p_uni[s] + 0.5 * p_multi
        target = target.detach()
        kl = (p_uni[s] * (torch.log(p_uni[s].clamp(1e-8)) -
                           torch.log(target.clamp(1e-8)))).sum(1).mean()
        loss += kl
    return loss / len(sequences)


# ─────────────────────────────────────────────
# 运行单个消融配置
# ─────────────────────────────────────────────

def run_ablation(name, step_fn, model, loader, device, registry,
                 config, sequences, ckpt_state):
    logger.info(f"\n{'─'*50}")
    logger.info(f"Running ablation: {name}")

    model.load_state_dict(ckpt_state)
    model.freeze_for_tta()
    model.train()
    model.encoder.eval()
    model.seq_projs.eval()
    model.decoder.eval()
    model.uni_heads.eval()
    model.rec_nets.eval()

    tracker = MetricTracker()
    anchor_counts = {s: 0 for s in sequences}

    for batch in loader:
        x = batch['x'].to(device)
        label = batch.get('label')

        result = step_fn(model, x, registry, config, sequences)
        anchor_counts[result['anchor']] += 1

        if label is not None:
            for b in range(result['pred'].shape[0]):
                p = result['pred'][b].cpu().numpy()
                t = label[b].to(device).cpu().numpy()
                tracker.update(p, t)

    summary = tracker.summary()
    total = sum(anchor_counts.values())
    summary['anchor_stats'] = {s: cnt/max(total,1)
                                for s, cnt in anchor_counts.items()}

    logger.info(f"  ET={summary['Dice_ET_mean']:.4f} "
                f"TC={summary['Dice_TC_mean']:.4f} "
                f"WT={summary['Dice_WT_mean']:.4f} "
                f"Avg={summary['Dice_avg']:.4f}")
    if 'T1ce' in anchor_counts:
        pct = 100 * anchor_counts['T1ce'] / max(total, 1)
        logger.info(f"  T1ce anchor rate: {pct:.1f}%")

    return summary


# ─────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='./checkpoints/best_model.pth')
    parser.add_argument('--baseline_errors', type=str,
                        default='./checkpoints/baseline_errors.json')
    parser.add_argument('--data_root', type=str, default='./data/brats_ped')
    parser.add_argument('--save_dir', type=str, default='./results/ablation')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config()

    # 加载模型
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = MultiSeqTTANet(
        in_channels=config.model.in_channels,
        num_classes=config.model.num_classes,
        base_ch=config.model.base_channels,
        depth=config.model.depth,
        sequences=config.model.sequences,
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    ckpt_state = ckpt['model_state']

    # 加载基准误差
    registry = BaselineErrorRegistry()
    if os.path.exists(args.baseline_errors):
        registry.load(args.baseline_errors)

    # 数据
    loader = DataLoader(
        BraTSDataset(args.data_root, mode='test', augment=False),
        batch_size=1, shuffle=False, num_workers=4,
    )

    # 消融配置
    ablations = [
        ('①Full',          AblationTTAStep.full),
        ('②FixedT1ce',     AblationTTAStep.fixed_T1ce),
        ('③FixedT1',       AblationTTAStep.fixed_T1),
        ('④AbsoluteErr',   AblationTTAStep.absolute_error),
        ('⑤NoLayerNorm',   AblationTTAStep.no_layernorm),
        ('⑥SymmetricDSIS', AblationTTAStep.symmetric_dsis),
        ('⑦NoHier',        AblationTTAStep.no_hier),
        ('⑧NoRAF',         AblationTTAStep.no_raf),
    ]

    all_results = {}
    for name, step_fn in ablations:
        result = run_ablation(
            name, step_fn, model, loader, device,
            registry, config, config.model.sequences, ckpt_state
        )
        all_results[name] = result

    # 打印汇总表
    logger.info(f"\n{'='*70}")
    logger.info(f"{'Config':<20} {'ET':>8} {'TC':>8} {'WT':>8} {'Avg':>8}")
    logger.info(f"{'─'*70}")
    for name, res in all_results.items():
        logger.info(f"{name:<20} "
                    f"{res['Dice_ET_mean']:>8.4f} "
                    f"{res['Dice_TC_mean']:>8.4f} "
                    f"{res['Dice_WT_mean']:>8.4f} "
                    f"{res['Dice_avg']:>8.4f}")

    # 保存
    out_path = os.path.join(args.save_dir, 'ablation_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
