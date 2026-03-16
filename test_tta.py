"""
TTA测试脚本：在目标域（PED或SSA）上执行测试时自适应。

Usage:
    # 测试PED（结构性偏移）
    python test_tta.py --target ped --checkpoint ./checkpoints/best_model.pth

    # 测试SSA（成像偏移）
    python test_tta.py --target ssa --checkpoint ./checkpoints/best_model.pth

    # 消融：不使用TTA（Source Only）
    python test_tta.py --target ped --no_tta

    # 消融：硬编码T1ce锚点
    python test_tta.py --target ped --fixed_anchor T1ce
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from configs.config import Config, TTAConfig
from models.model import MultiSeqTTANet
from models.reconstruction import BaselineErrorRegistry
from data.dataset import BraTSDataset
from torch.utils.data import DataLoader
from tta.tta_engine import TTAEngine
from tta.sre import SequenceReliabilityEstimator
from scripts.metrics import MetricTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='ped',
                        choices=['ped', 'ssa'],
                        help='目标域：ped（儿童）或ssa（非洲）')
    parser.add_argument('--checkpoint', type=str,
                        default='./checkpoints/best_model.pth')
    parser.add_argument('--baseline_errors', type=str,
                        default='./checkpoints/baseline_errors.json')
    parser.add_argument('--data_root_ped', type=str,
                        default='./data/brats_ped')
    parser.add_argument('--data_root_ssa', type=str,
                        default='./data/brats_ssa')
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--no_tta', action='store_true',
                        help='不使用TTA（Source Only基线）')
    parser.add_argument('--fixed_anchor', type=str, default=None,
                        choices=[None, 't1n', 't1c', 't2w', 't2f'],
                        help='固定锚点（消融实验用）')
    parser.add_argument('--n_seeds', type=int, default=5,
                        help='重复实验次数（不同随机种子）')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


# ─────────────────────────────────────────────
# Source Only 评估（无TTA）
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate_source_only(model, loader, device) -> dict:
    model.eval()
    tracker = MetricTracker()

    for batch in loader:
        x = batch['x'].to(device)
        label = batch.get('label')
        if label is None:
            continue
        label = label.to(device)

        out = model(x)
        pred = out['logits'].argmax(dim=1)

        for b in range(pred.shape[0]):
            p = pred[b].cpu().numpy()
            t = label[b].cpu().numpy()
            tracker.update(p, t)

    return tracker.summary()


# ─────────────────────────────────────────────
# TTA 评估
# ─────────────────────────────────────────────

def evaluate_tta(model, loader, device, tta_config,
                 baseline_registry, sequences,
                 fixed_anchor=None) -> dict:
    """
    执行TTA并评估。
    fixed_anchor: 消融实验时固定锚点序列名称。
    """
    engine = TTAEngine(
        model=model,
        baseline_registry=baseline_registry,
        tta_config=tta_config,
        device=device,
        sequences=sequences,
    )

    tracker = MetricTracker()
    anchor_stats = {s: 0 for s in sequences}
    skip_count = 0
    entropy_vals = []

    model.train()
    # 编码器/解码器等冻结组件保持eval
    model.encoder.eval()
    model.seq_projs.eval()
    model.decoder.eval()
    model.uni_heads.eval()
    model.rec_nets.eval()

    for batch_idx, batch in enumerate(loader):
        x = batch['x'].to(device)
        label = batch.get('label')

        if fixed_anchor is not None:
            # 消融：强制使用指定锚点，跳过SRE
            result = engine._forced_anchor_adapt(x, fixed_anchor)
        else:
            result = engine.adapt_and_predict(x)

        anchor_stats[result['anchor']] += 1
        entropy_vals.append(result['entropy'])
        if result['skipped']:
            skip_count += 1

        if label is not None:
            label = label.to(device)
            pred = result['pred']
            for b in range(pred.shape[0]):
                p = pred[b].cpu().numpy()
                t = label[b].cpu().numpy()
                tracker.update(p, t)

        if (batch_idx + 1) % 10 == 0:
            logger.info(f"  [{batch_idx+1}/{len(loader)}] "
                        f"anchor={result['anchor']} H={result['entropy']:.3f}")

    # 锚点统计
    total = len(loader)
    logger.info(f"\nAnchor statistics:")
    for s, cnt in anchor_stats.items():
        logger.info(f"  {s}: {cnt}/{total} ({100*cnt/total:.1f}%)")
    logger.info(f"  Skipped: {skip_count}/{total} "
                f"({100*skip_count/total:.1f}%)")
    logger.info(f"  Mean entropy: {np.mean(entropy_vals):.4f}")

    summary = tracker.summary()
    summary['anchor_stats'] = {s: cnt/total for s, cnt in anchor_stats.items()}
    summary['skip_ratio'] = skip_count / total
    summary['mean_entropy'] = float(np.mean(entropy_vals))
    return summary


# ─────────────────────────────────────────────
# 消融：固定锚点的适配（注入到TTAEngine）
# ─────────────────────────────────────────────

def _forced_anchor_adapt(self, x, anchor):
    """为消融实验添加到TTAEngine的方法：强制使用指定锚点"""
    import torch.nn.functional as F
    x = x.to(self.device)
    target_size = x.shape[2:]

    F_bot, skips, z_dict, z_norm = self.model.encode_and_project(x)

    # 用均匀可靠性但强制指定锚点
    r_dict = {s: 0.25 for s in self.sequences}
    r_dict[anchor] = 0.7  # 给指定锚点更高权重
    # 重归一化
    total = sum(r_dict.values())
    r_dict = {s: v/total for s, v in r_dict.items()}

    z_fused = self.model.fusion(z_dict, r_dict)
    with torch.no_grad():
        logits = self.model.decode_from_fused(z_fused, skips, target_size)
        p_multi_init = F.softmax(logits, dim=1)

    skip_update, H_val = self.sre.entropy_filter(
        p_multi_init, self.model.num_classes,
        ratio=self.config.entropy_threshold_ratio
    )

    if skip_update:
        return {
            'pred': p_multi_init.argmax(dim=1),
            'prob': p_multi_init,
            'r_dict': r_dict, 'anchor': anchor,
            'skipped': True, 'entropy': H_val,
        }

    self.optimizer.zero_grad()
    z_fused = self.model.fusion(z_dict, r_dict)
    logits = self.model.decode_from_fused(z_fused, skips, target_size)
    p_multi = F.softmax(logits, dim=1)

    with torch.no_grad():
        uni_logits = self.model.uni_heads(z_dict, target_size)
    p_uni = {s: F.softmax(uni_logits[s], dim=1).detach()
             for s in self.sequences}

    from losses.losses import tta_loss
    losses = tta_loss(p_multi, p_uni, r_dict, anchor,
                      self.config.lambda_dsis, self.config.lambda_hier)
    losses['total'].backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in self.model.parameters() if p.requires_grad],
        max_norm=self.config.max_grad_norm,
    )
    self.optimizer.step()

    return {
        'pred': p_multi.detach().argmax(dim=1),
        'prob': p_multi.detach(),
        'r_dict': r_dict, 'anchor': anchor,
        'skipped': False, 'entropy': H_val,
    }


# 动态注入方法（消融用）
TTAEngine._forced_anchor_adapt = _forced_anchor_adapt


# ─────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config()

    # 确定数据根目录
    data_root = args.data_root_ped if args.target == 'ped' else args.data_root_ssa

    # ── 加载模型 ──────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = MultiSeqTTANet(
        in_channels=config.model.in_channels,
        num_classes=config.model.num_classes,
        base_ch=config.model.base_channels,
        depth=config.model.depth,
        num_heads=config.model.num_heads,
        num_attn_layers=config.model.num_attn_layers,
        sequences=config.model.sequences,
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    logger.info(f"Loaded checkpoint from {args.checkpoint}")

    # ── 加载基准误差 ──────────────────────────────────────────────
    baseline_registry = BaselineErrorRegistry()
    if os.path.exists(args.baseline_errors):
        baseline_registry.load(args.baseline_errors)
        logger.info(f"Loaded baseline errors from {args.baseline_errors}")
    else:
        logger.warning("Baseline errors not found. Using μ=1 (absolute errors).")

    # ── 数据加载 ──────────────────────────────────────────────────
    test_dataset = BraTSDataset(data_root, mode='test', augment=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=4)
    logger.info(f"Test set: {len(test_dataset)} cases "
                f"({args.target.upper()})")

    # ── 多次重复实验 ──────────────────────────────────────────────
    all_results = []

    for seed_i in range(args.n_seeds):
        seed = args.seed + seed_i
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 每次重置模型到原始checkpoint
        model.load_state_dict(ckpt['model_state'])

        logger.info(f"\n{'='*50}")
        logger.info(f"Run {seed_i+1}/{args.n_seeds} (seed={seed})")
        logger.info(f"Mode: {'Source Only' if args.no_tta else 'TTA'}")
        if args.fixed_anchor:
            logger.info(f"Fixed anchor: {args.fixed_anchor}")

        if args.no_tta:
            results = evaluate_source_only(model, test_loader, device)
        else:
            results = evaluate_tta(
                model, test_loader, device,
                tta_config=config.tta,
                baseline_registry=baseline_registry,
                sequences=config.model.sequences,
                fixed_anchor=args.fixed_anchor,
            )

        all_results.append(results)

        logger.info(f"Run {seed_i+1} Dice: "
                    f"ET={results['Dice_ET_mean']:.4f} "
                    f"TC={results['Dice_TC_mean']:.4f} "
                    f"WT={results['Dice_WT_mean']:.4f} "
                    f"Avg={results['Dice_avg']:.4f}")

    # ── 汇总统计 ──────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"Final Results ({args.n_seeds} runs, {args.target.upper()})")
    logger.info(f"{'='*60}")

    final = {}
    for metric in ['Dice_ET_mean', 'Dice_TC_mean', 'Dice_WT_mean',
                   'Dice_avg', 'HD95_ET_mean', 'HD95_TC_mean',
                   'HD95_WT_mean']:
        vals = [r[metric] for r in all_results if metric in r]
        if vals:
            final[metric] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
            }
            logger.info(f"  {metric}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # 保存结果
    mode_str = 'source_only' if args.no_tta else 'tta'
    if args.fixed_anchor:
        mode_str = f'tta_fixed_{args.fixed_anchor}'
    result_file = os.path.join(
        args.save_dir, f'{args.target}_{mode_str}_results.json'
    )
    with open(result_file, 'w') as f:
        json.dump({'final': final, 'all_runs': all_results}, f, indent=2)
    logger.info(f"\nResults saved to {result_file}")


if __name__ == '__main__':
    main()

# python test_tta.py --target ped --checkpoint ./checkpoints/best_model.pth --no_tta --data_root_ped /gaoxieping/yzh/Dataset/BraTs_2023_PED_npy
# python test_tta.py --target ped --checkpoint ./checkpoints/best_model.pth --data_root_ped /gaoxieping/yzh/Dataset/BraTs_2023_PED_npy --n_seeds 1
# python test_tta.py --target ped --checkpoint ./checkpoints/best_model.pth --fixed_anchor t1c --data_root_ped /gaoxieping/yzh/Dataset/BraTs_2023_PED_npy --n_seeds 1