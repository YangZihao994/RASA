"""
训练脚本：在BraTS GLI上训练MultiSeqTTANet。
"""
import os
import sys
import json
import logging
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Iterable

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable: Iterable, *args, **kwargs):
        return iterable

sys.path.insert(0, str(Path(__file__).parent))

from configs.config import Config
from models.model import MultiSeqTTANet
from models.reconstruction import BaselineErrorRegistry, RECONSTRUCTION_PAIRS
from data.dataset import BraTSDataset, split_cases
from losses.losses import seg_loss, uni_seg_loss

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_aug_z(z_norm_dict, sequences, delta_dict):
    z_aug = dict(z_norm_dict)  
    for s in ['t2w', 't2f']:
        if s in sequences:
            delta = delta_dict.get(s, 0.01)
            noise = torch.randn_like(z_norm_dict[s]) * delta
            z_aug[s] = z_norm_dict[s] + noise
    return z_aug

def train_one_epoch(model, loader, optimizer, scaler, device, config, delta_dict, epoch: int):
    model.train()
    total_losses = {'total': 0, 'seg': 0, 'rec': 0, 'uni': 0}
    n_batches = 0
    data_time_total, step_time_total = 0.0, 0.0
    iter_start = time.perf_counter()

    train_iter = tqdm(loader, desc=f'Train {epoch:03d}', leave=False, dynamic_ncols=True, disable=not bool(config.train.use_tqdm))
    
    for batch_idx, batch in enumerate(train_iter, start=1):
        data_ready = time.perf_counter()
        data_time_total += data_ready - iter_start

        x = batch['x'].to(device, non_blocking=True)           
        label = batch['label'].to(device, non_blocking=True)   

        optimizer.zero_grad()
        amp_enabled = bool(config.train.use_amp and device.type == 'cuda')
        
        with torch.amp.autocast('cuda', enabled=amp_enabled):
            out = model(x)
            logits     = out['logits']       
            uni_logits = out['uni_logits']   
            rec_errors = out['rec_errors']   
            z_norm     = out['z_norm']       

            L_seg = seg_loss(logits, label)
            L_rec_clean = sum(rec_errors.values())

            z_aug = get_aug_z(z_norm, model.sequences, delta_dict)
            L_rec_aug_extra = torch.tensor(0.0, device=device)
            for src, tgt in RECONSTRUCTION_PAIRS:
                z_src = z_aug.get(src, z_norm[src])
                z_hat = model.rec_nets.reconstruct({**z_norm, src: z_src}, src, tgt)
                L_rec_aug_extra += (z_norm[tgt] - z_hat).pow(2).mean()

            L_rec = L_rec_clean + 0.5 * L_rec_aug_extra
            L_uni = uni_seg_loss(uni_logits, label)
            loss = L_seg + config.train.lambda_rec * L_rec + config.train.lambda_uni * L_uni

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        step_time_total += time.perf_counter() - data_ready
        iter_start = time.perf_counter()

        total_losses['total'] += loss.item()
        total_losses['seg']   += L_seg.item()
        total_losses['rec']   += L_rec.item()
        total_losses['uni']   += L_uni.item()
        n_batches += 1

        if bool(config.train.use_tqdm) and batch_idx % 5 == 0:
            train_iter.set_postfix({
                'loss': f"{(total_losses['total'] / n_batches):.4f}",
                'data': f"{(data_time_total / n_batches):.3f}s",
                'step': f"{(step_time_total / n_batches):.3f}s",
            })

    avg_losses = {k: v / n_batches for k, v in total_losses.items()}
    avg_losses['data_time'] = data_time_total / n_batches
    avg_losses['step_time'] = step_time_total / n_batches
    return avg_losses

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    dice_et, dice_tc, dice_wt = [], [], []

    for batch in loader:
        x = batch['x'].to(device)
        label = batch['label'].to(device)

        out = model(x)
        pred = out['logits'].argmax(dim=1)  

        for b in range(pred.shape[0]):
            p = pred[b].cpu().numpy()
            t = label[b].cpu().numpy()

            dice_et.append(_dice(p == 1, t == 1))
            dice_tc.append(_dice((p == 1) | (p == 2), (t == 1) | (t == 2)))
            dice_wt.append(_dice(p > 0, t > 0))

    # 使用 np.nanmean 忽略 np.nan 的值
    return {
        'ET': float(np.nanmean(dice_et)),
        'TC': float(np.nanmean(dice_tc)),
        'WT': float(np.nanmean(dice_wt)),
        'avg': float(np.nanmean(dice_et) + np.nanmean(dice_tc) + np.nanmean(dice_wt)) / 3.0,
    }

def _dice(p: np.ndarray, t: np.ndarray, eps=1e-5) -> float:
    inter = (p & t).sum()
    union = p.sum() + t.sum()
    
    if t.sum() == 0:
        if p.sum() == 0:
            return np.nan  # 真实标签和预测都为空，忽略该Patch不计入平均
        else:
            return 0.0     # 真实标签为空，但模型乱猜了（假阳性），惩罚得0分
            
    return float(2 * inter / (union + eps))

@torch.no_grad()
def compute_baseline_errors(model, val_loader, device) -> BaselineErrorRegistry:
    model.eval()
    registry = BaselineErrorRegistry()
    for batch in val_loader:
        x = batch['x'].to(device)
        out = model(x)
        z_norm = out['z_norm']
        for src, tgt in RECONSTRUCTION_PAIRS:
            z_hat = model.rec_nets.reconstruct(z_norm, src, tgt)
            err = (z_norm[tgt] - z_hat).pow(2).mean().item()
            registry.update({(src, tgt): err})
    registry.finalize()
    return registry

@torch.no_grad()
def estimate_delta(model, train_loader, device, n_batches=20) -> dict:
    """改为直接统计输入数据的std，避免完整前向传播"""
    stds = {'t2w': [], 't2f': []}
    seq_idx = {'t2w': 2, 't2f': 3}  # 对应 ['t1n','t1c','t2w','t2f'] 的索引
    
    for i, batch in enumerate(train_loader):
        if i >= n_batches:
            break
        x = batch['x']  # 不需要转到GPU
        for s, idx in seq_idx.items():
            stds[s].append(x[:, idx].std().item())
    
    return {s: 0.1 * np.mean(v) for s, v in stds.items()}

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Train MultiSeqTTANet on BraTS GLI')
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    config = Config()

    if args.data_root:   config.train.data_root  = args.data_root
    if args.save_dir:    config.train.save_dir    = args.save_dir
    if args.batch_size:  config.train.batch_size  = args.batch_size
    if args.num_epochs:  config.train.num_epochs  = args.num_epochs
    if args.lr:          config.train.lr          = args.lr
    if args.seed:        config.train.seed        = args.seed
    if args.num_workers: config.train.num_workers = args.num_workers

    set_seed(config.train.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda': torch.backends.cudnn.benchmark = True
    os.makedirs(config.train.save_dir, exist_ok=True)

    logger.info(f"Device:    {device}")
    logger.info(f"Data root: {config.train.data_root}")

    train_ids, val_ids = split_cases(config.train.data_root, train_ratio=0.8, seed=config.train.seed)
    logger.info(f"Train: {len(train_ids)} | Val: {len(val_ids)}")

    train_loader = DataLoader(
        BraTSDataset(config.train.data_root, train_ids, mode='train', augment=True),
        batch_size=config.train.batch_size, shuffle=True, num_workers=config.train.num_workers,
        pin_memory=True, drop_last=True, persistent_workers=(config.train.num_workers > 0),
        prefetch_factor=(config.train.prefetch_factor if config.train.num_workers > 0 else None),
    )
    val_loader = DataLoader(
        BraTSDataset(config.train.data_root, val_ids, mode='val'),
        batch_size=1, shuffle=False, num_workers=2,
        pin_memory=True, persistent_workers=(config.train.num_workers > 0),
        prefetch_factor=(config.train.prefetch_factor if config.train.num_workers > 0 else None),
    )

    model = MultiSeqTTANet(
        in_channels=config.model.in_channels, num_classes=config.model.num_classes,
        base_ch=config.model.base_channels, depth=config.model.depth,
        num_heads=config.model.num_heads, num_attn_layers=config.model.num_attn_layers,
        sequences=config.model.sequences,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=bool(config.train.use_amp and device.type == 'cuda')) # 修复：更新API
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.num_epochs, eta_min=1e-6)

    model.eval()
    delta_dict = estimate_delta(model, train_loader, device)

    best_dice = 0.0

    for epoch in range(1, config.train.num_epochs + 1):
        model.train()
        train_losses = train_one_epoch(model, train_loader, optimizer, scaler, device, config, delta_dict, epoch)
        scheduler.step()

        log_str = (f"Epoch {epoch:03d}/{config.train.num_epochs} | Loss={train_losses['total']:.4f} | "
                   f"Seg={train_losses['seg']:.4f} | Rec={train_losses['rec']:.4f} | Uni={train_losses['uni']:.4f} | "
                   f"Data={train_losses['data_time']:.3f}s Step={train_losses['step_time']:.3f}s")

        if epoch % config.train.val_interval == 0:
            val_dice = validate(model, val_loader, device)
            log_str += f" | Val Dice: ET={val_dice['ET']:.4f} TC={val_dice['TC']:.4f} WT={val_dice['WT']:.4f} Avg={val_dice['avg']:.4f}"

            if val_dice['avg'] > best_dice:
                best_dice = val_dice['avg']
                torch.save({
                    'epoch': epoch, 'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'val_dice': val_dice, 'config': config,
                }, os.path.join(config.train.save_dir, 'best_model.pth'))
                log_str += ' ✓ Best'

        logger.info(log_str)

    ckpt = torch.load(os.path.join(config.train.save_dir, 'best_model.pth'), map_location=device)
    model.load_state_dict(ckpt['model_state'])
    registry = compute_baseline_errors(model, val_loader, device)
    registry.save(os.path.join(config.train.save_dir, 'baseline_errors.json'))

if __name__ == '__main__':
    main()