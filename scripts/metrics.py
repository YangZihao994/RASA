import torch
import numpy as np
from typing import Dict, List
from scipy.spatial.distance import directed_hausdorff

def compute_dice(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    results = {}
    
    # BraTS 标准后处理：如果预测的 ET 区域小于 500 个体素，则将其置零
    pred_et = (pred == 3)
    if pred_et.sum() < 500:
        pred_et = np.zeros_like(pred_et)

    # 严格对齐你的评估逻辑
    # 1=NCR, 2=ED, 3=ET
    masks = {
        'ET': (pred_et, target == 3),
        'TC': ((pred == 1) | pred_et, (target == 1) | (target == 3)),
        'WT': (pred > 0, target > 0)
    }
    
    for name, (p, t) in masks.items():
        intersection = (p & t).sum()
        union = p.sum() + t.sum()
        
        if t.sum() == 0:
            if p.sum() == 0:
                dice = np.nan  # 真阴性，不计入平均
            else:
                dice = 0.0     # 假阳性，惩罚0分
        else:
            dice = 2 * intersection / union if union > 0 else 0.0
            
        results[name] = float(dice) if not np.isnan(dice) else np.nan
    return results

def compute_hd95(pred: np.ndarray, target: np.ndarray,
                 voxel_spacing: tuple = (1.0, 1.0, 1.0)) -> Dict[str, float]:
    results = {}
    
    # 同样的后处理
    pred_et = (pred == 3)
    if pred_et.sum() < 500:
        pred_et = np.zeros_like(pred_et)

    masks = {
        'ET': (pred_et, target == 3),
        'TC': ((pred == 1) | pred_et, (target == 1) | (target == 3)),
        'WT': (pred > 0, target > 0)
    }

    for name, (p, t) in masks.items():
        p_bool = p.astype(bool)
        t_bool = t.astype(bool)

        if t_bool.sum() == 0:
            if p_bool.sum() == 0:
                results[name] = np.nan 
            else:
                results[name] = 373.13 
        elif p_bool.sum() == 0:
            results[name] = 373.13
        else:
            p_pts = np.array(np.where(p_bool)).T * np.array(voxel_spacing)
            t_pts = np.array(np.where(t_bool)).T * np.array(voxel_spacing)

            d1 = directed_hausdorff(p_pts, t_pts)[0]
            d2 = directed_hausdorff(t_pts, p_pts)[0]
            results[name] = float(max(d1, d2))

    return results

class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.dice_scores = {'ET': [], 'TC': [], 'WT': []}
        self.hd95_scores = {'ET': [], 'TC': [], 'WT': []}

    def update(self, pred: np.ndarray, target: np.ndarray,
               voxel_spacing=(1.0, 1.0, 1.0)):
        dice = compute_dice(pred, target)
        hd95 = compute_hd95(pred, target, voxel_spacing)
        
        for region in ['ET', 'TC', 'WT']:
            if not np.isnan(dice[region]):
                self.dice_scores[region].append(dice[region])
            if not np.isnan(hd95[region]):
                self.hd95_scores[region].append(hd95[region])

    def summary(self) -> Dict:
        results = {}
        for region in ['ET', 'TC', 'WT']:
            d_scores = self.dice_scores[region]
            results[f'Dice_{region}_mean'] = float(np.mean(d_scores)) if d_scores else 0.0
            results[f'Dice_{region}_std'] = float(np.std(d_scores)) if d_scores else 0.0
            
            hd_scores = self.hd95_scores[region]
            results[f'HD95_{region}_mean'] = float(np.mean(hd_scores)) if hd_scores else 0.0
            results[f'HD95_{region}_std'] = float(np.std(hd_scores)) if hd_scores else 0.0
            
        results['Dice_avg'] = (results['Dice_ET_mean'] + results['Dice_TC_mean'] + results['Dice_WT_mean']) / 3.0
        
        return results