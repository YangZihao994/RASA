# data/preprocess.py
import os
import numpy as np
import random
from typing import Tuple, List, Dict, Optional

def find_all_cases_npy(data_root: str, case_ids: List[str] = None) -> Dict[str, str]:
    """
    扫描数据目录，找到所有病例的路径。
    如果提供了 case_ids，则只过滤这些病例。
    """
    cases = {}
    if case_ids is not None:
        case_list = case_ids
    else:
        case_list = sorted([
            d for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d))
        ])
        
    for case_id in case_list:
        case_dir = os.path.join(data_root, case_id)
        if os.path.isdir(case_dir):
            cases[case_id] = case_dir
            
    return cases

def normalize_volume(vol: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """Z-score 归一化"""
    if mask is None:
        mask = vol > 0
    mean = vol[mask].mean()
    std = vol[mask].std() + 1e-8
    vol = (vol - mean) / std
    vol[~mask] = 0
    return vol

def convert_label(seg: np.ndarray) -> np.ndarray:
    """BraTS 标签转换"""
    label = np.zeros_like(seg, dtype=np.int64)
    label[(seg == 1) | (seg == 2) | (seg == 4)] = 3  # WT
    label[(seg == 1) | (seg == 4)]               = 2  # TC
    label[seg == 4]                               = 1  # ET
    return label

def random_crop(vols: List[np.ndarray], label: Optional[np.ndarray], patch_size: Tuple[int, int, int]):
    """随机裁剪至指定 patch_size"""
    H, W, D = vols[0].shape
    pH, pW, pD = patch_size

    pad_h = max(0, pH - H)
    pad_w = max(0, pW - W)
    pad_d = max(0, pD - D)
    
    # 尺寸不够时先 Padding
    if pad_h > 0 or pad_w > 0 or pad_d > 0:
        vols = [np.pad(v, ((0, pad_h), (0, pad_w), (0, pad_d)), mode='constant') for v in vols]
        if label is not None:
            label = np.pad(label, ((0, pad_h), (0, pad_w), (0, pad_d)), mode='constant')
            
    H, W, D = vols[0].shape

    # 随机生成起点
    sh = random.randint(0, H - pH)
    sw = random.randint(0, W - pW)
    sd = random.randint(0, D - pD)

    # 裁剪
    vols = [v[sh:sh+pH, sw:sw+pW, sd:sd+pD] for v in vols]
    if label is not None:
        label = label[sh:sh+pH, sw:sw+pW, sd:sd+pD]

    return vols, label

def load_case_npy(
    case_id: str,
    case_dir: str,
    sequences: List[str],
    patch_size: Tuple[int, int, int]
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    """
    加载单个病例的 NPY 文件，执行归一化、标签转换及裁剪。
    """
    vols = []
    # 1. 加载所有序列并归一化
    for s in sequences:
        path = os.path.join(case_dir, f'{s}.npy')
        vol = np.load(path).astype(np.float32)
        vol = normalize_volume(vol)
        vols.append(vol)

    # 2. 加载标签并转换
    seg_path = os.path.join(case_dir, 'seg.npy')
    label = None
    if os.path.exists(seg_path):
        seg = np.round(np.load(seg_path)).astype(np.int16)
        label = convert_label(seg)

    # 3. 裁剪
    vols, label = random_crop(vols, label, patch_size)

    # 4. 堆叠为 [C, D, H, W]
    image = np.stack(vols, axis=0)
    
    info = {
        'case_id': case_id,
        'sequences': sequences,
        'patch_size': patch_size,
        'file_paths': case_dir
    }
    
    return image, label, info