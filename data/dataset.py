# data/dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
import random

from .preprocess import find_all_cases_npy, load_case_npy


class BraTSDataset(Dataset):
    """
    BraTS多序列MRI数据集 (NPY格式)。
    已重构，底层数据加载解耦至 preprocess.py 中。
    """

    def __init__(self,
                 data_root: str,
                 case_ids: List[str] = None,
                 patch_size: Tuple[int, int, int] = (128, 128, 128),
                 mode: str = 'train',
                 sequences: List[str] = None,
                 augment: bool = True):
        self.data_root = data_root
        self.patch_size = patch_size
        self.mode = mode
        self.sequences = sequences or ['t1n', 't1c', 't2w', 't2f']
        self.augment = augment and (mode == 'train')

        # 扫描并过滤病例（与预处理解耦）
        self.cases = find_all_cases_npy(data_root, case_ids)
        self.case_ids = list(self.cases.keys())

    def __len__(self) -> int:
        return len(self.case_ids)

    def _augment(self, image: np.ndarray, mask: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """数据增强操作（空间维度的翻转）"""
        # image shape is [C, H, W, D], spatial axes are 1, 2, 3
        for axis in [1, 2, 3]:
            if random.random() > 0.5:
                image = np.flip(image, axis=axis).copy()
                if mask is not None:
                    # mask shape is [H, W, D], spatial axes are 0, 1, 2
                    mask = np.flip(mask, axis=axis-1).copy()
        return image, mask

    def __getitem__(self, idx: int) -> Dict:
        case_id = self.case_ids[idx]
        case_dir = self.cases[case_id]

        # 调用外部加载模块（内部包含了归一化与裁剪逻辑）
        image, label, info = load_case_npy(
            case_id=case_id,
            case_dir=case_dir,
            sequences=self.sequences,
            patch_size=self.patch_size
        )

        # 训练时的数据增强
        if self.augment:
            image, label = self._augment(image, label)

        # 确保内存连续性，加速后续 Tensor 的运算
        image = np.ascontiguousarray(image)
        x = torch.from_numpy(image).float()

        result = {'x': x, 'case_id': case_id}
        
        if label is not None:
            label = np.ascontiguousarray(label)
            result['label'] = torch.from_numpy(label).long()

        return result


def get_dataloader(data_root: str, case_ids: List[str] = None,
                   patch_size=(128, 128, 128), mode='train',
                   batch_size=2, num_workers=4,
                   sequences=None) -> DataLoader:
    """接口保持不变"""
    dataset = BraTSDataset(
        data_root=data_root,
        case_ids=case_ids,
        patch_size=patch_size,
        mode=mode,
        sequences=sequences,
        augment=(mode == 'train'),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(mode == 'train'),
    )


def split_cases(data_root: str,
                train_ratio: float = 0.8,
                seed: int = 42) -> Tuple[List[str], List[str]]:
    """接口保持不变"""
    cases = sorted([
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    ])
    random.seed(seed)
    random.shuffle(cases)
    n_train = int(len(cases) * train_ratio)
    return cases[:n_train], cases[n_train:]