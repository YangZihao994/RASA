from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ModelConfig:
    in_channels: int = 4
    num_classes: int = 4  # <--- 修改这里：0(bg), 1(ET), 2(NCR), 3(ED)
    base_channels: int = 32       # C，瓶颈层通道数为 4*C
    depth: int = 4                # UNet下采样层数
    num_heads: int = 8
    num_attn_layers: int = 2
    sequences: List[str] = field(
        default_factory=lambda: ['t1n', 't1c', 't2w', 't2f']
    )


@dataclass
class TrainConfig:
    data_root: str = './data/brats_gli'
    save_dir: str = './checkpoints'
    batch_size: int = 2
    num_epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-5
    lambda_rec: float = 0.1
    lambda_uni: float = 0.3
    aug_delta_ratio: float = 0.1   # δ = ratio * std(z_T2_train)
    val_interval: int = 1
    num_workers: int = 4
    prefetch_factor: int = 4
    use_amp: bool = True
    use_tqdm: bool = True
    seed: int = 42


@dataclass
class TTAConfig:
    lr: float = 1e-4
    max_grad_norm: float = 1.0
    entropy_threshold_ratio: float = 0.4   # γ_m = ratio * ln(C)
    lambda_dsis: float = 5.0
    lambda_hier: float = 1.0
    beta_init: float = 1.0                  # RAF缩放因子初始值
    temperature_eps: float = 1e-6
    rec_eps: float = 1e-8                   # log(R + eps) 数值稳定


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    tta: TTAConfig = field(default_factory=TTAConfig)