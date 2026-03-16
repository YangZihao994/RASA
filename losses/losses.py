import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


# ─────────────────────────────────────────────
# 训练损失
# ─────────────────────────────────────────────

def dice_loss(pred: torch.Tensor, target: torch.Tensor,
              smooth: float = 1e-5) -> torch.Tensor:
    """
    Soft Dice Loss（多类别，每类独立计算后平均）。
    pred:   [B, C, H, W, D]，softmax后的概率
    target: [B, H, W, D]，整数标签，值域必须在 [0, C)
    """
    C = pred.shape[1]
    target_long = target.long().clamp(0, C - 1)   # 防御性clamp，避免越界崩溃
    target_onehot = F.one_hot(target_long, C).permute(0, 4, 1, 2, 3).float()

    pred_flat = pred.reshape(pred.shape[0], C, -1)
    tgt_flat = target_onehot.reshape(target_onehot.shape[0], C, -1)

    intersection = (pred_flat * tgt_flat).sum(-1)
    union = pred_flat.sum(-1) + tgt_flat.sum(-1)

    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def seg_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """主分割损失：Dice + CE"""
    C = logits.shape[1]
    target_long = target.long().clamp(0, C - 1)   # 防御性clamp
    pred = F.softmax(logits, dim=1)
    loss_dice = dice_loss(pred, target_long)
    loss_ce = F.cross_entropy(logits, target_long)
    return loss_dice + loss_ce

def rec_loss_aug(rec_errors: Dict, z_dict_aug: Dict[str, torch.Tensor],
                 rec_nets, z_norm_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    L_rec = sum(rec_errors.values())
    L_aug = torch.tensor(0.0, device=L_rec.device)
    
    from models.reconstruction import RECONSTRUCTION_PAIRS
    for src, tgt in RECONSTRUCTION_PAIRS:
        z_src = z_dict_aug.get(src, z_norm_dict[src])
        # 修复：直接用原有特征字典覆盖当前源特征，避免手动硬编码其它序列
        input_dict = {**z_norm_dict, src: z_src}
        z_hat = rec_nets.reconstruct(input_dict, src, tgt)
        L_aug = L_aug + (z_norm_dict[tgt] - z_hat).pow(2).mean()

    return L_rec + 0.5 * L_aug


def uni_seg_loss(uni_logits: Dict[str, torch.Tensor],
                 target: torch.Tensor) -> torch.Tensor:
    """单模态辅助分割损失：对每个序列独立计算Dice+CE后求和"""
    loss = torch.tensor(0.0, device=target.device)
    for s, logits in uni_logits.items():
        loss = loss + seg_loss(logits, target)
    return loss


def total_train_loss(logits, uni_logits, rec_errors, z_norm, z_dict_aug,
                     rec_nets, target,
                     lambda_rec: float = 0.1,
                     lambda_uni: float = 0.3) -> Dict[str, torch.Tensor]:
    """
    总训练损失：
    L_train = L_seg + λ_rec * L_rec^aug + λ_uni * L_uni
    """
    L_seg = seg_loss(logits, target)
    L_uni = uni_seg_loss(uni_logits, target)

    # 重建损失（简化：直接用已计算的误差）
    L_rec = sum(rec_errors.values())

    total = L_seg + lambda_rec * L_rec + lambda_uni * L_uni

    return {
        'total': total,
        'seg': L_seg,
        'rec': L_rec,
        'uni': L_uni,
    }


# ─────────────────────────────────────────────
# TTA损失
# ─────────────────────────────────────────────

def entropy_loss(p: torch.Tensor) -> torch.Tensor:
    """
    预测分布的熵：H(p) = -Σ p_c log(p_c)
    p: [B, C, H, W, D]，概率值
    返回标量（batch平均）
    """
    return -(p * torch.log(p.clamp(min=1e-8))).sum(dim=1).mean()


def dsis_loss(p_uni: Dict[str, torch.Tensor],
              p_multi: torch.Tensor,
              r_dict: Dict[str, float],
              anchor: str,
              eps: float = 1e-6) -> torch.Tensor:
    r_vals = torch.tensor(list(r_dict.values()), dtype=torch.float32, device=p_multi.device)
    tau_alpha = r_vals.std() + eps
    r_anchor = r_dict[anchor]
    p_anchor = p_uni[anchor]

    sequences = list(r_dict.keys())
    non_anchor = [s for s in sequences if s != anchor]
    total_kl = torch.tensor(0.0, device=p_multi.device)

    for s in non_anchor:
        r_s = r_dict[s]
        # 计算相对可靠性权重 (注意：不需要计算梯度)
        alpha = torch.sigmoid(
            torch.tensor((r_anchor - r_s) / tau_alpha.item(), device=p_multi.device)
        ).detach()

        # 【核心修复】：构建 Teacher 分布 
        # 融合高置信度的锚点与单模态先验 
        # 🏷️ Teacher [❄️] (不可导，仅作监督信号)
        target_dist = (alpha * p_anchor + (1 - alpha) * p_uni[s]).detach()

        # 【核心修复】：计算 KL(Teacher || Student)
        # 🏷️ Student [✨] (需计算梯度)
        kl = (target_dist * (
            torch.log(target_dist.clamp(min=1e-8)) -
            torch.log(p_multi.clamp(min=1e-8))
        )).sum(dim=1).mean()

        total_kl = total_kl + kl

    return total_kl / len(non_anchor)

def hierarchy_loss(p_uni: Dict[str, torch.Tensor], 
                   p_multi: torch.Tensor) -> torch.Tensor:
    """
    【修复2】：多模态预测(p_multi) 作为变量更新，单模态(p_uni) 作为冻结先验约束。
    同时修复 BraTS 嵌套区域概率相加的逻辑。
    """
    # ET ⊆ TC：多模态预测的 ET 概率 应受限于 T2w 预测的 TC(ET+NCR) 概率
    p_multi_et = p_multi[:, 1]
    p_t2_tc = p_uni['t2w'][:, 1] + p_uni['t2w'][:, 2]
    L_ET_TC = (p_multi_et * F.relu(p_multi_et - p_t2_tc)).mean()

    # TC ⊆ WT：多模态预测的 TC(ET+NCR) 概率 应受限于 FLAIR 预测的 WT(ET+NCR+ED) 概率
    p_multi_tc = p_multi[:, 1] + p_multi[:, 2]
    p_flair_wt = p_uni['t2f'][:, 1] + p_uni['t2f'][:, 2] + p_uni['t2f'][:, 3]
    L_TC_WT = (p_multi_tc * F.relu(p_multi_tc - p_flair_wt)).mean()

    return L_ET_TC + L_TC_WT


def tta_loss(p_multi: torch.Tensor,
             p_uni: Dict[str, torch.Tensor],
             r_dict: Dict[str, float],
             anchor: str,
             lambda_dsis: float = 5.0,
             lambda_hier: float = 1.0) -> Dict[str, torch.Tensor]:
    
    L_entropy = entropy_loss(p_multi)
    L_dsis = dsis_loss(p_uni, p_multi, r_dict, anchor)
    
    # 【修复3】：必须把带梯度的 p_multi 传入 hierarchy_loss
    L_hier = hierarchy_loss(p_uni, p_multi)

    total = L_entropy + lambda_dsis * L_dsis + lambda_hier * L_hier

    return {
        'total': total,
        'entropy': L_entropy,
        'dsis': L_dsis,
        'hier': L_hier,
    }