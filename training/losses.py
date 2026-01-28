"""
Loss functions and class balancing utilities.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


def compute_multitask_loss(
    reg_pred: torch.FloatTensor,
    y_reg: torch.FloatTensor,
    cls_logit: torch.FloatTensor,
    y_cls: torch.FloatTensor,
    reg_loss: str,
    cls_loss: str,
    reg_weight: float,
    cls_weight: float,
    pos_weight: torch.FloatTensor = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute multi-task loss combining regression and classification.
    
    Args:
        reg_pred: Regression predictions [B, F]
        y_reg: Regression targets [B, F]
        cls_logit: Classification logits [B, F]
        y_cls: Classification targets [B, F]
        reg_loss: Regression loss type ('mse' or 'huber')
        cls_loss: Classification loss type ('bce' or 'focal')
        reg_weight: Weight for regression loss
        cls_weight: Weight for classification loss
        pos_weight: Positive class weights for BCE [F]
        
    Returns:
        total_loss: Combined loss
        loss_reg: Regression loss
        loss_cls: Classification loss
    """
    # Regression loss
    if reg_loss == "mse":
        loss_reg = nn.MSELoss()(reg_pred, y_reg)
    elif reg_loss == "huber":
        loss_reg = nn.HuberLoss()(reg_pred, y_reg)
    else:
        raise ValueError(f"Unknown regression loss: {reg_loss}")
    
    # Classification loss
    if cls_loss == "bce":
        if pos_weight is not None:
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            loss_fn = nn.BCEWithLogitsLoss()
        loss_cls = loss_fn(cls_logit, y_cls)
    elif cls_loss == "focal":
        loss_cls = focal_loss(cls_logit, y_cls)
    else:
        raise ValueError(f"Unknown classification loss: {cls_loss}")
    
    # Combined loss
    total_loss = reg_weight * loss_reg + cls_weight * loss_cls
    
    return total_loss, loss_reg, loss_cls


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.0,
    gamma: float = 2.0
) -> torch.Tensor:
    """
    Focal loss for addressing class imbalance.
    
    Args:
        logits: Prediction logits [B, F]
        targets: Target labels [B, F]
        alpha: Weighting factor
        gamma: Focusing parameter
        
    Returns:
        Focal loss value
    """
    bce_loss = nn.BCEWithLogitsLoss(reduction='none')(logits, targets)
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    return focal_loss.mean()


def compute_pos_weight(y_cls_train: np.ndarray, strategy: str) -> torch.FloatTensor:
    """
    Compute positive class weights for handling class imbalance.
    
    Args:
        y_cls_train: Training classification labels [N, F]
        strategy: Weighting strategy
        
    Returns:
        pos_weight: Positive class weights [F]
    """
    n_samples, n_frames = y_cls_train.shape
    pos_weight = torch.ones(n_frames)
    
    for i in range(n_frames):
        pos_count = y_cls_train[:, i].sum()
        neg_count = n_samples - pos_count
        
        if pos_count == 0:
            # All negative samples
            pos_weight[i] = 1.0
        elif neg_count == 0:
            # All positive samples
            pos_weight[i] = 1.0
        else:
            if strategy == "inverse_freq":
                pos_weight[i] = neg_count / pos_count
            elif strategy == "sqrt_inv_freq":
                pos_weight[i] = np.sqrt(neg_count / pos_count)
            elif strategy == "none":
                pos_weight[i] = 1.0
            else:
                raise ValueError(f"Unknown pos_weight strategy: {strategy}")
    
    return pos_weight