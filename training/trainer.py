"""
Training loop and utilities.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
from tqdm import tqdm
import os

from ..eval.metrics import compute_regression_metrics, compute_presence_metrics, merge_metrics
from .losses import compute_multitask_loss


@dataclass
class TrainState:
    """Training state tracking."""
    best_metric: float
    best_path: str
    epoch: int
    global_step: int


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[GradScaler],
    device: torch.device,
    cfg: Dict,
    pos_weight: Optional[torch.FloatTensor] = None
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training dataloader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        scaler: Gradient scaler for mixed precision
        device: Device
        cfg: Configuration
        pos_weight: Positive class weights
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    
    total_loss = 0.0
    total_loss_reg = 0.0
    total_loss_cls = 0.0
    num_batches = 0
    
    grad_accum_steps = cfg['training']['grad_accum_steps']
    max_grad_norm = cfg['training']['max_grad_norm']
    mixed_precision = cfg['training']['mixed_precision']
    
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for step, batch in enumerate(progress_bar):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        global_attention_mask = batch.get('global_attention_mask')
        if global_attention_mask is not None:
            global_attention_mask = global_attention_mask.to(device)
        y_reg = batch['y_reg'].to(device)
        y_cls = batch['y_cls'].to(device)
        
        # Forward pass
        if mixed_precision in ['fp16', 'bf16']:
            with autocast(dtype=torch.bfloat16 if mixed_precision == 'bf16' else torch.float16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    global_attention_mask=global_attention_mask
                )
                
                # Compute loss manually with pos_weight
                loss, loss_reg, loss_cls = compute_multitask_loss(
                    reg_pred=outputs['reg_pred'],
                    y_reg=y_reg,
                    cls_logit=outputs['cls_logit'],
                    y_cls=y_cls,
                    reg_loss=cfg['loss']['reg_loss'],
                    cls_loss=cfg['loss']['cls_loss'],
                    reg_weight=cfg['loss']['reg_weight'],
                    cls_weight=cfg['loss']['cls_weight'],
                    pos_weight=pos_weight
                )
                
                loss = loss / grad_accum_steps
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask
            )
            
            loss, loss_reg, loss_cls = compute_multitask_loss(
                reg_pred=outputs['reg_pred'],
                y_reg=y_reg,
                cls_logit=outputs['cls_logit'],
                y_cls=y_cls,
                reg_loss=cfg['loss']['reg_loss'],
                cls_loss=cfg['loss']['cls_loss'],
                reg_weight=cfg['loss']['reg_weight'],
                cls_weight=cfg['loss']['cls_weight'],
                pos_weight=pos_weight
            )
            
            loss = loss / grad_accum_steps
        
        # Backward pass
        if mixed_precision in ['fp16', 'bf16']:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights
        if (step + 1) % grad_accum_steps == 0:
            if mixed_precision in ['fp16', 'bf16']:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            optimizer.zero_grad()
        
        # Accumulate metrics
        total_loss += loss.item() * grad_accum_steps
        total_loss_reg += loss_reg.item()
        total_loss_cls += loss_cls.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': total_loss / num_batches,
            'loss_reg': total_loss_reg / num_batches,
            'loss_cls': total_loss_cls / num_batches
        })
    
    return {
        'loss': total_loss / num_batches,
        'loss_reg': total_loss_reg / num_batches,
        'loss_cls': total_loss_cls / num_batches
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    cfg: Dict,
    frame_names: list
) -> Dict[str, float]:
    """
    Validate model.
    
    Args:
        model: Model to validate
        dataloader: Validation dataloader
        device: Device
        cfg: Configuration
        frame_names: List of frame names
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    
    all_reg_pred = []
    all_reg_true = []
    all_cls_prob = []
    all_cls_true = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            global_attention_mask = batch.get('global_attention_mask')
            if global_attention_mask is not None:
                global_attention_mask = global_attention_mask.to(device)
            y_reg = batch['y_reg'].to(device)
            y_cls = batch['y_cls'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask
            )
            
            # Collect predictions
            all_reg_pred.append(outputs['reg_pred'].cpu().numpy())
            all_reg_true.append(y_reg.cpu().numpy())
            all_cls_prob.append(outputs['cls_prob'].cpu().numpy())
            all_cls_true.append(y_cls.cpu().numpy())
    
    # Concatenate all predictions
    y_reg_pred = np.concatenate(all_reg_pred, axis=0)
    y_reg_true = np.concatenate(all_reg_true, axis=0)
    y_cls_prob = np.concatenate(all_cls_prob, axis=0)
    y_cls_true = np.concatenate(all_cls_true, axis=0)
    
    # Compute metrics
    reg_metrics = compute_regression_metrics(y_reg_true, y_reg_pred, frame_names)
    cls_metrics = compute_presence_metrics(
        y_cls_true, y_cls_prob, frame_names,
        threshold_strategy=cfg['eval']['threshold_strategy'],
        fixed_threshold=cfg['eval']['fixed_threshold']
    )
    
    metrics = merge_metrics(reg_metrics, cls_metrics)
    
    return metrics


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    cfg: Dict,
    frame_names: list,
    device: torch.device,
    save_dir: str
) -> TrainState:
    """
    Full training loop with early stopping.
    
    Args:
        model: Model to train
        train_loader: Training dataloader
        valid_loader: Validation dataloader
        cfg: Configuration
        frame_names: List of frame names
        device: Device
        save_dir: Directory to save checkpoints
        
    Returns:
        Final training state
    """
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['training']['lr'],
        weight_decay=cfg['training']['weight_decay']
    )
    
    total_steps = len(train_loader) * cfg['training']['epochs'] // cfg['training']['grad_accum_steps']
    warmup_steps = int(total_steps * cfg['training']['warmup_ratio'])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Setup mixed precision
    scaler = None
    if cfg['training']['mixed_precision'] in ['fp16', 'bf16']:
        scaler = GradScaler()
    
    # Compute positive weights for class balancing
    pos_weight = None
    if cfg['loss']['class_balance']['enabled']:
        # Collect training labels
        train_y_cls = []
        for batch in train_loader:
            train_y_cls.append(batch['y_cls'].numpy())
        train_y_cls = np.concatenate(train_y_cls, axis=0)
        
        from .losses import compute_pos_weight
        pos_weight = compute_pos_weight(
            train_y_cls, 
            cfg['loss']['class_balance']['pos_weight_strategy']
        )
    
    # Training state
    state = TrainState(
        best_metric=-float('inf'),
        best_path="",
        epoch=0,
        global_step=0
    )
    
    # Early stopping
    early_stopping = cfg['training']['early_stopping']
    patience_counter = 0
    
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(cfg['training']['epochs']):
        print(f"\nEpoch {epoch + 1}/{cfg['training']['epochs']}")
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, cfg, pos_weight
        )
        
        # Validate
        valid_metrics = validate(model, valid_loader, device, cfg, frame_names)
        
        # Print metrics
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Reg: {train_metrics['loss_reg']:.4f}, "
              f"Cls: {train_metrics['loss_cls']:.4f}")
        print(f"Valid - Overall Alignment: {valid_metrics.get('overall_alignment', 0):.4f}")
        
        # Check for improvement
        current_metric = valid_metrics.get(early_stopping['metric'], 0)
        
        if early_stopping['mode'] == 'max':
            improved = current_metric > state.best_metric
        else:
            improved = current_metric < state.best_metric
        
        if improved:
            state.best_metric = current_metric
            state.best_path = os.path.join(save_dir, 'best.pt')
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'metrics': valid_metrics,
                'config': cfg
            }, state.best_path)
            
            patience_counter = 0
            print(f"New best model saved with {early_stopping['metric']}: {current_metric:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{early_stopping['patience']}")
        
        # Early stopping check
        if early_stopping['enabled'] and patience_counter >= early_stopping['patience']:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
        
        state.epoch = epoch
    
    return state