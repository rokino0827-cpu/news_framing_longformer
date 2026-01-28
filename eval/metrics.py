"""
Evaluation metrics for frame monitoring.
"""
import numpy as np
from typing import Dict, List
from sklearn.metrics import (
    pearson_corrcoef, r2_score, mean_absolute_error,
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score
)
from scipy.stats import pearsonr


def compute_regression_metrics(
    y_true_reg: np.ndarray,
    y_pred_reg: np.ndarray,
    frame_names: List[str]
) -> Dict:
    """
    Compute regression metrics for frame intensity prediction.
    
    Args:
        y_true_reg: True regression values [N, F]
        y_pred_reg: Predicted regression values [N, F]
        frame_names: List of frame names
        
    Returns:
        Dictionary with regression metrics
    """
    n_frames = len(frame_names)
    per_frame_metrics = {}
    
    pearson_scores = []
    r2_scores = []
    mae_scores = []
    
    for i, frame_name in enumerate(frame_names):
        y_true_frame = y_true_reg[:, i]
        y_pred_frame = y_pred_reg[:, i]
        
        # Pearson correlation
        if np.std(y_true_frame) > 0 and np.std(y_pred_frame) > 0:
            pearson_corr, _ = pearsonr(y_true_frame, y_pred_frame)
            if np.isnan(pearson_corr):
                pearson_corr = 0.0
        else:
            pearson_corr = 0.0
        
        # R² score
        try:
            r2 = r2_score(y_true_frame, y_pred_frame)
            if np.isnan(r2) or np.isinf(r2):
                r2 = 0.0
        except:
            r2 = 0.0
        
        # MAE
        mae = mean_absolute_error(y_true_frame, y_pred_frame)
        
        per_frame_metrics[frame_name] = {
            'pearson': pearson_corr,
            'r2': r2,
            'mae': mae
        }
        
        pearson_scores.append(pearson_corr)
        r2_scores.append(r2)
        mae_scores.append(mae)
    
    # Macro averages
    macro_metrics = {
        'pearson_mean': np.mean(pearson_scores),
        'r2_mean': np.mean(r2_scores),
        'mae_mean': np.mean(mae_scores)
    }
    
    # Overall alignment (average of pearson correlations)
    overall_alignment = np.mean(pearson_scores)
    
    return {
        'per_frame': per_frame_metrics,
        'macro': macro_metrics,
        'overall_alignment': overall_alignment
    }


def compute_presence_metrics(
    y_true_cls: np.ndarray,
    y_prob_cls: np.ndarray,
    frame_names: List[str],
    threshold_strategy: str,
    fixed_threshold: float = 0.5
) -> Dict:
    """
    Compute classification metrics for frame presence detection.
    
    Args:
        y_true_cls: True binary labels [N, F]
        y_prob_cls: Predicted probabilities [N, F]
        frame_names: List of frame names
        threshold_strategy: Threshold selection strategy
        fixed_threshold: Fixed threshold value
        
    Returns:
        Dictionary with classification metrics
    """
    n_frames = len(frame_names)
    per_frame_metrics = {}
    
    auc_roc_scores = []
    auc_pr_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for i, frame_name in enumerate(frame_names):
        y_true_frame = y_true_cls[:, i]
        y_prob_frame = y_prob_cls[:, i]
        
        # AUC-ROC
        try:
            if len(np.unique(y_true_frame)) > 1:
                auc_roc = roc_auc_score(y_true_frame, y_prob_frame)
            else:
                auc_roc = 0.5  # Random performance for single class
        except:
            auc_roc = 0.5
        
        # AUC-PR
        try:
            if len(np.unique(y_true_frame)) > 1:
                auc_pr = average_precision_score(y_true_frame, y_prob_frame)
            else:
                auc_pr = y_true_frame.mean()  # Baseline for single class
        except:
            auc_pr = y_true_frame.mean()
        
        # Determine threshold
        if threshold_strategy == "fixed":
            threshold = fixed_threshold
        elif threshold_strategy == "per_frame_opt_f1":
            threshold = find_optimal_threshold_f1(y_true_frame, y_prob_frame)
        elif threshold_strategy == "per_frame_opt_pr":
            threshold = find_optimal_threshold_precision(y_true_frame, y_prob_frame)
        else:
            threshold = fixed_threshold
        
        # Binary predictions
        y_pred_frame = (y_prob_frame >= threshold).astype(int)
        
        # Precision, Recall, F1
        try:
            precision = precision_score(y_true_frame, y_pred_frame, zero_division=0)
            recall = recall_score(y_true_frame, y_pred_frame, zero_division=0)
            f1 = f1_score(y_true_frame, y_pred_frame, zero_division=0)
        except:
            precision = recall = f1 = 0.0
        
        per_frame_metrics[frame_name] = {
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'threshold': threshold
        }
        
        auc_roc_scores.append(auc_roc)
        auc_pr_scores.append(auc_pr)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
    
    # Macro averages
    macro_metrics = {
        'auc_roc_mean': np.mean(auc_roc_scores),
        'auc_pr_mean': np.mean(auc_pr_scores),
        'precision_mean': np.mean(precision_scores),
        'recall_mean': np.mean(recall_scores),
        'f1_mean': np.mean(f1_scores)
    }
    
    return {
        'per_frame': per_frame_metrics,
        'macro': macro_metrics
    }


def find_optimal_threshold_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find threshold that maximizes F1 score."""
    thresholds = np.linspace(0.1, 0.9, 9)
    best_f1 = 0.0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        try:
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        except:
            continue
    
    return best_threshold


def find_optimal_threshold_precision(y_true: np.ndarray, y_prob: np.ndarray, target_precision: float = 0.8) -> float:
    """Find threshold that achieves target precision or maximizes F1."""
    thresholds = np.linspace(0.1, 0.9, 9)
    best_f1 = 0.0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        try:
            precision = precision_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            if precision >= target_precision and f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
            elif precision < target_precision and f1 > best_f1:
                # Fallback to best F1 if target precision not achievable
                best_f1 = f1
                best_threshold = threshold
        except:
            continue
    
    return best_threshold


def merge_metrics(reg_metrics: Dict, cls_metrics: Dict) -> Dict:
    """
    Merge regression and classification metrics into single dictionary.
    
    Args:
        reg_metrics: Regression metrics
        cls_metrics: Classification metrics
        
    Returns:
        Combined metrics dictionary
    """
    merged = {
        'regression': reg_metrics,
        'classification': cls_metrics
    }
    
    # Add top-level metrics for easy access
    merged['overall_alignment'] = reg_metrics.get('overall_alignment', 0.0)
    merged['pearson_mean'] = reg_metrics['macro']['pearson_mean']
    merged['r2_mean'] = reg_metrics['macro']['r2_mean']
    merged['mae_mean'] = reg_metrics['macro']['mae_mean']
    merged['auc_roc_mean'] = cls_metrics['macro']['auc_roc_mean']
    merged['auc_pr_mean'] = cls_metrics['macro']['auc_pr_mean']
    merged['precision_mean'] = cls_metrics['macro']['precision_mean']
    merged['recall_mean'] = cls_metrics['macro']['recall_mean']
    merged['f1_mean'] = cls_metrics['macro']['f1_mean']
    
    return merged