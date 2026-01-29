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
    threshold_strategies: Dict,
    default_strategy: str = "balanced_thresholds"
) -> Dict:
    """
    Compute classification metrics for frame presence detection with multiple threshold strategies.
    
    Args:
        y_true_cls: True binary labels [N, F]
        y_prob_cls: Predicted probabilities [N, F]
        frame_names: List of frame names
        threshold_strategies: Dictionary of threshold strategies from config
        default_strategy: Default strategy to use for primary metrics
        
    Returns:
        Dictionary with classification metrics for all threshold strategies
    """
    n_frames = len(frame_names)
    results = {}
    
    # Compute metrics for each threshold strategy
    for strategy_name, strategy_config in threshold_strategies.items():
        per_frame_metrics = {}
        
        auc_roc_scores = []
        auc_pr_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        thresholds_used = []
        
        for i, frame_name in enumerate(frame_names):
            y_true_frame = y_true_cls[:, i]
            y_prob_frame = y_prob_cls[:, i]
            
            # AUC-ROC (threshold-independent)
            try:
                if len(np.unique(y_true_frame)) > 1:
                    auc_roc = roc_auc_score(y_true_frame, y_prob_frame)
                else:
                    auc_roc = 0.5  # Random performance for single class
            except:
                auc_roc = 0.5
            
            # AUC-PR (threshold-independent)
            try:
                if len(np.unique(y_true_frame)) > 1:
                    auc_pr = average_precision_score(y_true_frame, y_prob_frame)
                else:
                    auc_pr = y_true_frame.mean()  # Baseline for single class
            except:
                auc_pr = y_true_frame.mean()
            
            # Determine threshold based on strategy
            threshold = determine_threshold(
                y_true_frame, y_prob_frame, strategy_config
            )
            
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
            thresholds_used.append(threshold)
        
        # Macro averages for this strategy
        macro_metrics = {
            'auc_roc_mean': np.mean(auc_roc_scores),
            'auc_pr_mean': np.mean(auc_pr_scores),
            'precision_mean': np.mean(precision_scores),
            'recall_mean': np.mean(recall_scores),
            'f1_mean': np.mean(f1_scores),
            'threshold_mean': np.mean(thresholds_used),
            'threshold_std': np.std(thresholds_used)
        }
        
        results[strategy_name] = {
            'per_frame': per_frame_metrics,
            'macro': macro_metrics,
            'strategy_config': strategy_config,
            'use_case': strategy_config.get('use_case', 'Not specified')
        }
    
    # Mark the default strategy for easy access
    if default_strategy in results:
        results['default'] = results[default_strategy]
    
    return results


def determine_threshold(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    strategy_config: Dict
) -> float:
    """
    Determine optimal threshold based on strategy configuration.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        strategy_config: Strategy configuration dictionary
        
    Returns:
        Optimal threshold value
    """
    strategy = strategy_config.get('strategy', 'per_frame_opt_f1')
    fallback_threshold = strategy_config.get('fallback_threshold', 0.5)
    
    if strategy == "per_frame_opt_f1":
        return find_optimal_threshold_f1(y_true, y_prob, fallback_threshold)
    
    elif strategy == "per_frame_opt_precision":
        target_precision = strategy_config.get('target_precision', 0.8)
        return find_optimal_threshold_precision(y_true, y_prob, target_precision, fallback_threshold)
    
    elif strategy == "per_frame_opt_recall":
        target_recall = strategy_config.get('target_recall', 0.85)
        return find_optimal_threshold_recall(y_true, y_prob, target_recall, fallback_threshold)
    
    elif strategy == "fixed":
        return fallback_threshold
    
    else:
        print(f"Warning: Unknown threshold strategy '{strategy}', using fallback")
        return fallback_threshold


def find_optimal_threshold_f1(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    fallback: float = 0.5
) -> float:
    """Find threshold that maximizes F1 score."""
    thresholds = np.linspace(0.05, 0.95, 19)  # More granular search
    best_f1 = 0.0
    best_threshold = fallback
    
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


def find_optimal_threshold_precision(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    target_precision: float = 0.8,
    fallback: float = 0.7
) -> float:
    """Find threshold that achieves target precision or maximizes F1."""
    thresholds = np.linspace(0.05, 0.95, 19)
    best_f1 = 0.0
    best_threshold = fallback
    target_achieved = False
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        try:
            precision = precision_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            if precision >= target_precision:
                if not target_achieved or f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                    target_achieved = True
            elif not target_achieved and f1 > best_f1:
                # Fallback to best F1 if target precision not achievable
                best_f1 = f1
                best_threshold = threshold
        except:
            continue
    
    return best_threshold


def find_optimal_threshold_recall(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    target_recall: float = 0.85,
    fallback: float = 0.3
) -> float:
    """Find threshold that achieves target recall or maximizes F1."""
    thresholds = np.linspace(0.05, 0.95, 19)
    best_f1 = 0.0
    best_threshold = fallback
    target_achieved = False
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        try:
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            if recall >= target_recall:
                if not target_achieved or f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                    target_achieved = True
            elif not target_achieved and f1 > best_f1:
                # Fallback to best F1 if target recall not achievable
                best_f1 = f1
                best_threshold = threshold
        except:
            continue
    
    return best_threshold


def merge_metrics(reg_metrics: Dict, cls_metrics: Dict, default_strategy: str = "balanced_thresholds") -> Dict:
    """
    Merge regression and classification metrics into single dictionary.
    
    Args:
        reg_metrics: Regression metrics
        cls_metrics: Classification metrics (with multiple threshold strategies)
        default_strategy: Which classification strategy to use for top-level metrics
        
    Returns:
        Combined metrics dictionary
    """
    merged = {
        'regression': reg_metrics,
        'classification': cls_metrics
    }
    
    # Add top-level metrics for easy access (using default strategy)
    merged['overall_alignment'] = reg_metrics.get('overall_alignment', 0.0)
    merged['pearson_mean'] = reg_metrics['macro']['pearson_mean']
    merged['r2_mean'] = reg_metrics['macro']['r2_mean']
    merged['mae_mean'] = reg_metrics['macro']['mae_mean']
    
    # Use default classification strategy for top-level metrics
    if default_strategy in cls_metrics:
        default_cls = cls_metrics[default_strategy]['macro']
        merged['auc_roc_mean'] = default_cls['auc_roc_mean']
        merged['auc_pr_mean'] = default_cls['auc_pr_mean']
        merged['precision_mean'] = default_cls['precision_mean']
        merged['recall_mean'] = default_cls['recall_mean']
        merged['f1_mean'] = default_cls['f1_mean']
        merged['threshold_mean'] = default_cls['threshold_mean']
    else:
        # Fallback to first available strategy
        first_strategy = list(cls_metrics.keys())[0]
        if first_strategy != 'default':
            default_cls = cls_metrics[first_strategy]['macro']
            merged['auc_roc_mean'] = default_cls['auc_roc_mean']
            merged['auc_pr_mean'] = default_cls['auc_pr_mean']
            merged['precision_mean'] = default_cls['precision_mean']
            merged['recall_mean'] = default_cls['recall_mean']
            merged['f1_mean'] = default_cls['f1_mean']
            merged['threshold_mean'] = default_cls['threshold_mean']
    
    return merged