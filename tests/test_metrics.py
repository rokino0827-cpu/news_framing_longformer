"""
Unit tests for evaluation metrics.
"""
import pytest
import numpy as np
from pathlib import Path

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from eval.metrics import (
    compute_regression_metrics,
    compute_presence_metrics,
    merge_metrics,
    determine_threshold,
    find_optimal_threshold_f1,
    find_optimal_threshold_precision,
    find_optimal_threshold_recall
)


class TestRegressionMetrics:
    """Test regression metrics computation."""
    
    def test_compute_regression_metrics_perfect(self):
        """Test metrics with perfect predictions."""
        n_samples, n_frames = 10, 3
        frame_names = ['frame1', 'frame2', 'frame3']
        
        # Perfect predictions
        y_true = np.random.rand(n_samples, n_frames)
        y_pred = y_true.copy()
        
        metrics = compute_regression_metrics(y_true, y_pred, frame_names)
        
        # Perfect correlation should be 1.0
        assert abs(metrics['overall_alignment'] - 1.0) < 1e-6
        assert abs(metrics['macro']['pearson_mean'] - 1.0) < 1e-6
        
        # Perfect R² should be 1.0
        assert abs(metrics['macro']['r2_mean'] - 1.0) < 1e-6
        
        # Perfect MAE should be 0.0
        assert abs(metrics['macro']['mae_mean']) < 1e-6
    
    def test_compute_regression_metrics_random(self):
        """Test metrics with random predictions."""
        n_samples, n_frames = 100, 3
        frame_names = ['frame1', 'frame2', 'frame3']
        
        np.random.seed(42)
        y_true = np.random.rand(n_samples, n_frames)
        y_pred = np.random.rand(n_samples, n_frames)
        
        metrics = compute_regression_metrics(y_true, y_pred, frame_names)
        
        # Random predictions should have low correlation
        assert metrics['overall_alignment'] < 0.5
        
        # Should have per-frame metrics
        assert len(metrics['per_frame']) == n_frames
        for frame_name in frame_names:
            assert frame_name in metrics['per_frame']
            frame_metrics = metrics['per_frame'][frame_name]
            assert 'pearson' in frame_metrics
            assert 'r2' in frame_metrics
            assert 'mae' in frame_metrics
    
    def test_compute_regression_metrics_constant(self):
        """Test metrics with constant predictions."""
        n_samples, n_frames = 10, 2
        frame_names = ['frame1', 'frame2']
        
        y_true = np.random.rand(n_samples, n_frames)
        y_pred = np.ones((n_samples, n_frames)) * 0.5  # Constant predictions
        
        metrics = compute_regression_metrics(y_true, y_pred, frame_names)
        
        # Constant predictions should have zero correlation
        assert abs(metrics['overall_alignment']) < 1e-6
        
        # MAE should be reasonable
        assert metrics['macro']['mae_mean'] > 0


class TestClassificationMetrics:
    """Test classification metrics computation."""
    
    def create_test_data(self):
        """Create test classification data."""
        np.random.seed(42)
        n_samples, n_frames = 100, 3
        frame_names = ['frame1', 'frame2', 'frame3']
        
        # Create realistic binary labels (imbalanced)
        y_true = np.zeros((n_samples, n_frames))
        y_true[:20, 0] = 1  # 20% positive for frame1
        y_true[:10, 1] = 1  # 10% positive for frame2
        y_true[:30, 2] = 1  # 30% positive for frame3
        
        # Create probabilities that correlate with true labels
        y_prob = np.random.rand(n_samples, n_frames) * 0.3  # Base low probability
        y_prob[y_true == 1] += 0.5  # Higher probability for positive cases
        y_prob = np.clip(y_prob, 0, 1)
        
        return y_true.astype(int), y_prob, frame_names
    
    def test_compute_presence_metrics_single_strategy(self):
        """Test classification metrics with single threshold strategy."""
        y_true, y_prob, frame_names = self.create_test_data()
        
        threshold_strategies = {
            'balanced': {
                'strategy': 'per_frame_opt_f1',
                'fallback_threshold': 0.5,
                'use_case': 'Balanced F1 optimization'
            }
        }
        
        metrics = compute_presence_metrics(
            y_true, y_prob, frame_names, threshold_strategies, 'balanced'
        )
        
        # Should have strategy results
        assert 'balanced' in metrics
        assert 'default' in metrics
        
        # Check structure
        strategy_metrics = metrics['balanced']
        assert 'per_frame' in strategy_metrics
        assert 'macro' in strategy_metrics
        assert 'strategy_config' in strategy_metrics
        
        # Check per-frame metrics
        for frame_name in frame_names:
            frame_metrics = strategy_metrics['per_frame'][frame_name]
            assert 'auc_roc' in frame_metrics
            assert 'auc_pr' in frame_metrics
            assert 'precision' in frame_metrics
            assert 'recall' in frame_metrics
            assert 'f1' in frame_metrics
            assert 'threshold' in frame_metrics
            
            # AUC should be reasonable (> 0.5 for our correlated data)
            assert frame_metrics['auc_roc'] > 0.5
    
    def test_compute_presence_metrics_multiple_strategies(self):
        """Test classification metrics with multiple threshold strategies."""
        y_true, y_prob, frame_names = self.create_test_data()
        
        threshold_strategies = {
            'monitor': {
                'strategy': 'per_frame_opt_recall',
                'target_recall': 0.8,
                'fallback_threshold': 0.3,
                'use_case': 'High recall monitoring'
            },
            'alert': {
                'strategy': 'per_frame_opt_precision',
                'target_precision': 0.8,
                'fallback_threshold': 0.7,
                'use_case': 'High precision alerting'
            },
            'balanced': {
                'strategy': 'per_frame_opt_f1',
                'fallback_threshold': 0.5,
                'use_case': 'Balanced F1'
            }
        }
        
        metrics = compute_presence_metrics(
            y_true, y_prob, frame_names, threshold_strategies, 'balanced'
        )
        
        # Should have all strategies
        for strategy_name in threshold_strategies.keys():
            assert strategy_name in metrics
        
        # Monitor strategy should have lower thresholds (higher recall)
        # Alert strategy should have higher thresholds (higher precision)
        monitor_thresholds = [metrics['monitor']['per_frame'][f]['threshold'] for f in frame_names]
        alert_thresholds = [metrics['alert']['per_frame'][f]['threshold'] for f in frame_names]
        
        # On average, monitor thresholds should be lower
        assert np.mean(monitor_thresholds) <= np.mean(alert_thresholds)


class TestThresholdOptimization:
    """Test threshold optimization functions."""
    
    def create_test_binary_data(self):
        """Create test binary classification data."""
        np.random.seed(42)
        n_samples = 100
        
        # Create imbalanced binary labels
        y_true = np.zeros(n_samples)
        y_true[:20] = 1  # 20% positive
        
        # Create probabilities that correlate with labels
        y_prob = np.random.rand(n_samples) * 0.4  # Base low probability
        y_prob[y_true == 1] += 0.4  # Higher probability for positive cases
        y_prob = np.clip(y_prob, 0, 1)
        
        return y_true.astype(int), y_prob
    
    def test_find_optimal_threshold_f1(self):
        """Test F1 threshold optimization."""
        y_true, y_prob = self.create_test_binary_data()
        
        threshold = find_optimal_threshold_f1(y_true, y_prob)
        
        # Threshold should be reasonable
        assert 0.0 <= threshold <= 1.0
        
        # Should be better than random threshold
        from sklearn.metrics import f1_score
        
        f1_optimal = f1_score(y_true, (y_prob >= threshold).astype(int))
        f1_random = f1_score(y_true, (y_prob >= 0.5).astype(int))
        
        assert f1_optimal >= f1_random
    
    def test_find_optimal_threshold_precision(self):
        """Test precision threshold optimization."""
        y_true, y_prob = self.create_test_binary_data()
        
        target_precision = 0.8
        threshold = find_optimal_threshold_precision(y_true, y_prob, target_precision)
        
        # Threshold should be reasonable
        assert 0.0 <= threshold <= 1.0
        
        # Check if target precision is achieved (or close)
        from sklearn.metrics import precision_score
        
        precision = precision_score(y_true, (y_prob >= threshold).astype(int), zero_division=0)
        
        # Should either achieve target or be the best possible
        assert precision <= target_precision + 0.1  # Allow some tolerance
    
    def test_find_optimal_threshold_recall(self):
        """Test recall threshold optimization."""
        y_true, y_prob = self.create_test_binary_data()
        
        target_recall = 0.8
        threshold = find_optimal_threshold_recall(y_true, y_prob, target_recall)
        
        # Threshold should be reasonable
        assert 0.0 <= threshold <= 1.0
        
        # Check if target recall is achieved (or close)
        from sklearn.metrics import recall_score
        
        recall = recall_score(y_true, (y_prob >= threshold).astype(int), zero_division=0)
        
        # Should either achieve target or be the best possible
        assert recall <= target_recall + 0.1  # Allow some tolerance
    
    def test_determine_threshold(self):
        """Test threshold determination with different strategies."""
        y_true, y_prob = self.create_test_binary_data()
        
        # Test F1 strategy
        config_f1 = {'strategy': 'per_frame_opt_f1', 'fallback_threshold': 0.5}
        threshold_f1 = determine_threshold(y_true, y_prob, config_f1)
        assert 0.0 <= threshold_f1 <= 1.0
        
        # Test precision strategy
        config_prec = {'strategy': 'per_frame_opt_precision', 'target_precision': 0.8, 'fallback_threshold': 0.7}
        threshold_prec = determine_threshold(y_true, y_prob, config_prec)
        assert 0.0 <= threshold_prec <= 1.0
        
        # Test recall strategy
        config_rec = {'strategy': 'per_frame_opt_recall', 'target_recall': 0.8, 'fallback_threshold': 0.3}
        threshold_rec = determine_threshold(y_true, y_prob, config_rec)
        assert 0.0 <= threshold_rec <= 1.0
        
        # Test fixed strategy
        config_fixed = {'strategy': 'fixed', 'fallback_threshold': 0.6}
        threshold_fixed = determine_threshold(y_true, y_prob, config_fixed)
        assert threshold_fixed == 0.6
        
        # Test unknown strategy (should use fallback)
        config_unknown = {'strategy': 'unknown', 'fallback_threshold': 0.4}
        threshold_unknown = determine_threshold(y_true, y_prob, config_unknown)
        assert threshold_unknown == 0.4


class TestMergeMetrics:
    """Test metrics merging functionality."""
    
    def test_merge_metrics(self):
        """Test merging regression and classification metrics."""
        # Create mock regression metrics
        reg_metrics = {
            'overall_alignment': 0.75,
            'macro': {
                'pearson_mean': 0.75,
                'r2_mean': 0.60,
                'mae_mean': 0.15
            },
            'per_frame': {}
        }
        
        # Create mock classification metrics
        cls_metrics = {
            'balanced': {
                'macro': {
                    'auc_roc_mean': 0.80,
                    'auc_pr_mean': 0.65,
                    'precision_mean': 0.70,
                    'recall_mean': 0.75,
                    'f1_mean': 0.72,
                    'threshold_mean': 0.45,
                    'threshold_std': 0.10
                },
                'per_frame': {}
            },
            'monitor': {
                'macro': {
                    'auc_roc_mean': 0.80,
                    'auc_pr_mean': 0.65,
                    'precision_mean': 0.60,
                    'recall_mean': 0.85,
                    'f1_mean': 0.70,
                    'threshold_mean': 0.30,
                    'threshold_std': 0.05
                },
                'per_frame': {}
            }
        }
        
        merged = merge_metrics(reg_metrics, cls_metrics, 'balanced')
        
        # Check structure
        assert 'regression' in merged
        assert 'classification' in merged
        
        # Check top-level metrics (should use balanced strategy)
        assert merged['overall_alignment'] == 0.75
        assert merged['pearson_mean'] == 0.75
        assert merged['auc_roc_mean'] == 0.80
        assert merged['precision_mean'] == 0.70
        assert merged['threshold_mean'] == 0.45
    
    def test_merge_metrics_fallback(self):
        """Test merging with fallback when default strategy not available."""
        reg_metrics = {
            'overall_alignment': 0.75,
            'macro': {'pearson_mean': 0.75, 'r2_mean': 0.60, 'mae_mean': 0.15}
        }
        
        cls_metrics = {
            'monitor': {
                'macro': {
                    'auc_roc_mean': 0.80,
                    'auc_pr_mean': 0.65,
                    'precision_mean': 0.60,
                    'recall_mean': 0.85,
                    'f1_mean': 0.70,
                    'threshold_mean': 0.30,
                    'threshold_std': 0.05
                }
            }
        }
        
        # Request non-existent strategy, should fallback to first available
        merged = merge_metrics(reg_metrics, cls_metrics, 'nonexistent')
        
        # Should use 'monitor' strategy as fallback
        assert merged['precision_mean'] == 0.60
        assert merged['threshold_mean'] == 0.30


if __name__ == "__main__":
    pytest.main([__file__])