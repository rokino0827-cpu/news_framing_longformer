"""
Unit tests for data processing components.
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import (
    load_sv2000_dataframe, 
    build_labels, 
    SV2000ArticleDataset
)
from data.text_utils import (
    join_title_content,
    truncate_head_tail_tokens,
    sliding_window_encode
)


class TestDataLoading:
    """Test data loading functions."""
    
    def test_load_csv(self):
        """Test loading CSV files."""
        # Create temporary CSV
        data = {
            'article_id': ['test_001', 'test_002'],
            'title': ['Test Title 1', 'Test Title 2'],
            'content': ['Test content 1', 'Test content 2']
        }
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            loaded_df = load_sv2000_dataframe(temp_path)
            assert len(loaded_df) == 2
            assert 'article_id' in loaded_df.columns
            assert loaded_df.iloc[0]['title'] == 'Test Title 1'
        finally:
            os.unlink(temp_path)
    
    def test_load_parquet(self):
        """Test loading Parquet files."""
        # Create temporary Parquet
        data = {
            'article_id': ['test_001', 'test_002'],
            'title': ['Test Title 1', 'Test Title 2'],
            'content': ['Test content 1', 'Test content 2']
        }
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            df.to_parquet(f.name, index=False)
            temp_path = f.name
        
        try:
            loaded_df = load_sv2000_dataframe(temp_path)
            assert len(loaded_df) == 2
            assert 'article_id' in loaded_df.columns
        finally:
            os.unlink(temp_path)
    
    def test_unsupported_format(self):
        """Test error handling for unsupported file formats."""
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_sv2000_dataframe("test.txt")


class TestLabelConstruction:
    """Test label construction functions."""
    
    def create_test_dataframe(self):
        """Create test dataframe with SV2000 columns."""
        data = {
            'article_id': ['test_001', 'test_002', 'test_003'],
            'sv_conflict_q1_reflects_disagreement': [0.8, 0.2, 0.5],
            'sv_conflict_q2_disagreement_between_parties': [0.9, 0.1, 0.6],
            'sv_human_q1_human_example_or_face': [0.3, 0.9, 0.4],
            'sv_human_q2_human_story_or_experience': [0.2, 0.8, 0.3],
            'sv_econ_q1_financial_losses_gains': [0.7, 0.1, 0.8],
        }
        return pd.DataFrame(data)
    
    def test_build_labels_mean_aggregation(self):
        """Test label construction with mean aggregation."""
        df = self.create_test_dataframe()
        
        frame_defs = {
            'conflict': ['sv_conflict_q1_reflects_disagreement', 'sv_conflict_q2_disagreement_between_parties'],
            'human': ['sv_human_q1_human_example_or_face', 'sv_human_q2_human_story_or_experience'],
            'econ': ['sv_econ_q1_financial_losses_gains']
        }
        
        y_reg, y_cls, frame_names = build_labels(
            df, frame_defs, 'mean', 'none', 0.5
        )
        
        # Check shapes
        assert y_reg.shape == (3, 3)
        assert y_cls.shape == (3, 3)
        assert len(frame_names) == 3
        
        # Check specific values
        # Article 1: conflict = (0.8 + 0.9) / 2 = 0.85
        assert abs(y_reg[0, 0] - 0.85) < 1e-6
        
        # Article 2: human = (0.9 + 0.8) / 2 = 0.85
        assert abs(y_reg[1, 1] - 0.85) < 1e-6
        
        # Check binary labels (threshold 0.5)
        assert y_cls[0, 0] == 1  # conflict 0.85 > 0.5
        assert y_cls[1, 0] == 0  # conflict 0.15 < 0.5
    
    def test_build_labels_normalization(self):
        """Test label normalization."""
        df = self.create_test_dataframe()
        
        frame_defs = {
            'conflict': ['sv_conflict_q1_reflects_disagreement', 'sv_conflict_q2_disagreement_between_parties']
        }
        
        # Test minmax normalization
        y_reg, _, _ = build_labels(df, frame_defs, 'mean', 'minmax_0_1', 0.5)
        
        # Check that values are in [0, 1] range
        assert np.all(y_reg >= 0)
        assert np.all(y_reg <= 1)
        
        # Check that min and max are actually 0 and 1
        assert np.min(y_reg[:, 0]) == 0.0
        assert np.max(y_reg[:, 0]) == 1.0
    
    def test_build_labels_missing_columns(self):
        """Test handling of missing columns."""
        df = self.create_test_dataframe()
        
        frame_defs = {
            'conflict': ['sv_conflict_q1_reflects_disagreement', 'nonexistent_column']
        }
        
        # Should not raise error, but should warn
        y_reg, y_cls, frame_names = build_labels(df, frame_defs, 'mean', 'none', 0.5)
        
        # Should still work with existing columns
        assert y_reg.shape[1] == 1
        assert not np.isnan(y_reg).any()


class TestTextUtils:
    """Test text processing utilities."""
    
    def test_join_title_content(self):
        """Test title and content joining."""
        # Test with both title and content
        result = join_title_content("Test Title", "Test content", True)
        assert result == "Test Title [SEP] Test content"
        
        # Test without title
        result = join_title_content("", "Test content", True)
        assert result == "Test content"
        
        # Test with use_title=False
        result = join_title_content("Test Title", "Test content", False)
        assert result == "Test content"
        
        # Test with None title
        result = join_title_content(None, "Test content", True)
        assert result == "Test content"
    
    def test_truncate_head_tail_tokens(self):
        """Test head-tail token truncation."""
        from transformers import LongformerTokenizer
        
        tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        
        # Create long token sequence
        long_tokens = [tokenizer.cls_token_id] + list(range(100, 200)) + [tokenizer.sep_token_id]
        
        # Truncate to 50 tokens
        truncated = truncate_head_tail_tokens(long_tokens, 50, tokenizer)
        
        # Should be exactly 50 tokens
        assert len(truncated) == 50
        
        # Should start with CLS
        assert truncated[0] == tokenizer.cls_token_id
        
        # Should end with SEP
        assert truncated[-1] == tokenizer.sep_token_id
    
    def test_sliding_window_encode(self):
        """Test sliding window encoding."""
        from transformers import LongformerTokenizer
        
        tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        
        # Create long text
        text = "This is a test sentence. " * 100
        
        # Encode with sliding window
        result = sliding_window_encode(text, tokenizer, window=50, stride=25)
        
        # Should have multiple chunks
        assert result['num_chunks'] > 1
        assert len(result['chunks']) == result['num_chunks']
        
        # Each chunk should have correct shape
        for chunk in result['chunks']:
            assert chunk['input_ids'].shape[1] == 50


class TestDataset:
    """Test dataset class."""
    
    def create_test_data(self):
        """Create test data for dataset."""
        df = pd.DataFrame({
            'article_id': ['test_001', 'test_002'],
            'title': ['Short title', 'Another title'],
            'content': ['Short content.', 'Longer content with more text.']
        })
        
        y_reg = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]])
        y_cls = np.array([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
        
        return df, y_reg, y_cls
    
    def test_dataset_creation(self):
        """Test dataset creation and basic functionality."""
        from transformers import LongformerTokenizer
        
        tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        df, y_reg, y_cls = self.create_test_data()
        
        dataset = SV2000ArticleDataset(
            df=df,
            tokenizer=tokenizer,
            y_reg=y_reg,
            y_cls=y_cls,
            text_fields=['title', 'content'],
            id_field='article_id',
            max_length=128,
            long_text_strategy='truncate_head_tail',
            use_title=True
        )
        
        # Test length
        assert len(dataset) == 2
        
        # Test item access
        item = dataset[0]
        
        # Check required keys
        required_keys = ['article_id', 'input_ids', 'attention_mask', 'global_attention_mask', 'y_reg', 'y_cls', 'raw_text']
        for key in required_keys:
            assert key in item
        
        # Check tensor shapes
        assert item['input_ids'].shape[0] == 128
        assert item['attention_mask'].shape[0] == 128
        assert item['global_attention_mask'].shape[0] == 128
        assert item['y_reg'].shape[0] == 5
        assert item['y_cls'].shape[0] == 5
        
        # Check that CLS token gets global attention
        assert item['global_attention_mask'][0] == 1
    
    def test_dataset_without_title(self):
        """Test dataset with use_title=False."""
        from transformers import LongformerTokenizer
        
        tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        df, y_reg, y_cls = self.create_test_data()
        
        dataset = SV2000ArticleDataset(
            df=df,
            tokenizer=tokenizer,
            y_reg=y_reg,
            y_cls=y_cls,
            text_fields=['title', 'content'],
            id_field='article_id',
            max_length=128,
            use_title=False
        )
        
        item = dataset[0]
        
        # Raw text should not contain title
        assert 'Short title' not in item['raw_text']
        assert 'Short content.' in item['raw_text']


if __name__ == "__main__":
    pytest.main([__file__])