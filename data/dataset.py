"""
Data loading and dataset classes for SV2000 frame monitoring.
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Union
from transformers import PreTrainedTokenizer
from .text_utils import join_title_content, truncate_head_tail_tokens


def load_sv2000_dataframe(path: str) -> pd.DataFrame:
    """
    Load SV2000 dataset from CSV or Parquet file.
    
    Args:
        path: Path to CSV or Parquet file
        
    Returns:
        DataFrame with article data
    """
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    elif path.endswith('.csv'):
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")


def build_labels(
    df: pd.DataFrame,
    frame_defs: Dict[str, List[str]],
    regression_agg: str,
    normalize: str,
    presence_threshold: float
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build regression and classification labels from frame definitions.
    
    Args:
        df: Input dataframe
        frame_defs: Dictionary mapping frame names to column lists
        regression_agg: Aggregation method ('mean' or 'weighted_mean')
        normalize: Normalization method ('none', 'minmax_0_1', 'zscore')
        presence_threshold: Threshold for binary classification
        
    Returns:
        y_reg: Regression labels [N, F] float32
        y_cls: Classification labels [N, F] int64
        frame_names: List of frame names
    """
    frame_names = list(frame_defs.keys())
    n_samples = len(df)
    n_frames = len(frame_names)
    
    y_reg = np.zeros((n_samples, n_frames), dtype=np.float32)
    
    for i, frame_name in enumerate(frame_names):
        columns = frame_defs[frame_name]
        # Check which columns exist in the dataframe
        existing_columns = [col for col in columns if col in df.columns]
        
        if not existing_columns:
            print(f"Warning: No columns found for frame {frame_name}")
            continue
            
        frame_data = df[existing_columns].values
        
        if regression_agg == "mean":
            y_reg[:, i] = np.nanmean(frame_data, axis=1)
        elif regression_agg == "weighted_mean":
            # Simple weighted mean (equal weights for now)
            y_reg[:, i] = np.nanmean(frame_data, axis=1)
        else:
            raise ValueError(f"Unknown aggregation method: {regression_agg}")
    
    # Handle NaN values
    y_reg = np.nan_to_num(y_reg, nan=0.0)
    
    # Normalize regression labels
    if normalize == "minmax_0_1":
        for i in range(n_frames):
            col_min, col_max = y_reg[:, i].min(), y_reg[:, i].max()
            if col_max > col_min:
                y_reg[:, i] = (y_reg[:, i] - col_min) / (col_max - col_min)
    elif normalize == "zscore":
        for i in range(n_frames):
            col_mean, col_std = y_reg[:, i].mean(), y_reg[:, i].std()
            if col_std > 0:
                y_reg[:, i] = (y_reg[:, i] - col_mean) / col_std
    elif normalize == "none":
        pass
    else:
        raise ValueError(f"Unknown normalization method: {normalize}")
    
    # Build classification labels
    y_cls = (y_reg >= presence_threshold).astype(np.int64)
    
    return y_reg, y_cls, frame_names


class SV2000ArticleDataset(Dataset):
    """
    Dataset class for SV2000 articles with frame labels.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        y_reg: np.ndarray,
        y_cls: np.ndarray,
        text_fields: List[str],
        id_field: str,
        max_length: int,
        long_text_strategy: str = "truncate_head_tail",
        use_title: bool = True
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.y_reg = torch.FloatTensor(y_reg)
        self.y_cls = torch.FloatTensor(y_cls.astype(np.float32))  # BCE needs float
        self.text_fields = text_fields
        self.id_field = id_field
        self.max_length = max_length
        self.long_text_strategy = long_text_strategy
        self.use_title = use_title
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        
        # Extract text fields
        title = row.get('title', '') if 'title' in self.text_fields else ''
        content = row.get('content', '') if 'content' in self.text_fields else ''
        
        # Join title and content
        text = join_title_content(title, content, self.use_title)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=False,  # We'll handle truncation manually
            padding=False,
            return_tensors=None
        )
        
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        
        # Handle long text
        if len(input_ids) > self.max_length:
            if self.long_text_strategy == "truncate_head_tail":
                input_ids = truncate_head_tail_tokens(input_ids, self.max_length, self.tokenizer)
                attention_mask = [1] * len(input_ids)
            else:
                # Simple truncation fallback
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
        
        # Pad to max_length
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids.extend([self.tokenizer.pad_token_id] * padding_length)
            attention_mask.extend([0] * padding_length)
        
        # Create global attention mask for Longformer
        global_attention_mask = [0] * self.max_length
        global_attention_mask[0] = 1  # CLS token gets global attention
        
        item = {
            'article_id': row[self.id_field],
            'input_ids': torch.LongTensor(input_ids),
            'attention_mask': torch.LongTensor(attention_mask),
            'global_attention_mask': torch.LongTensor(global_attention_mask),
            'y_reg': self.y_reg[idx],
            'y_cls': self.y_cls[idx],
            'raw_text': text
        }
        
        return item