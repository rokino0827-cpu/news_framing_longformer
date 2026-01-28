"""
Data preparation utilities for SV2000 dataset.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import os


def prepare_sv2000_splits(
    input_path: str,
    output_dir: str,
    test_size: float = 0.2,
    valid_size: float = 0.1,
    random_state: int = 42
):
    """
    Split SV2000 dataset into train/valid/test splits.
    
    Args:
        input_path: Path to full dataset
        output_dir: Output directory for splits
        test_size: Proportion for test set
        valid_size: Proportion for validation set
        random_state: Random seed
    """
    print(f"Loading dataset from: {input_path}")
    
    # Load data
    if input_path.endswith('.parquet'):
        df = pd.read_parquet(input_path)
    elif input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path}")
    
    print(f"Total samples: {len(df)}")
    
    # Check required columns
    required_cols = ['article_id', 'title', 'content']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
    
    # Check frame annotation columns
    frame_patterns = ['sv_conflict_', 'sv_human_', 'sv_econ_', 'sv_moral_', 'sv_resp_']
    frame_cols = []
    for pattern in frame_patterns:
        cols = [col for col in df.columns if col.startswith(pattern)]
        frame_cols.extend(cols)
    
    print(f"Found {len(frame_cols)} frame annotation columns")
    
    # Remove rows with missing critical data
    print("Cleaning data...")
    initial_len = len(df)
    
    # Remove rows with missing content
    df = df.dropna(subset=['content'])
    print(f"Removed {initial_len - len(df)} rows with missing content")
    
    # Remove rows with all NaN frame annotations
    if frame_cols:
        df = df.dropna(subset=frame_cols, how='all')
        print(f"Removed {initial_len - len(df)} rows with no frame annotations")
    
    print(f"Final dataset size: {len(df)}")
    
    # Create splits
    print("Creating train/valid/test splits...")
    
    # First split: train+valid vs test
    train_valid, test = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=None  # Could add stratification based on frame presence
    )
    
    # Second split: train vs valid
    valid_size_adjusted = valid_size / (1 - test_size)
    train, valid = train_test_split(
        train_valid,
        test_size=valid_size_adjusted,
        random_state=random_state
    )
    
    print(f"Train: {len(train)} samples ({len(train)/len(df)*100:.1f}%)")
    print(f"Valid: {len(valid)} samples ({len(valid)/len(df)*100:.1f}%)")
    print(f"Test: {len(test)} samples ({len(test)/len(df)*100:.1f}%)")
    
    # Save splits
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'train.parquet')
    valid_path = os.path.join(output_dir, 'valid.parquet')
    test_path = os.path.join(output_dir, 'test.parquet')
    
    train.to_parquet(train_path, index=False)
    valid.to_parquet(valid_path, index=False)
    test.to_parquet(test_path, index=False)
    
    print(f"Saved splits to:")
    print(f"  Train: {train_path}")
    print(f"  Valid: {valid_path}")
    print(f"  Test: {test_path}")
    
    # Create summary
    summary = {
        'total_samples': len(df),
        'train_samples': len(train),
        'valid_samples': len(valid),
        'test_samples': len(test),
        'frame_columns': frame_cols,
        'splits': {
            'test_size': test_size,
            'valid_size': valid_size,
            'random_state': random_state
        }
    }
    
    import json
    with open(os.path.join(output_dir, 'split_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("Data preparation completed!")


def analyze_dataset(input_path: str):
    """
    Analyze SV2000 dataset structure and statistics.
    
    Args:
        input_path: Path to dataset file
    """
    print(f"Analyzing dataset: {input_path}")
    
    # Load data
    if input_path.endswith('.parquet'):
        df = pd.read_parquet(input_path)
    elif input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path}")
    
    print(f"\n=== Dataset Overview ===")
    print(f"Total samples: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    
    # Check text fields
    print(f"\n=== Text Fields ===")
    text_fields = ['title', 'content']
    for field in text_fields:
        if field in df.columns:
            non_null = df[field].notna().sum()
            avg_length = df[field].str.len().mean()
            print(f"{field}: {non_null}/{len(df)} non-null, avg length: {avg_length:.1f}")
        else:
            print(f"{field}: NOT FOUND")
    
    # Check frame annotation columns
    print(f"\n=== Frame Annotations ===")
    frame_patterns = ['sv_conflict_', 'sv_human_', 'sv_econ_', 'sv_moral_', 'sv_resp_']
    
    for pattern in frame_patterns:
        cols = [col for col in df.columns if col.startswith(pattern)]
        if cols:
            print(f"{pattern[:-1]}: {len(cols)} columns")
            for col in cols:
                non_null = df[col].notna().sum()
                if non_null > 0:
                    mean_val = df[col].mean()
                    print(f"  {col}: {non_null} non-null, mean: {mean_val:.3f}")
        else:
            print(f"{pattern[:-1]}: NO COLUMNS FOUND")
    
    # Check for other interesting columns
    print(f"\n=== Other Columns ===")
    other_cols = [col for col in df.columns if not any(col.startswith(p) for p in frame_patterns + ['title', 'content', 'article_id'])]
    for col in other_cols[:10]:  # Show first 10
        print(f"{col}: {df[col].dtype}")
    
    if len(other_cols) > 10:
        print(f"... and {len(other_cols) - 10} more columns")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SV2000 dataset")
    parser.add_argument("--input", type=str, required=True, help="Path to input dataset")
    parser.add_argument("--output", type=str, help="Output directory for splits")
    parser.add_argument("--analyze", action="store_true", help="Only analyze dataset structure")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set proportion")
    parser.add_argument("--valid-size", type=float, default=0.1, help="Validation set proportion")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_dataset(args.input)
    else:
        if not args.output:
            raise ValueError("--output is required when not using --analyze")
        prepare_sv2000_splits(
            args.input, 
            args.output, 
            args.test_size, 
            args.valid_size, 
            args.seed
        )