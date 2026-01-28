"""
Evaluation script for news framing longformer models.
"""
import argparse
import yaml
import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from data.dataset import load_sv2000_dataframe, build_labels, SV2000ArticleDataset
from models.multitask import build_model
from eval.metrics import compute_regression_metrics, compute_presence_metrics, merge_metrics


def main_evaluate(cfg_path: str, ckpt_path: str, split: str = "test") -> None:
    """
    Main evaluation function.
    
    Args:
        cfg_path: Path to configuration file
        ckpt_path: Path to model checkpoint
        split: Dataset split to evaluate ('test', 'valid', 'train')
    """
    # Load configuration
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading {split} dataset...")
    if split == "test":
        data_path = cfg['data']['test_path']
    elif split == "valid":
        data_path = cfg['data']['valid_path']
    elif split == "train":
        data_path = cfg['data']['train_path']
    else:
        raise ValueError(f"Unknown split: {split}")
    
    df = load_sv2000_dataframe(data_path)
    
    # Build labels
    print("Building labels...")
    y_reg, y_cls, frame_names = build_labels(
        df,
        cfg['data']['frame_defs'],
        cfg['label']['regression_agg'],
        cfg['label']['normalize'],
        cfg['label']['presence_threshold']
    )
    
    print(f"Frame names: {frame_names}")
    print(f"Evaluation samples: {len(df)}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg['model']['pretrained_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset
    print("Creating dataset...")
    dataset = SV2000ArticleDataset(
        df=df,
        tokenizer=tokenizer,
        y_reg=y_reg,
        y_cls=y_cls,
        text_fields=cfg['data']['text_fields'],
        id_field=cfg['data']['id_field'],
        max_length=cfg['model']['max_length'],
        long_text_strategy=cfg['model']['long_text_strategy'],
        use_title=cfg['model']['use_title']
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Build model
    print("Building model...")
    model = build_model(
        backbone=cfg['model']['backbone'],
        pretrained_name=cfg['model']['pretrained_name'],
        num_frames=len(frame_names),
        dropout=cfg['model']['dropout'],
        global_attention=cfg['model'].get('global_attention', 'cls'),
        num_global_tokens=cfg['model'].get('num_global_tokens', 1)
    )
    
    # Load checkpoint
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Run evaluation
    print("Running evaluation...")
    all_reg_pred = []
    all_reg_true = []
    all_cls_prob = []
    all_cls_true = []
    all_article_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            global_attention_mask = batch.get('global_attention_mask')
            if global_attention_mask is not None:
                global_attention_mask = global_attention_mask.to(device)
            y_reg_batch = batch['y_reg'].to(device)
            y_cls_batch = batch['y_cls'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask
            )
            
            # Collect predictions
            all_reg_pred.append(outputs['reg_pred'].cpu().numpy())
            all_reg_true.append(y_reg_batch.cpu().numpy())
            all_cls_prob.append(outputs['cls_prob'].cpu().numpy())
            all_cls_true.append(y_cls_batch.cpu().numpy())
            all_article_ids.extend(batch['article_id'])
    
    # Concatenate all predictions
    y_reg_pred = np.concatenate(all_reg_pred, axis=0)
    y_reg_true = np.concatenate(all_reg_true, axis=0)
    y_cls_prob = np.concatenate(all_cls_prob, axis=0)
    y_cls_true = np.concatenate(all_cls_true, axis=0)
    
    print(f"Prediction shapes: reg={y_reg_pred.shape}, cls={y_cls_prob.shape}")
    
    # Compute metrics
    print("Computing metrics...")
    reg_metrics = compute_regression_metrics(y_reg_true, y_reg_pred, frame_names)
    cls_metrics = compute_presence_metrics(
        y_cls_true, y_cls_prob, frame_names,
        threshold_strategy=cfg['eval']['threshold_strategy'],
        fixed_threshold=cfg['eval']['fixed_threshold']
    )
    
    metrics = merge_metrics(reg_metrics, cls_metrics)
    
    # Print results
    print(f"\n=== Evaluation Results ({split} set) ===")
    print(f"Overall Alignment: {metrics['overall_alignment']:.4f}")
    print(f"Pearson Mean: {metrics['pearson_mean']:.4f}")
    print(f"R² Mean: {metrics['r2_mean']:.4f}")
    print(f"MAE Mean: {metrics['mae_mean']:.4f}")
    print(f"AUC-ROC Mean: {metrics['auc_roc_mean']:.4f}")
    print(f"AUC-PR Mean: {metrics['auc_pr_mean']:.4f}")
    print(f"Precision Mean: {metrics['precision_mean']:.4f}")
    print(f"Recall Mean: {metrics['recall_mean']:.4f}")
    print(f"F1 Mean: {metrics['f1_mean']:.4f}")
    
    print(f"\n=== Per-Frame Results ===")
    for frame_name in frame_names:
        reg_frame = metrics['regression']['per_frame'][frame_name]
        cls_frame = metrics['classification']['per_frame'][frame_name]
        print(f"{frame_name}:")
        print(f"  Regression - Pearson: {reg_frame['pearson']:.4f}, R²: {reg_frame['r2']:.4f}, MAE: {reg_frame['mae']:.4f}")
        print(f"  Classification - AUC-ROC: {cls_frame['auc_roc']:.4f}, AUC-PR: {cls_frame['auc_pr']:.4f}, F1: {cls_frame['f1']:.4f}")
    
    # Save results
    output_dir = os.path.dirname(ckpt_path)
    results_path = os.path.join(output_dir, f'eval_{split}_metrics.json')
    
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate news framing model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"], 
                       help="Dataset split to evaluate")
    args = parser.parse_args()
    
    main_evaluate(args.config, args.ckpt, args.split)