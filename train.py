"""
Training script for news framing longformer models.
"""
import argparse
import yaml
import os
import json
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data.dataset import load_sv2000_dataframe, build_labels, SV2000ArticleDataset
from models.multitask import build_model
from training.trainer import fit


def main_train(cfg_path: str) -> None:
    """
    Main training function.
    
    Args:
        cfg_path: Path to configuration file
    """
    # Load configuration
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Set random seed
    torch.manual_seed(cfg['project']['seed'])
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg['project']['output_dir'], f"run_{timestamp}")
    ckpt_dir = os.path.join(cfg['project']['ckpt_dir'], f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Load data
    print("Loading datasets...")
    train_df = load_sv2000_dataframe(cfg['data']['train_path'])
    valid_df = load_sv2000_dataframe(cfg['data']['valid_path'])
    
    # Build labels
    print("Building labels...")
    train_y_reg, train_y_cls, frame_names = build_labels(
        train_df,
        cfg['data']['frame_defs'],
        cfg['label']['regression_agg'],
        cfg['label']['normalize'],
        cfg['label']['presence_threshold']
    )
    
    valid_y_reg, valid_y_cls, _ = build_labels(
        valid_df,
        cfg['data']['frame_defs'],
        cfg['label']['regression_agg'],
        cfg['label']['normalize'],
        cfg['label']['presence_threshold']
    )
    
    print(f"Frame names: {frame_names}")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(valid_df)}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg['model']['pretrained_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = SV2000ArticleDataset(
        df=train_df,
        tokenizer=tokenizer,
        y_reg=train_y_reg,
        y_cls=train_y_cls,
        text_fields=cfg['data']['text_fields'],
        id_field=cfg['data']['id_field'],
        max_length=cfg['model']['max_length'],
        long_text_strategy=cfg['model']['long_text_strategy'],
        use_title=cfg['model']['use_title']
    )
    
    valid_dataset = SV2000ArticleDataset(
        df=valid_df,
        tokenizer=tokenizer,
        y_reg=valid_y_reg,
        y_cls=valid_y_cls,
        text_fields=cfg['data']['text_fields'],
        id_field=cfg['data']['id_field'],
        max_length=cfg['model']['max_length'],
        long_text_strategy=cfg['model']['long_text_strategy'],
        use_title=cfg['model']['use_title']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True if device.type == 'cuda' else False
    )
    
    valid_loader = DataLoader(
        valid_dataset,
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
    
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("Starting training...")
    state = fit(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        cfg=cfg,
        frame_names=frame_names,
        device=device,
        save_dir=ckpt_dir
    )
    
    # Save final results
    results = {
        'best_metric': state.best_metric,
        'best_epoch': state.epoch,
        'frame_names': frame_names,
        'config': cfg,
        'timestamp': timestamp
    }
    
    # Save results
    with open(os.path.join(run_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Copy config
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f, indent=2)
    
    # Create summary
    summary = f"""Training Summary
================
Project: {cfg['project']['name']}
Timestamp: {timestamp}
Best Metric: {state.best_metric:.4f}
Best Epoch: {state.epoch}
Frame Names: {', '.join(frame_names)}
Model: {cfg['model']['backbone']} ({cfg['model']['pretrained_name']})
Training Samples: {len(train_df)}
Validation Samples: {len(valid_df)}
"""
    
    with open(os.path.join(run_dir, 'summary.txt'), 'w') as f:
        f.write(summary)
    
    print(f"\nTraining completed!")
    print(f"Results saved to: {run_dir}")
    print(f"Best model saved to: {state.best_path}")
    print(f"Best {cfg['training']['early_stopping']['metric']}: {state.best_metric:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train news framing model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    main_train(args.config)