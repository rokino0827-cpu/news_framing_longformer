"""
Inference script for news framing longformer models.
"""
import argparse
import yaml
import os
import json
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from data.dataset import SV2000ArticleDataset
from models.multitask import build_model


def main_infer(cfg_path: str, ckpt_path: str, input_path: str, output_path: str) -> None:
    """
    Main inference function.
    
    Args:
        cfg_path: Path to configuration file
        ckpt_path: Path to model checkpoint
        input_path: Path to input data file
        output_path: Path to output file
    """
    # Load configuration
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading input data: {input_path}")
    if input_path.endswith('.parquet'):
        df = pd.read_parquet(input_path)
    elif input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path}")
    
    print(f"Input samples: {len(df)}")
    
    # Get frame names from config
    frame_names = list(cfg['data']['frame_defs'].keys())
    print(f"Frame names: {frame_names}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg['model']['pretrained_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dummy labels for dataset (not used in inference)
    import numpy as np
    dummy_y_reg = np.zeros((len(df), len(frame_names)), dtype=np.float32)
    dummy_y_cls = np.zeros((len(df), len(frame_names)), dtype=np.float32)
    
    # Create dataset
    print("Creating dataset...")
    dataset = SV2000ArticleDataset(
        df=df,
        tokenizer=tokenizer,
        y_reg=dummy_y_reg,
        y_cls=dummy_y_cls,
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
    
    # Run inference
    print("Running inference...")
    all_reg_pred = []
    all_cls_prob = []
    all_article_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            global_attention_mask = batch.get('global_attention_mask')
            if global_attention_mask is not None:
                global_attention_mask = global_attention_mask.to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask
            )
            
            # Collect predictions
            all_reg_pred.append(outputs['reg_pred'].cpu().numpy())
            all_cls_prob.append(outputs['cls_prob'].cpu().numpy())
            all_article_ids.extend(batch['article_id'])
    
    # Concatenate all predictions
    reg_pred = np.concatenate(all_reg_pred, axis=0)
    cls_prob = np.concatenate(all_cls_prob, axis=0)
    
    print(f"Prediction shapes: reg={reg_pred.shape}, cls={cls_prob.shape}")
    
    # Create results
    results = []
    for i, article_id in enumerate(all_article_ids):
        result = {
            'article_id': article_id,
            'frame_scores': {},
            'frame_probabilities': {}
        }
        
        for j, frame_name in enumerate(frame_names):
            result['frame_scores'][frame_name] = float(reg_pred[i, j])
            result['frame_probabilities'][frame_name] = float(cls_prob[i, j])
        
        # Add overall frame dashboard
        result['dashboard'] = {
            'strongest_frame': frame_names[np.argmax(reg_pred[i])],
            'strongest_frame_score': float(np.max(reg_pred[i])),
            'most_likely_present': frame_names[np.argmax(cls_prob[i])],
            'most_likely_present_prob': float(np.max(cls_prob[i])),
            'avg_intensity': float(np.mean(reg_pred[i])),
            'avg_presence_prob': float(np.mean(cls_prob[i]))
        }
        
        results.append(result)
    
    # Save results
    print(f"Saving results to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if output_path.endswith('.jsonl'):
        with open(output_path, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
    elif output_path.endswith('.json'):
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    elif output_path.endswith('.csv'):
        # Create flat CSV format
        csv_data = []
        for result in results:
            row = {'article_id': result['article_id']}
            
            # Add frame scores
            for frame_name in frame_names:
                row[f'{frame_name}_intensity'] = result['frame_scores'][frame_name]
                row[f'{frame_name}_probability'] = result['frame_probabilities'][frame_name]
            
            # Add dashboard metrics
            row.update(result['dashboard'])
            csv_data.append(row)
        
        pd.DataFrame(csv_data).to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {output_path}")
    
    print(f"Inference completed! Processed {len(results)} articles.")
    
    # Print sample results
    print(f"\n=== Sample Results ===")
    for i, result in enumerate(results[:3]):
        print(f"Article {result['article_id']}:")
        print(f"  Strongest frame: {result['dashboard']['strongest_frame']} "
              f"(intensity: {result['dashboard']['strongest_frame_score']:.3f})")
        print(f"  Most likely present: {result['dashboard']['most_likely_present']} "
              f"(prob: {result['dashboard']['most_likely_present_prob']:.3f})")
        print(f"  Average intensity: {result['dashboard']['avg_intensity']:.3f}")
        print(f"  Average presence prob: {result['dashboard']['avg_presence_prob']:.3f}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with news framing model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--input", type=str, required=True, help="Path to input data file")
    parser.add_argument("--output", type=str, required=True, help="Path to output file")
    args = parser.parse_args()
    
    main_infer(args.config, args.ckpt, args.input, args.output)