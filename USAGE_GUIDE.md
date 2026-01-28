# News Framing Longformer - Usage Guide

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n framing_longformer python=3.10
conda activate framing_longformer

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate evaluate scikit-learn pandas numpy pyarrow pyyaml tqdm wandb scipy

# Or use the setup script
python scripts/setup_environment.py
```

### 2. Data Preparation

```bash
# Analyze your dataset structure
python scripts/data_preparation.py --input /path/to/your/sv2000_data.parquet --analyze

# Create train/valid/test splits
python scripts/data_preparation.py --input /path/to/your/sv2000_data.parquet --output data/
```

### 3. Training

```bash
# Train Longformer baseline
python train.py --config configs/longformer_sv2000.yaml

# Train BigBird comparison
python train.py --config configs/bigbird_sv2000.yaml
```

### 4. Evaluation

```bash
# Evaluate on test set
python evaluate.py --config configs/longformer_sv2000.yaml --ckpt checkpoints/run_xxx/best.pt --split test

# Evaluate on validation set
python evaluate.py --config configs/longformer_sv2000.yaml --ckpt checkpoints/run_xxx/best.pt --split valid
```

### 5. Inference

```bash
# Run inference on new articles
python infer.py --config configs/longformer_sv2000.yaml --ckpt checkpoints/run_xxx/best.pt --input data/unlabeled.csv --output reports/predictions.jsonl
```

## Configuration

### Key Configuration Options

#### Model Settings
```yaml
model:
  backbone: "longformer"  # longformer|bigbird|hat
  pretrained_name: "allenai/longformer-base-4096"
  max_length: 4096
  use_title: true
  long_text_strategy: "truncate_head_tail"  # truncate_head_tail|sliding_window_pool
  dropout: 0.1
```

#### Training Settings
```yaml
training:
  epochs: 10
  batch_size: 1
  grad_accum_steps: 8  # Effective batch size = batch_size * grad_accum_steps
  lr: 2e-5
  weight_decay: 0.01
  mixed_precision: "bf16"  # fp16|bf16|no
```

#### Loss Configuration
```yaml
loss:
  reg_loss: "huber"  # mse|huber
  cls_loss: "bce"    # bce|focal
  reg_weight: 1.0
  cls_weight: 1.0
  class_balance:
    enabled: true
    pos_weight_strategy: "inverse_freq"  # inverse_freq|sqrt_inv_freq|none
```

### Frame Definitions

Update the frame definitions in your config to match your dataset:

```yaml
data:
  frame_defs:
    conflict:
      - "sv_conflict_q1_reflects_disagreement"
      - "sv_conflict_q2_disagreement_between_parties"
      # ... add your conflict frame columns
    human:
      - "sv_human_q1_human_example_or_face"
      - "sv_human_q2_human_story_or_experience"
      # ... add your human interest frame columns
    # ... define other frames
```

## Advanced Usage

### Running Multiple Experiments

```bash
# Create and run all experiment configurations
python scripts/run_experiments.py --experiments all

# Run specific experiments
python scripts/run_experiments.py --experiments longformer_baseline bigbird_baseline

# Just create configs without running
python scripts/run_experiments.py --create-configs
```

### Custom Model Architectures

To add a new backbone model, modify `models/multitask.py`:

```python
def build_custom_model(pretrained_name: str, num_frames: int, dropout: float) -> FrameMonitoringModel:
    # Your custom model implementation
    pass
```

### Custom Metrics

Add new evaluation metrics in `eval/metrics.py`:

```python
def compute_custom_metrics(y_true, y_pred, frame_names):
    # Your custom metric implementation
    pass
```

## Output Structure

### Training Outputs
```
reports/run_YYYYMMDD_HHMMSS/
├── results.json      # Training results and metrics
├── config.yaml       # Configuration used
└── summary.txt       # Human-readable summary

checkpoints/run_YYYYMMDD_HHMMSS/
└── best.pt          # Best model checkpoint
```

### Evaluation Outputs
```
checkpoints/run_YYYYMMDD_HHMMSS/
└── eval_test_metrics.json  # Detailed evaluation metrics
```

### Inference Outputs
```
reports/
├── predictions.jsonl     # JSONL format with detailed predictions
├── predictions.json      # JSON format
└── predictions.csv       # CSV format with flattened results
```

## Metrics Explanation

### Regression Metrics (Frame Intensity)
- **Pearson Correlation**: Measures linear relationship between predicted and true intensities
- **R² Score**: Coefficient of determination, proportion of variance explained
- **MAE**: Mean Absolute Error, average absolute difference
- **Overall Alignment**: Average Pearson correlation across all frames

### Classification Metrics (Frame Presence)
- **AUC-ROC**: Area under ROC curve, discrimination ability
- **AUC-PR**: Area under Precision-Recall curve, especially important for imbalanced classes
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   - Reduce `batch_size` to 1
   - Increase `grad_accum_steps` to maintain effective batch size
   - Reduce `max_length`
   - Use `mixed_precision: "bf16"`

2. **Slow Training**
   - Use `mixed_precision: "bf16"`
   - Increase `batch_size` if memory allows
   - Use `num_workers > 0` in DataLoader (but start with 0)

3. **Poor Performance**
   - Check data quality and preprocessing
   - Adjust loss weights (`reg_weight`, `cls_weight`)
   - Try different learning rates
   - Enable class balancing

4. **Model Not Loading**
   - Ensure config matches the one used during training
   - Check checkpoint path
   - Verify model architecture compatibility

### Performance Optimization

1. **Memory Optimization**
   ```yaml
   training:
     batch_size: 1
     grad_accum_steps: 16
     mixed_precision: "bf16"
   ```

2. **Speed Optimization**
   ```yaml
   model:
     max_length: 2048  # Reduce if articles are shorter
   training:
     batch_size: 2     # Increase if memory allows
   ```

3. **Quality Optimization**
   ```yaml
   loss:
     class_balance:
       enabled: true
   training:
     early_stopping:
       patience: 5     # Increase for more training
   ```

## Data Format Requirements

### Input Data Format
Your dataset should have these columns:
- `article_id`: Unique identifier (str/int)
- `title`: Article title (str, optional)
- `content`: Article content (str, required)
- Frame annotation columns matching your `frame_defs` configuration

### Example Data Structure
```csv
article_id,title,content,sv_conflict_q1_reflects_disagreement,sv_human_q1_human_example_or_face,...
1,"Article Title","Article content here...",0.8,0.2,...
2,"Another Title","More content...",0.1,0.9,...
```

## Model Architecture Details

### Longformer Multi-task Model
- **Input**: Title + Content (up to 4096 tokens)
- **Encoder**: Longformer with global attention on CLS token
- **Heads**: 
  - Regression head: Linear(hidden_size → num_frames) for intensity scores
  - Classification head: Linear(hidden_size → num_frames) + Sigmoid for presence probabilities
- **Loss**: Combined Huber loss (regression) + BCE loss (classification)

### BigBird Multi-task Model
- Same architecture as Longformer but with BigBird encoder
- Supports same max_length (4096 tokens)
- More memory efficient for very long documents

### HAT (Hierarchical Attention Transformer)
- Sentence-level encoding followed by document-level aggregation
- Better for extremely long documents or memory-constrained environments
- Currently simplified implementation (can be enhanced)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{news_framing_longformer,
  title={News Framing Longformer: Long-text Article-level Framework Bias Detection},
  author={Research Team},
  year={2024},
  url={https://github.com/your-repo/news-framing-longformer}
}
```