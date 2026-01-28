# News Framing Longformer - Framework Bias Detection

## Overview
Long-text article-level framework bias detection system with multiple baseline models supporting multi-task learning (regression + classification).

## Project Structure
```
news_framing_longformer/
├── configs/                    # Configuration files
├── data/                      # Data processing modules
├── models/                    # Model definitions
├── training/                  # Training utilities
├── eval/                      # Evaluation metrics
├── scripts/                   # Utility scripts
├── reports/                   # Output reports
├── checkpoints/               # Model checkpoints
├── train.py                   # Training script
├── evaluate.py                # Evaluation script
├── infer.py                   # Inference script
└── README.md
```

## Quick Start

### Environment Setup
```bash
# Create conda environment
conda create -n framing_longformer python=3.10
conda activate framing_longformer

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate evaluate scikit-learn pandas numpy pyarrow pyyaml tqdm wandb
```

### Training
```bash
python train.py --config configs/longformer_sv2000.yaml
```

### Evaluation
```bash
python evaluate.py --config configs/longformer_sv2000.yaml --ckpt checkpoints/run_xxx/best.pt --split test
```

### Inference
```bash
python infer.py --config configs/longformer_sv2000.yaml --ckpt checkpoints/run_xxx/best.pt --input data/unlabeled.csv --output reports/infer_out.jsonl
```

## Models
- **Baseline-A**: Longformer multi-task (regression + classification)
- **Baseline-B**: BigBird multi-task (optional)
- **Baseline-C**: HAT hierarchical transformer (optional)

## Frame Types
- conflict: Conflict framing
- human: Human interest framing  
- econ: Economic framing
- moral: Moral framing
- resp: Responsibility framing