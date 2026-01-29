# News Framing Longformer - SV2000 Framework Bias Detection

## Overview
Production-ready long-text article-level framework bias detection system implementing the SV2000 (Semetko & Valkenburg, 2000) framework. Supports multiple transformer architectures with comprehensive multi-task learning (regression + classification) for monitoring news framing patterns.

**Key Features:**
- **Long-text processing**: Handles articles averaging 3600+ words with configurable strategies
- **Multi-task learning**: Simultaneous regression (frame intensity) and classification (frame presence)
- **Dual threshold strategies**: Optimized for both monitoring (high recall) and alerting (high precision)
- **Complete reproducibility**: Deterministic training with comprehensive logging
- **Production monitoring**: Confidence estimation and calibrated predictions

## Quick Start

### Environment Setup
```bash
# Create conda environment
conda create -n framing_longformer python=3.10
conda activate framing_longformer

# Install dependencies (exact versions for reproducibility)
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121

# For CPU-only installation:
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
```

### One-Command Reproducible Training
```bash
# Complete training with full reproducibility
python scripts/reproduce_training.py --config configs/longformer_sv2000.yaml

# Dry run to validate setup
python scripts/reproduce_training.py --config configs/longformer_sv2000.yaml --dry_run

# Resume from checkpoint
python scripts/reproduce_training.py --config configs/longformer_sv2000.yaml --resume checkpoints/run_xxx/latest.pt
```

### Demo and Validation
```bash
# Create demo data (5 synthetic articles with known patterns)
python demo/create_demo_data.py

# Validate data processing pipeline
python demo/validate_demo.py

# Run demo training
python scripts/reproduce_training.py --config demo/demo_config.yaml
```

## Project Structure
```
news_framing_longformer/
├── configs/                    # Configuration files with explicit parameters
│   ├── longformer_sv2000.yaml # Production config with full documentation
│   └── bigbird_sv2000.yaml    # Alternative architecture config
├── data/                      # Data processing modules
│   ├── dataset.py             # Dataset classes and label construction
│   ├── text_utils.py          # Long-text processing utilities
│   └── label_construction.md  # Complete SV2000 label mapping documentation
├── models/                    # Model definitions
│   └── multitask.py          # Multi-task transformer architectures
├── training/                  # Training utilities
│   ├── trainer.py            # Training loop with logging
│   └── losses.py             # Multi-task loss functions
├── eval/                      # Evaluation metrics
│   └── metrics.py            # Dual-threshold evaluation system
├── scripts/                   # Utility scripts
│   ├── reproduce_training.py # One-command reproducible training
│   ├── data_preparation.py   # Data preprocessing pipeline
│   └── run_experiments.py    # Batch experiment runner
├── demo/                      # Demo data and validation
│   ├── create_demo_data.py   # Generate synthetic test data
│   └── validate_demo.py      # Validate processing pipeline
├── tests/                     # Unit tests
│   ├── test_data_processing.py
│   └── test_metrics.py
├── .github/workflows/         # CI/CD pipeline
│   └── ci.yml                # Automated testing and validation
├── reports/                   # Output reports (created during training)
├── checkpoints/              # Model checkpoints (created during training)
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
├── infer.py                  # Inference script
├── requirements.txt          # Exact dependency versions
└── README.md
```

## Long-Text Processing Strategies

### Critical Configuration Parameters
All long-text processing parameters are **explicitly documented** in config files:

```yaml
model:
  max_length: 4096                      # Tokenizer limit (Longformer: 4096)
  long_text_strategy: "sliding_window_pool"  # truncate_head_tail|sliding_window_pool|hierarchical_attention
  chunk_size: 512                       # Chunk size for sliding window (tokens)
  stride: 256                           # Stride for overlapping chunks (50% overlap)
  chunk_aggregation: "attention"        # mean|max|attention aggregation across chunks
  regression_aggregation: "attention"   # How to aggregate regression scores
  classification_aggregation: "max"     # How to aggregate existence probabilities
```

### Supported Strategies
1. **Truncate Head-Tail**: Keep beginning and end portions (preserves title + conclusion)
2. **Sliding Window**: Process overlapping chunks with aggregation (recommended for 3600+ word articles)
3. **Hierarchical Attention**: Paragraph-level encoding with document-level aggregation (future)

## SV2000 Framework Implementation

### Frame Definitions
- **Conflict**: Disagreement between parties, groups, or individuals (4 items)
- **Human Interest**: Personal stories, emotional angles, human consequences (5 items)
- **Economic**: Financial impacts, costs, economic consequences (3 items)
- **Moral**: Moral messages, religious tenets, social prescriptions (3 items)
- **Responsibility**: Government/individual responsibility, action efficacy (5 items)

### Label Construction Pipeline
1. **Item-level validation**: Handle missing values, validate ranges
2. **Frame-level aggregation**: Combine multiple items per frame (mean/weighted)
3. **Normalization**: MinMax [0,1] or Z-score standardization
4. **Binary conversion**: Threshold-based presence detection

Complete documentation: [`data/label_construction.md`](data/label_construction.md)

## Dual Threshold Strategy for Production Monitoring

### Three Threshold Strategies
```yaml
eval:
  threshold_strategies:
    monitor_thresholds:      # High recall (catch most cases)
      target_recall: 0.85
      use_case: "Continuous monitoring, early warning"
    alert_thresholds:        # High precision (minimize false positives)
      target_precision: 0.8
      use_case: "Human review triggers, actionable alerts"
    balanced_thresholds:     # F1 optimization
      use_case: "Research evaluation, model comparison"
```

### Usage Examples
```python
# Monitoring mode: catch 85% of biased content
if prediction_prob >= monitor_threshold:
    flag_for_review()

# Alerting mode: only high-confidence cases
if prediction_prob >= alert_threshold:
    send_alert_to_human_reviewer()
```

## Models and Baselines

### Supported Architectures
- **Longformer** (Baseline-A): Efficient attention for long sequences
- **BigBird** (Baseline-B): Sparse attention patterns
- **HAT** (Baseline-C): Hierarchical attention transformer (planned)

### Multi-task Learning
- **Regression head**: Frame intensity scores [0,1]
- **Classification head**: Frame presence probabilities
- **Configurable loss weights**: Balance regression vs classification importance

## Evaluation Metrics

### Regression Metrics
- **Overall Alignment**: Mean Pearson correlation across frames
- **Per-frame Pearson**: Correlation between predicted and true intensities
- **R² Score**: Coefficient of determination
- **MAE**: Mean absolute error

### Classification Metrics
- **AUC-ROC**: Area under ROC curve (threshold-independent)
- **AUC-PR**: Area under precision-recall curve (better for imbalanced data)
- **Precision/Recall/F1**: At optimized thresholds per strategy
- **Threshold Analysis**: Performance across different threshold values

## Reproducibility Guarantees

### Deterministic Training
- **Fixed seeds**: All random number generators seeded
- **Deterministic algorithms**: PyTorch deterministic mode enabled
- **Environment logging**: Complete system and package version capture
- **Data checksums**: MD5 validation of input data

### Comprehensive Logging
Every training run generates:
- **System info**: Hardware, software, Git commit details
- **Configuration snapshot**: All parameters with resolved paths
- **Data validation**: File sizes, checksums, label statistics
- **Training logs**: Loss curves, metric progression, timing
- **Model checkpoints**: Best and latest model states
- **Evaluation reports**: JSON + Markdown summaries

### Run Directory Structure
```
runs/run_20240129_143022/
├── checkpoints/          # Model checkpoints
├── logs/                # Training logs
├── reports/             # Evaluation reports and plots
└── config/              # Complete reproducibility information
    ├── system_info.json # Hardware/software environment
    ├── config.yaml      # Runtime configuration
    ├── args.json        # Command line arguments
    ├── requirements.txt # Dependency snapshot
    └── data_info.json   # Data validation checksums
```

## Testing and Validation

### Automated Testing
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Test specific components
pytest tests/test_data_processing.py
pytest tests/test_metrics.py
```

### Continuous Integration
- **Code quality**: Black formatting, Flake8 linting, MyPy type checking
- **Unit tests**: Data processing, metrics computation, model forward pass
- **Integration tests**: End-to-end pipeline validation
- **Reproducibility tests**: Training setup and run directory creation

## Performance Benchmarks

### Expected Performance (Longformer baseline)
- **Overall Alignment**: 0.65-0.75 (Pearson correlation)
- **AUC-ROC**: 0.75-0.85 (frame presence detection)
- **Training time**: ~2-4 hours on V100 (full dataset)
- **Memory usage**: ~16GB GPU memory (batch_size=1, max_length=4096)

### Optimization Tips
- **Gradient accumulation**: Increase effective batch size without memory increase
- **Mixed precision**: Use `bf16` for 30-40% speedup
- **Chunk processing**: Reduce `max_length` for memory-constrained environments

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce `batch_size`, `max_length`, or enable gradient checkpointing
2. **Low frame detection**: Adjust `presence_threshold` or check label balance
3. **Poor alignment scores**: Verify label construction and normalization
4. **Inconsistent results**: Check random seed settings and deterministic flags

### Debug Commands
```bash
# Validate data processing
python -c "from data.dataset import load_sv2000_dataframe; print(load_sv2000_dataframe('data/train.parquet').info())"

# Check model forward pass
python -c "from models.multitask import LongformerMultiTask; print('Model loads successfully')"

# Validate configuration
python scripts/reproduce_training.py --config configs/longformer_sv2000.yaml --dry_run
```

## Citation

If you use this system in your research, please cite:

```bibtex
@software{news_framing_longformer,
  title={News Framing Longformer: SV2000 Framework Bias Detection},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[your-repo]/news_framing_longformer}
}

@article{semetko2000framing,
  title={Framing European politics: A content analysis of press and television news},
  author={Semetko, Holli A and Valkenburg, Patti M},
  journal={Journal of communication},
  volume={50},
  number={2},
  pages={93--109},
  year={2000},
  publisher={Oxford University Press}
}
```

## License

[Specify your license here]

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest tests/`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## Support

- **Issues**: [GitHub Issues](https://github.com/[your-repo]/news_framing_longformer/issues)
- **Documentation**: See [`data/label_construction.md`](data/label_construction.md) for detailed SV2000 implementation
- **Demo**: Run `python demo/create_demo_data.py` for quick validation