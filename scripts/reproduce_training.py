#!/usr/bin/env python3
"""
One-command reproducible training script for SV2000 frame monitoring.

This script ensures complete reproducibility by:
1. Setting all random seeds
2. Logging all configuration and environment info
3. Creating timestamped run directories
4. Saving all outputs and checkpoints
5. Generating comprehensive reports

Usage:
    python scripts/reproduce_training.py --config configs/longformer_sv2000.yaml
    python scripts/reproduce_training.py --config configs/longformer_sv2000.yaml --data_dir /path/to/data
    python scripts/reproduce_training.py --config configs/longformer_sv2000.yaml --resume checkpoints/run_20240129_143022/latest.pt
"""

import os
import sys
import argparse
import json
import hashlib
import shutil
from datetime import datetime
from pathlib import Path
import yaml
import torch
import numpy as np
import random
import platform
import subprocess

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from train import main as train_main
from evaluate import main as evaluate_main


def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Enable deterministic algorithms (PyTorch 1.12+)
    try:
        torch.use_deterministic_algorithms(True)
    except:
        print("Warning: torch.use_deterministic_algorithms not available")


def get_system_info():
    """Collect comprehensive system information for reproducibility."""
    info = {
        'timestamp': datetime.now().isoformat(),
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'processor': platform.processor(),
        'architecture': platform.architecture(),
        'hostname': platform.node(),
    }
    
    # PyTorch info
    info['torch_version'] = torch.__version__
    info['cuda_available'] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['cudnn_version'] = torch.backends.cudnn.version()
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        info['gpu_memory'] = [torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())]
    
    # Git info (if available)
    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('ascii').strip()
        git_dirty = len(subprocess.check_output(['git', 'diff', '--name-only']).decode('ascii').strip()) > 0
        info['git'] = {
            'commit_hash': git_hash,
            'branch': git_branch,
            'dirty': git_dirty
        }
    except:
        info['git'] = None
    
    # Package versions
    try:
        import transformers
        info['transformers_version'] = transformers.__version__
    except:
        pass
    
    try:
        import datasets
        info['datasets_version'] = datasets.__version__
    except:
        pass
    
    try:
        import sklearn
        info['sklearn_version'] = sklearn.__version__
    except:
        pass
    
    return info


def create_run_directory(base_dir: str, config: dict) -> Path:
    """Create timestamped run directory with all necessary subdirectories."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    
    run_dir = Path(base_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "reports").mkdir(exist_ok=True)
    (run_dir / "config").mkdir(exist_ok=True)
    
    return run_dir


def save_reproducibility_info(run_dir: Path, config: dict, args: argparse.Namespace):
    """Save all information needed for reproducibility."""
    
    # Save system info
    system_info = get_system_info()
    with open(run_dir / "config" / "system_info.json", 'w') as f:
        json.dump(system_info, f, indent=2)
    
    # Save config with resolved paths
    config_copy = config.copy()
    with open(run_dir / "config" / "config.yaml", 'w') as f:
        yaml.dump(config_copy, f, default_flow_style=False)
    
    # Save command line arguments
    args_dict = vars(args)
    with open(run_dir / "config" / "args.json", 'w') as f:
        json.dump(args_dict, f, indent=2)
    
    # Save requirements.txt snapshot
    req_file = Path(__file__).parent.parent / "requirements.txt"
    if req_file.exists():
        shutil.copy(req_file, run_dir / "config" / "requirements.txt")
    
    # Create data checksums for validation
    data_info = {}
    for split in ['train', 'valid', 'test']:
        data_path = config['data'].get(f'{split}_path')
        if data_path and os.path.exists(data_path):
            # Calculate file hash
            with open(data_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            data_info[f'{split}_hash'] = file_hash
            data_info[f'{split}_size'] = os.path.getsize(data_path)
    
    with open(run_dir / "config" / "data_info.json", 'w') as f:
        json.dump(data_info, f, indent=2)
    
    print(f"Reproducibility info saved to: {run_dir / 'config'}")


def validate_environment():
    """Validate that the environment is set up correctly."""
    
    # Check CUDA availability if expected
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.version.cuda}")
        print(f"✓ GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("⚠ CUDA not available, using CPU")
    
    # Check key packages
    required_packages = ['transformers', 'datasets', 'sklearn', 'pandas', 'numpy']
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} available")
        except ImportError:
            print(f"✗ {package} not available")
            return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Reproducible SV2000 training")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Override data directory in config')
    parser.add_argument('--output_dir', type=str, default='runs',
                       help='Base directory for run outputs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--eval_only', action='store_true',
                       help='Only run evaluation, skip training')
    parser.add_argument('--dry_run', action='store_true',
                       help='Validate setup without running training')
    
    args = parser.parse_args()
    
    # Validate environment
    print("=== Environment Validation ===")
    if not validate_environment():
        print("Environment validation failed!")
        return 1
    
    # Load configuration
    print(f"\n=== Loading Configuration ===")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Config loaded from: {args.config}")
    print(f"Project: {config['project']['name']}")
    print(f"Seed: {config['project']['seed']}")
    
    # Override data directory if specified
    if args.data_dir:
        for split in ['train', 'valid', 'test']:
            if f'{split}_path' in config['data']:
                original_path = config['data'][f'{split}_path']
                filename = os.path.basename(original_path)
                config['data'][f'{split}_path'] = os.path.join(args.data_dir, filename)
        print(f"Data directory overridden to: {args.data_dir}")
    
    # Set reproducibility
    print(f"\n=== Setting Reproducibility ===")
    seed = config['project']['seed']
    set_all_seeds(seed)
    print(f"All seeds set to: {seed}")
    
    # Create run directory
    print(f"\n=== Creating Run Directory ===")
    run_dir = create_run_directory(args.output_dir, config)
    print(f"Run directory: {run_dir}")
    
    # Update config with run directory
    config['project']['output_dir'] = str(run_dir / "reports")
    config['project']['ckpt_dir'] = str(run_dir / "checkpoints")
    
    # Save reproducibility information
    save_reproducibility_info(run_dir, config, args)
    
    if args.dry_run:
        print("\n=== Dry Run Complete ===")
        print(f"Run directory created: {run_dir}")
        print("Use --eval_only to skip training, or remove --dry_run to start training")
        return 0
    
    # Run training or evaluation
    try:
        if not args.eval_only:
            print(f"\n=== Starting Training ===")
            print(f"Output directory: {config['project']['output_dir']}")
            print(f"Checkpoint directory: {config['project']['ckpt_dir']}")
            
            # Save config to temporary file for training script
            temp_config_path = run_dir / "config" / "runtime_config.yaml"
            with open(temp_config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            # Run training
            train_args = argparse.Namespace(
                config=str(temp_config_path),
                resume=args.resume
            )
            train_main(train_args)
            
            print(f"✓ Training completed successfully")
        
        # Run evaluation on test set
        print(f"\n=== Running Evaluation ===")
        
        # Find best checkpoint
        ckpt_dir = Path(config['project']['ckpt_dir'])
        best_ckpt = ckpt_dir / "best.pt"
        
        if not best_ckpt.exists():
            print(f"Warning: Best checkpoint not found at {best_ckpt}")
            # Look for any checkpoint
            ckpt_files = list(ckpt_dir.glob("*.pt"))
            if ckpt_files:
                best_ckpt = ckpt_files[0]
                print(f"Using checkpoint: {best_ckpt}")
            else:
                print("No checkpoints found, skipping evaluation")
                return 1
        
        # Run evaluation
        eval_args = argparse.Namespace(
            config=str(run_dir / "config" / "runtime_config.yaml"),
            ckpt=str(best_ckpt),
            split='test',
            output=str(run_dir / "reports" / "test_results.json")
        )
        evaluate_main(eval_args)
        
        print(f"✓ Evaluation completed successfully")
        
        # Generate summary report
        print(f"\n=== Generating Summary Report ===")
        generate_summary_report(run_dir)
        
        print(f"\n=== Run Complete ===")
        print(f"All outputs saved to: {run_dir}")
        print(f"Summary report: {run_dir / 'reports' / 'summary.md'}")
        
    except Exception as e:
        print(f"\n✗ Error during execution: {e}")
        # Save error info
        error_info = {
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'traceback': str(e.__traceback__) if hasattr(e, '__traceback__') else None
        }
        with open(run_dir / "error.json", 'w') as f:
            json.dump(error_info, f, indent=2)
        return 1
    
    return 0


def generate_summary_report(run_dir: Path):
    """Generate a human-readable summary report."""
    
    summary_lines = []
    summary_lines.append("# SV2000 Frame Monitoring - Run Summary")
    summary_lines.append("")
    summary_lines.append(f"**Run Directory**: `{run_dir.name}`")
    summary_lines.append(f"**Timestamp**: {datetime.now().isoformat()}")
    summary_lines.append("")
    
    # System info
    system_info_file = run_dir / "config" / "system_info.json"
    if system_info_file.exists():
        with open(system_info_file) as f:
            system_info = json.load(f)
        
        summary_lines.append("## System Information")
        summary_lines.append(f"- **Python**: {system_info.get('python_version', 'Unknown')}")
        summary_lines.append(f"- **PyTorch**: {system_info.get('torch_version', 'Unknown')}")
        summary_lines.append(f"- **Platform**: {system_info.get('platform', 'Unknown')}")
        if system_info.get('cuda_available'):
            summary_lines.append(f"- **CUDA**: {system_info.get('cuda_version', 'Unknown')}")
            summary_lines.append(f"- **GPUs**: {len(system_info.get('gpu_names', []))}")
        summary_lines.append("")
    
    # Configuration summary
    config_file = run_dir / "config" / "config.yaml"
    if config_file.exists():
        with open(config_file) as f:
            config = yaml.safe_load(f)
        
        summary_lines.append("## Configuration")
        summary_lines.append(f"- **Model**: {config.get('model', {}).get('backbone', 'Unknown')}")
        summary_lines.append(f"- **Max Length**: {config.get('model', {}).get('max_length', 'Unknown')}")
        summary_lines.append(f"- **Strategy**: {config.get('model', {}).get('long_text_strategy', 'Unknown')}")
        summary_lines.append(f"- **Batch Size**: {config.get('training', {}).get('batch_size', 'Unknown')}")
        summary_lines.append(f"- **Learning Rate**: {config.get('training', {}).get('lr', 'Unknown')}")
        summary_lines.append(f"- **Epochs**: {config.get('training', {}).get('epochs', 'Unknown')}")
        summary_lines.append("")
    
    # Results summary
    results_file = run_dir / "reports" / "test_results.json"
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
        
        summary_lines.append("## Results")
        summary_lines.append("### Regression Metrics")
        summary_lines.append(f"- **Overall Alignment**: {results.get('overall_alignment', 0):.4f}")
        summary_lines.append(f"- **Pearson Mean**: {results.get('pearson_mean', 0):.4f}")
        summary_lines.append(f"- **R² Mean**: {results.get('r2_mean', 0):.4f}")
        summary_lines.append(f"- **MAE Mean**: {results.get('mae_mean', 0):.4f}")
        
        summary_lines.append("### Classification Metrics")
        summary_lines.append(f"- **AUC-ROC Mean**: {results.get('auc_roc_mean', 0):.4f}")
        summary_lines.append(f"- **AUC-PR Mean**: {results.get('auc_pr_mean', 0):.4f}")
        summary_lines.append(f"- **Precision Mean**: {results.get('precision_mean', 0):.4f}")
        summary_lines.append(f"- **Recall Mean**: {results.get('recall_mean', 0):.4f}")
        summary_lines.append(f"- **F1 Mean**: {results.get('f1_mean', 0):.4f}")
        summary_lines.append("")
    
    # File listing
    summary_lines.append("## Generated Files")
    for subdir in ['checkpoints', 'reports', 'config', 'logs']:
        subdir_path = run_dir / subdir
        if subdir_path.exists():
            files = list(subdir_path.glob("*"))
            if files:
                summary_lines.append(f"### {subdir}/")
                for file in sorted(files):
                    size_mb = file.stat().st_size / (1024 * 1024)
                    summary_lines.append(f"- `{file.name}` ({size_mb:.1f} MB)")
                summary_lines.append("")
    
    # Write summary
    summary_file = run_dir / "reports" / "summary.md"
    with open(summary_file, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"Summary report generated: {summary_file}")


if __name__ == "__main__":
    exit(main())