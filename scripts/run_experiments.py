"""
Script to run multiple experiments with different configurations.
"""
import os
import subprocess
import yaml
import argparse
from datetime import datetime


def run_experiment(config_path: str, experiment_name: str):
    """
    Run a single experiment.
    
    Args:
        config_path: Path to configuration file
        experiment_name: Name of the experiment
    """
    print(f"\n{'='*50}")
    print(f"Running experiment: {experiment_name}")
    print(f"Config: {config_path}")
    print(f"{'='*50}")
    
    # Run training
    train_cmd = f"python train.py --config {config_path}"
    print(f"Command: {train_cmd}")
    
    result = subprocess.run(train_cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Training completed successfully for {experiment_name}")
        print("Training output:")
        print(result.stdout[-1000:])  # Last 1000 characters
    else:
        print(f"❌ Training failed for {experiment_name}")
        print("Error output:")
        print(result.stderr)
        return False
    
    return True


def create_experiment_configs():
    """
    Create different experimental configurations.
    """
    base_config_path = "configs/longformer_sv2000.yaml"
    
    # Load base config
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    experiments = []
    
    # Experiment 1: Longformer baseline
    exp1_config = base_config.copy()
    exp1_config['project']['name'] = "sv2000_longformer_baseline"
    experiments.append(("longformer_baseline", exp1_config))
    
    # Experiment 2: BigBird comparison
    exp2_config = base_config.copy()
    exp2_config['project']['name'] = "sv2000_bigbird_baseline"
    exp2_config['model']['backbone'] = "bigbird"
    exp2_config['model']['pretrained_name'] = "google/bigbird-roberta-base"
    experiments.append(("bigbird_baseline", exp2_config))
    
    # Experiment 3: Longformer with different loss weights
    exp3_config = base_config.copy()
    exp3_config['project']['name'] = "sv2000_longformer_balanced"
    exp3_config['loss']['reg_weight'] = 0.7
    exp3_config['loss']['cls_weight'] = 1.3
    experiments.append(("longformer_balanced", exp3_config))
    
    # Experiment 4: Longformer with higher learning rate
    exp4_config = base_config.copy()
    exp4_config['project']['name'] = "sv2000_longformer_high_lr"
    exp4_config['training']['lr'] = 3e-5
    experiments.append(("longformer_high_lr", exp4_config))
    
    # Experiment 5: Longformer without title
    exp5_config = base_config.copy()
    exp5_config['project']['name'] = "sv2000_longformer_no_title"
    exp5_config['model']['use_title'] = False
    experiments.append(("longformer_no_title", exp5_config))
    
    # Save experiment configs
    os.makedirs("configs/experiments", exist_ok=True)
    
    config_paths = []
    for exp_name, exp_config in experiments:
        config_path = f"configs/experiments/{exp_name}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(exp_config, f, indent=2)
        config_paths.append((exp_name, config_path))
    
    return config_paths


def main():
    """
    Main function to run experiments.
    """
    parser = argparse.ArgumentParser(description="Run multiple experiments")
    parser.add_argument("--experiments", nargs="+", 
                       choices=["longformer_baseline", "bigbird_baseline", "longformer_balanced", 
                               "longformer_high_lr", "longformer_no_title", "all"],
                       default=["all"], help="Experiments to run")
    parser.add_argument("--create-configs", action="store_true", 
                       help="Create experiment configs only")
    
    args = parser.parse_args()
    
    # Create experiment configurations
    print("Creating experiment configurations...")
    config_paths = create_experiment_configs()
    print(f"Created {len(config_paths)} experiment configs")
    
    if args.create_configs:
        print("Experiment configs created. Exiting.")
        return
    
    # Filter experiments to run
    if "all" in args.experiments:
        experiments_to_run = config_paths
    else:
        experiments_to_run = [(name, path) for name, path in config_paths 
                             if name in args.experiments]
    
    print(f"\nRunning {len(experiments_to_run)} experiments...")
    
    # Run experiments
    results = {}
    start_time = datetime.now()
    
    for exp_name, config_path in experiments_to_run:
        exp_start = datetime.now()
        success = run_experiment(config_path, exp_name)
        exp_end = datetime.now()
        
        results[exp_name] = {
            'success': success,
            'duration': str(exp_end - exp_start),
            'config_path': config_path
        }
    
    end_time = datetime.now()
    total_duration = end_time - start_time
    
    # Print summary
    print(f"\n{'='*50}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*50}")
    print(f"Total duration: {total_duration}")
    print(f"Experiments run: {len(experiments_to_run)}")
    
    successful = sum(1 for r in results.values() if r['success'])
    failed = len(results) - successful
    
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    print(f"\nDetailed results:")
    for exp_name, result in results.items():
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        print(f"  {exp_name}: {status} ({result['duration']})")
    
    # Save results summary
    import json
    summary_path = f"reports/experiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("reports", exist_ok=True)
    
    with open(summary_path, 'w') as f:
        json.dump({
            'total_duration': str(total_duration),
            'experiments': results,
            'summary': {
                'total': len(results),
                'successful': successful,
                'failed': failed
            }
        }, f, indent=2)
    
    print(f"\nExperiment summary saved to: {summary_path}")


if __name__ == "__main__":
    main()