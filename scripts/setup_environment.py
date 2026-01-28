"""
Environment setup script for news framing longformer project.
"""
import subprocess
import sys
import os


def run_command(command):
    """Run shell command and print output."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    print(result.stdout)
    return True


def setup_environment():
    """Setup conda environment and install dependencies."""
    
    print("Setting up news framing longformer environment...")
    
    # Create conda environment
    env_name = "framing_longformer"
    print(f"\n1. Creating conda environment: {env_name}")
    
    create_env_cmd = f"conda create -n {env_name} python=3.10 -y"
    if not run_command(create_env_cmd):
        print("Failed to create conda environment")
        return False
    
    # Activate environment and install PyTorch
    print("\n2. Installing PyTorch with CUDA support...")
    torch_cmd = f"conda run -n {env_name} pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    if not run_command(torch_cmd):
        print("Failed to install PyTorch")
        return False
    
    # Install other dependencies
    print("\n3. Installing other dependencies...")
    deps = [
        "transformers",
        "datasets", 
        "accelerate",
        "evaluate",
        "scikit-learn",
        "pandas",
        "numpy",
        "pyarrow",
        "pyyaml",
        "tqdm",
        "wandb",
        "scipy"
    ]
    
    deps_cmd = f"conda run -n {env_name} pip install {' '.join(deps)}"
    if not run_command(deps_cmd):
        print("Failed to install dependencies")
        return False
    
    print(f"\n✅ Environment setup complete!")
    print(f"To activate the environment, run: conda activate {env_name}")
    print(f"Then you can run training with: python train.py --config configs/longformer_sv2000.yaml")
    
    return True


if __name__ == "__main__":
    setup_environment()