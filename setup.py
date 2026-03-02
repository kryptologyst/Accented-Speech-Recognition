#!/usr/bin/env python3
"""Setup script for accented speech recognition project."""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("Setting up Accented Speech Recognition Project")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("Error: Python 3.10 or higher is required")
        sys.exit(1)
    
    print(f"Python version: {sys.version}")
    
    # Install dependencies
    commands = [
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install -e .", "Installing package in development mode"),
        ("pip install -e .[dev]", "Installing development dependencies"),
    ]
    
    success_count = 0
    for command, description in commands:
        if run_command(command, description):
            success_count += 1
    
    # Create necessary directories
    directories = [
        "data/raw",
        "data/processed", 
        "checkpoints",
        "outputs",
        "assets",
        "logs"
    ]
    
    print("\nCreating directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Setup pre-commit hooks (optional)
    if Path(".pre-commit-config.yaml").exists():
        if run_command("pre-commit install", "Installing pre-commit hooks"):
            print("✓ Pre-commit hooks installed")
        else:
            print("! Pre-commit hooks installation failed (optional)")
    
    print("\n" + "=" * 50)
    print("Setup completed!")
    print(f"Successfully completed {success_count}/{len(commands)} installation steps")
    
    if success_count == len(commands):
        print("\nNext steps:")
        print("1. Run tests: pytest tests/")
        print("2. Start training: python scripts/train.py")
        print("3. Launch demo: streamlit run demo/streamlit_app.py")
        print("4. Read the README.md for more information")
    else:
        print("\nSome installation steps failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
