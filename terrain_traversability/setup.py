#!/usr/bin/env python3
"""
Setup script for Terrain Traversability Estimation Project

This script helps set up the project environment and check dependencies.
"""

import os
import sys
import subprocess
import importlib

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'torch', 'torchvision', 'numpy', 'sklearn', 'cv2', 
        'rasterio', 'skimage', 'PIL', 'matplotlib', 'tqdm'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install them using: pip install -r requirements.txt")
        return False
    
    return True

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return True
        else:
            print("⚠️  No GPU detected - will use CPU (slower)")
            return True
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        "data/raw",
        "data/processed", 
        "models",
        "outputs",
        "notebooks"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def main():
    """Main setup function"""
    print("🔧 TERRAIN TRAVERSABILITY PROJECT SETUP")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        print("\n💡 To install dependencies, run:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    print("\n🖥️  Checking GPU...")
    check_gpu()
    
    print("\n📁 Creating directories...")
    create_directories()
    
    print("\n✅ Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Place your satellite image as 'data/raw/sentinel_rgb.tiff'")
    print("2. Place your DEM as 'data/raw/dem.tif'")
    print("3. Run the pipeline: python run_pipeline.py")
    print("\n📚 For more information, see README.md")

if __name__ == "__main__":
    main() 