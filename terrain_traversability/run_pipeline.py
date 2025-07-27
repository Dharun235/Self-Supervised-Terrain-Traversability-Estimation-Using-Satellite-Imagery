# terrain_traversability/run_pipeline.py
"""
Complete Pipeline Runner for Terrain Traversability Estimation

This script runs the entire pipeline from data preprocessing to risk assessment.
The pipeline includes:
1. Data preprocessing and alignment
2. SimCLR model training
3. Terrain segmentation
4. Risk estimation and analysis

Author: Terrain Traversability Project
Date: 2025
"""

import os
import sys
import time
import subprocess

def run_command(command, description):
    """Run a command with error handling and progress tracking"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {command}")
    
    start_time = time.time()
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        elapsed_time = time.time() - start_time
        print(f"✅ {description} completed successfully in {elapsed_time:.2f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error:")
        print(f"Error code: {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False

def check_prerequisites():
    """Check if required files and directories exist"""
    print("Checking prerequisites...")
    
    required_files = [
        "data/raw/sentinel_rgb.tiff",
        "data/raw/dem.tif"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\nPlease ensure all required data files are in place before running the pipeline.")
        return False
    
    print("✅ All required files found")
    return True

def main():
    """Run the complete terrain traversability pipeline"""
    print("🚀 TERRAIN TRAVERSABILITY ESTIMATION PIPELINE")
    print("="*60)
    print("This pipeline will:")
    print("1. Preprocess satellite and DEM data")
    print("2. Train SimCLR model for terrain feature learning")
    print("3. Segment terrain using learned features")
    print("4. Assess traversability risk")
    print("5. Generate comprehensive visualizations")
    print("="*60)
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Create necessary directories
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    # Pipeline steps
    steps = [
        ("python preprocess.py", "Data Preprocessing"),
        ("python train_simclr.py", "SimCLR Model Training"),
        ("python terrain_segmentation.py", "Terrain Segmentation"),
        ("python risk_estimation.py", "Risk Assessment"),
        ("python pipeline_summary.py", "Results Summary")
    ]
    
    # Run each step
    start_time = time.time()
    for command, description in steps:
        if not run_command(command, description):
            print(f"\n❌ Pipeline failed at: {description}")
            print("Please check the error messages above and try again.")
            sys.exit(1)
    
    # Pipeline completion
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"Total execution time: {total_time/60:.2f} minutes")
    print("\n📁 Results available in:")
    print("  - outputs/segmentation.npy (Terrain segmentation)")
    print("  - outputs/risk_map.npy (Risk assessment)")
    print("  - outputs/segmentation_vis.png (Segmentation visualization)")
    print("  - outputs/risk_analysis.png (Comprehensive analysis)")
    print("\n📊 Model saved in:")
    print("  - models/simclr.pth (Trained SimCLR model)")
    print("\n💡 Next steps:")
    print("1. Review the visualizations in outputs/")
    print("2. Use risk_map.npy for path planning")
    print("3. Integrate with robot navigation systems")

if __name__ == "__main__":
    main()