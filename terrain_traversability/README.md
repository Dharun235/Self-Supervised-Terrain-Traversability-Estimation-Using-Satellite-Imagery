# Self-Supervised Terrain Traversability Estimation Using Satellite Imagery

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive pipeline for autonomous robot navigation that uses self-supervised learning to estimate terrain traversability from satellite imagery and elevation data. This project enables robots to understand terrain characteristics and plan safe navigation paths without requiring labeled training data.

## 🎯 Problem Statement

Autonomous robots operating in outdoor environments need to understand terrain characteristics to plan safe and efficient navigation paths. Traditional approaches require extensive labeled datasets, which are expensive and time-consuming to create. This project addresses this challenge by:

- **Self-supervised learning**: Uses contrastive learning (SimCLR) to learn terrain features without labels
- **Multi-modal fusion**: Combines satellite imagery with elevation data (DEM) for comprehensive terrain analysis
- **Real-time assessment**: Provides traversability risk maps for immediate path planning decisions

## 🏗️ Architecture Overview

### Model Architecture
- **Backbone**: ResNet-18 encoder (pretrained=False)
- **Projection Head**: 2-layer MLP (512 → 256 → 128 dimensions)
- **Loss Function**: NT-Xent (Normalized Temperature-scaled Cross Entropy)
- **Temperature**: 0.5

### Training Configuration
- **Input**: RGB satellite imagery (Landsat 8 bands 4,3,2)
- **Patch Size**: 64×64 pixels
- **Stride**: 32 pixels (50% overlap)
- **Batch Size**: 128 (GPU) / 64 (CPU)
- **Learning Rate**: 1e-3
- **Epochs**: 10
- **Augmentations**: Random crop, horizontal flip, color jitter, grayscale

## 📊 Results

### Model Performance
- **Feature Diversity**: High-quality terrain-specific feature representations
- **Training Time**: ~15-20 minutes with GPU acceleration
- **Model Size**: 43.4 MB
- **Dataset**: 41,374 quality patches from 7,721×7,601 pixel image

### Traversability Assessment
- **Overall Safety Score**: 98.8% safe for traversal
- **Risk Distribution**:
  - 🟢 Low Risk: 79.6% of area
  - 🟡 Medium Risk: 19.2% of area
  - 🟠 High Risk: 1.1% of area
  - 🔴 Very High Risk: 0.0% of area

### Terrain Segmentation
- **Terrain Types Identified**: 5 distinct terrain classifications
- **Segmentation Quality**: High-resolution mapping with clear boundaries
- **Coverage**: Complete area analysis with quality filtering

## 🚀 Quick Start

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt

# For GPU acceleration (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Data Preparation
1. **Satellite Imagery**: Place your RGB satellite image as `data/raw/sentinel_rgb.tiff`
2. **DEM Data**: Place your elevation data as `data/raw/dem.tif`
3. **For Landsat Data**: Use `stack_images.py` to create RGB composite

### Run Complete Pipeline
```bash
# Run the entire pipeline
python run_pipeline.py
```

### Individual Steps
```bash
# 1. Preprocess data (align satellite and DEM)
python preprocess.py

# 2. Train SimCLR model
python train_simclr.py

# 3. Segment terrain
python terrain_segmentation.py

# 4. Estimate traversability risk
python risk_estimation.py

# 5. View results summary
python pipeline_summary.py
```

## 📁 Project Structure

```
terrain_traversability/
├── data/
│   ├── raw/                    # Input data
│   │   ├── sentinel_rgb.tiff   # Satellite imagery
│   │   └── dem.tif            # Digital elevation model
│   └── processed/              # Aligned and processed data
├── models/
│   └── simclr.pth             # Trained SimCLR model
├── outputs/                    # Results and visualizations
│   ├── segmentation.npy       # Terrain segmentation map
│   ├── slope.npy              # Slope analysis
│   ├── risk_map.npy           # Traversability risk map
│   ├── segmentation_vis.png   # Segmentation visualization
│   └── risk_analysis.png      # Comprehensive analysis
├── utils/                      # Utility modules
│   ├── simclr.py              # SimCLR model and loss
│   ├── dem_utils.py           # DEM processing utilities
│   └── gradcam_utils.py       # Grad-CAM implementation
├── notebooks/                  # Jupyter notebooks for exploration
├── train_simclr.py            # SimCLR training script
├── terrain_segmentation.py    # Terrain segmentation
├── risk_estimation.py         # Risk assessment
├── preprocess.py              # Data preprocessing
├── stack_images.py            # Landsat band stacking
├── run_pipeline.py            # Complete pipeline runner
├── pipeline_summary.py        # Results summary
└── requirements.txt           # Python dependencies
```

## 🔧 Technical Details

### Self-Supervised Learning (SimCLR)
The model learns terrain features through contrastive learning:
1. **Data Augmentation**: Creates two views of each image patch
2. **Feature Extraction**: ResNet-18 encoder extracts 512-dimensional features
3. **Projection**: MLP projects features to 128-dimensional contrastive space
4. **Contrastive Loss**: Maximizes similarity between different views of same patch

### Terrain Segmentation
- **Clustering**: K-means clustering on learned features
- **Quality Filtering**: Removes dark/empty patches
- **Resolution**: Full-resolution segmentation mapping

### Risk Assessment
- **Slope Analysis**: Computes slope from DEM using finite differences
- **Multi-factor Scoring**: Combines terrain type and slope information
- **Risk Levels**: 4 levels (Low, Medium, High, Very High)

## 📈 Performance Metrics

### Training Performance
- **GPU Acceleration**: 10-20x speedup with CUDA
- **Memory Efficiency**: Optimized batch processing
- **Convergence**: Stable training with progress monitoring

### Model Quality
- **Feature Diversity**: High-quality terrain-specific representations
- **Generalization**: Robust to different terrain types
- **Scalability**: Handles large satellite images efficiently

## 🎨 Visualizations

The pipeline generates comprehensive visualizations:
- **Terrain Segmentation Map**: Color-coded terrain types
- **Risk Analysis Dashboard**: Multi-panel analysis including:
  - Satellite imagery
  - Digital elevation model
  - Slope map
  - Terrain segmentation
  - Risk map
  - Risk distribution statistics

## 🔬 Research Applications

This project enables:
- **Autonomous Navigation**: Safe path planning for ground robots
- **Terrain Analysis**: Understanding landscape characteristics
- **Risk Assessment**: Identifying challenging areas
- **Environmental Monitoring**: Large-scale terrain analysis

## 🛠️ Customization

### Model Parameters
```python
# In train_simclr.py
patch_size = 64          # Patch size for training
stride = 32              # Stride between patches
batch_size = 128         # Batch size (GPU) / 64 (CPU)
learning_rate = 1e-3     # Learning rate
epochs = 10              # Number of training epochs
temperature = 0.5        # NT-Xent temperature
```

### Risk Thresholds
```python
# In risk_estimation.py
slope_thresholds = [10, 20, 30]  # Slope thresholds in degrees
difficult_terrain = [2, 4]       # Terrain types with higher risk
```

## 📋 Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
scikit-learn>=1.0.0
opencv-python>=4.5.0
rasterio>=1.2.0
scikit-image>=0.18.0
Pillow>=8.0.0
matplotlib>=3.5.0
tqdm>=4.60.0
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **SimCLR**: Self-supervised learning framework
- **Landsat 8**: Satellite imagery data
- **PyTorch**: Deep learning framework
- **Rasterio**: Geospatial data processing

## 📞 Contact

For questions and support, please open an issue on GitHub.

---

**Note**: This project is designed for research and educational purposes. Always validate results with ground truth data for production applications.