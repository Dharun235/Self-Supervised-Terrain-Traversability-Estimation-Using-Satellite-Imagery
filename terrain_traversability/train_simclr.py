# terrain_traversability/train_simclr.py
"""
Self-Supervised Learning (SimCLR) Training for Terrain Traversability Estimation

This script trains a SimCLR model on satellite imagery to learn terrain-specific features
for autonomous robot navigation and path planning.

Model Architecture:
- Backbone: ResNet-18 encoder
- Projection head: 2-layer MLP (512 → 256 → 128)
- Loss: NT-Xent (Normalized Temperature-scaled Cross Entropy)
- Augmentations: Random crop, horizontal flip, color jitter, grayscale

Training Parameters:
- Patch size: 64x64 pixels
- Stride: 32 pixels (50% overlap)
- Batch size: 128 (GPU) / 64 (CPU)
- Learning rate: 1e-3
- Epochs: 10
- Temperature: 0.5

Author: Terrain Traversability Project
Date: 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import rasterio
import os
import time
from tqdm import tqdm
from utils.simclr import SimCLR, NTXentLoss, get_simclr_augmentations

class SatelliteDataset(Dataset):
    """
    Dataset for satellite imagery patches with quality filtering
    """
    def __init__(self, img_path, patch_size=64, stride=32):
        print(f"Loading satellite image from {img_path}...")
        with rasterio.open(img_path) as src:
            self.img = src.read().transpose(1,2,0)
        
        print(f"Image shape: {self.img.shape}")
        
        self.patches = []
        h, w, _ = self.img.shape
        patch_count = 0
        
        print("Extracting patches with quality filtering...")
        for i in range(0, h-patch_size, stride):
            for j in range(0, w-patch_size, stride):
                patch = self.img[i:i+patch_size, j:j+patch_size, :]
                # Filter out very dark/empty patches
                if np.mean(patch) > 10:
                    self.patches.append(patch)
                    patch_count += 1
        
        print(f"Created {len(self.patches)} quality patches")
        self.transform = get_simclr_augmentations(patch_size)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx].astype(np.uint8)
        patch = self.transform(patch)
        return patch, patch  # SimCLR expects two views

def train_simclr():
    """
    Train SimCLR model on satellite imagery for terrain feature learning
    """
    print("="*60)
    print("SELF-SUPERVISED TERRAIN TRAVERSABILITY ESTIMATION")
    print("="*60)
    print("Training SimCLR model on satellite imagery...")
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Load dataset
    dataset = SatelliteDataset("data/processed/sentinel_rgb.tiff")
    
    # Optimize batch size for device
    batch_size = 128 if torch.cuda.is_available() else 64
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Initialize model
    model = SimCLR(base_model='resnet18', out_dim=128)
    model = model.to(device)
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = NTXentLoss(temperature=0.5)
    
    print(f"\nTraining Configuration:")
    print(f"  Dataset size: {len(dataset)} patches")
    print(f"  Batch size: {batch_size}")
    print(f"  Batches per epoch: {len(loader)}")
    print(f"  Total epochs: 10")
    print(f"  Learning rate: 1e-3")
    print(f"  Temperature: 0.5")
    
    # Training loop
    model.train()
    start_time = time.time()
    
    for epoch in range(10):
        epoch_loss = 0.0
        batch_count = 0
        
        print(f"\nEpoch {epoch+1}/10")
        progress_bar = tqdm(loader, desc=f"Training")
        
        for batch_idx, (x1, x2) in enumerate(progress_bar):
            # Move data to device
            x1, x2 = x1.to(device), x2.to(device)
            
            # Forward pass
            z1, z2 = model(x1), model(x2)
            loss = criterion(z1, z2)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            batch_count += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{epoch_loss/batch_count:.4f}',
                'GPU Mem': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB' if torch.cuda.is_available() else 'N/A'
            })
        
        # Epoch summary
        avg_loss = epoch_loss / batch_count
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), f"models/simclr_epoch_{epoch+1}.pth")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save final model
    torch.save(model.state_dict(), "models/simclr.pth")
    
    # Training summary
    total_time = time.time() - start_time
    print(f"\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Final model saved to: models/simclr.pth")
    print(f"Model size: {os.path.getsize('models/simclr.pth')/1024/1024:.1f} MB")
    print("\nNext steps:")
    print("1. Run terrain_segmentation.py to segment terrain")
    print("2. Run risk_estimation.py to assess traversability")
    print("3. Review results in outputs/ directory")

if __name__ == "__main__":
    train_simclr()