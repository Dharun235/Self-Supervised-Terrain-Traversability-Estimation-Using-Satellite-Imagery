# terrain_traversability/terrain_segmentation.py
import torch
import numpy as np
from sklearn.cluster import KMeans
from utils.simclr import SimCLR, get_simclr_augmentations
import rasterio
import os
from tqdm import tqdm

def extract_features(model_path, img_path, patch_size=64, stride=64, max_patches=10000):
    """
    Extract features from satellite image using trained SimCLR model
    """
    print(f"Loading model from {model_path}...")
    model = SimCLR(base_model='resnet18', out_dim=128)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    
    print(f"Loading image from {img_path}...")
    with rasterio.open(img_path) as src:
        img = src.read().transpose(1,2,0)
    
    print(f"Image shape: {img.shape}")
    
    patches = []
    coords = []
    h, w, _ = img.shape
    patch_count = 0
    
    print("Extracting patches...")
    for i in range(0, h-patch_size, stride):
        for j in range(0, w-patch_size, stride):
            if patch_count >= max_patches:
                break
            patch = img[i:i+patch_size, j:j+patch_size, :]
            if np.mean(patch) > 10:  # Filter out dark patches
                patches.append(patch)
                coords.append((i, j))
                patch_count += 1
        if patch_count >= max_patches:
            break
    
    print(f"Extracted {len(patches)} patches")
    
    transform = get_simclr_augmentations(patch_size)
    features = []
    
    print("Extracting features...")
    with torch.no_grad():
        for patch in tqdm(patches, desc="Processing patches"):
            x = transform(patch.astype(np.uint8)).unsqueeze(0).to(device)
            feat = model.encoder(x).squeeze().cpu().numpy()  # Get encoder features
            features.append(feat)
    
    return np.array(features), coords, h, w

def segment_terrain(features, n_clusters=6):
    """
    Segment terrain using K-means clustering on learned features
    """
    print(f"Clustering features into {n_clusters} terrain types...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    
    # Print cluster statistics
    unique, counts = np.unique(labels, return_counts=True)
    print("Cluster distribution:")
    for cluster_id, count in zip(unique, counts):
        percentage = (count / len(labels)) * 100
        print(f"  Cluster {cluster_id}: {count} patches ({percentage:.1f}%)")
    
    return labels

def create_segmentation_map(labels, coords, h, w, patch_size=64):
    """
    Create full-resolution segmentation map
    """
    print("Creating segmentation map...")
    seg_map = np.zeros((h, w), dtype=np.uint8)
    
    for label, (i, j) in zip(labels, coords):
        seg_map[i:i+patch_size, j:j+patch_size] = label
    
    return seg_map

def save_segmentation_visualization(seg_map, output_path="outputs/segmentation_vis.png"):
    """
    Save segmentation map as a visualization
    """
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        plt.imshow(seg_map, cmap='tab10')
        plt.colorbar(label='Terrain Type')
        plt.title('Terrain Segmentation Map')
        plt.xlabel('Width')
        plt.ylabel('Height')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Segmentation visualization saved to {output_path}")
    except ImportError:
        print("Matplotlib not available, skipping visualization")

if __name__ == "__main__":
    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)
    
    # Use the best available model
    model_paths = [
        "models/simclr_original.pth",  # Original training (if completed)
        "models/simclr.pth",           # CPU training (current best)
        "models/simclr_gpu.pth"        # GPU training
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            print(f"Using model: {path}")
            break
    
    if model_path is None:
        print("No trained model found! Please run training first.")
        exit(1)
    
    # Extract features
    features, coords, h, w = extract_features(
        model_path, 
        "data/processed/sentinel_rgb.tiff",
        patch_size=64,
        stride=64,  # Larger stride for faster processing
        max_patches=8000  # Reasonable number for segmentation
    )
    
    # Segment terrain
    labels = segment_terrain(features, n_clusters=6)
    
    # Create segmentation map
    seg_map = create_segmentation_map(labels, coords, h, w)
    
    # Save results
    np.save("outputs/segmentation.npy", seg_map)
    print("Segmentation map saved to outputs/segmentation.npy")
    
    # Create visualization
    save_segmentation_visualization(seg_map)
    
    print("Terrain segmentation completed!")