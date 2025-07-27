# terrain_traversability/risk_estimation.py
import numpy as np
import rasterio
import os
from utils.dem_utils import compute_slope
import matplotlib.pyplot as plt

def compute_slope_from_dem(dem_path):
    """
    Compute slope from DEM data
    """
    print(f"Loading DEM from {dem_path}...")
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
    
    print(f"DEM shape: {dem.shape}")
    print(f"DEM range: {dem.min():.1f} to {dem.max():.1f} meters")
    
    # Compute slope
    print("Computing slope...")
    slope_map = compute_slope(dem, cellsize=30)  # Assuming 30m resolution
    
    print(f"Slope range: {slope_map.min():.1f}° to {slope_map.max():.1f}°")
    print(f"Mean slope: {np.mean(slope_map):.1f}°")
    
    return slope_map, dem

def analyze_terrain_slope_relationship(seg_map, slope_map):
    """
    Analyze the relationship between terrain types and slope
    """
    print("\nTerrain-Slope Analysis:")
    print("=" * 50)
    
    unique_terrain = np.unique(seg_map)
    
    for terrain_id in unique_terrain:
        if terrain_id == 0:  # Skip background
            continue
        
        terrain_mask = (seg_map == terrain_id)
        terrain_slopes = slope_map[terrain_mask]
        
        if len(terrain_slopes) > 0:
            mean_slope = np.mean(terrain_slopes)
            std_slope = np.std(terrain_slopes)
            max_slope = np.max(terrain_slopes)
            min_slope = np.min(terrain_slopes)
            
            print(f"Terrain Type {terrain_id}:")
            print(f"  Mean slope: {mean_slope:.1f}° ± {std_slope:.1f}°")
            print(f"  Slope range: {min_slope:.1f}° to {max_slope:.1f}°")
            print(f"  Area coverage: {np.sum(terrain_mask)} pixels")

def assign_risk_scores(seg_map, slope_map, slope_thresholds=[10, 20, 30]):
    """
    Assign risk scores based on terrain type and slope
    Risk levels: 0=Low, 1=Medium, 2=High, 3=Very High
    """
    print(f"\nAssigning risk scores with slope thresholds: {slope_thresholds}°")
    
    risk_map = np.zeros_like(seg_map, dtype=np.uint8)
    
    # Risk based on slope
    risk_map[slope_map > slope_thresholds[2]] = 3  # Very high risk (>30°)
    risk_map[(slope_map > slope_thresholds[1]) & (slope_map <= slope_thresholds[2])] = 2  # High risk (20-30°)
    risk_map[(slope_map > slope_thresholds[0]) & (slope_map <= slope_thresholds[1])] = 1  # Medium risk (10-20°)
    
    # Additional risk based on terrain type (some terrain types are inherently more difficult)
    # This is a simplified heuristic - in practice, you'd use domain knowledge
    difficult_terrain_types = [2, 4]  # Example: rocky/rough terrain
    for terrain_id in difficult_terrain_types:
        terrain_mask = (seg_map == terrain_id)
        # Increase risk for difficult terrain types
        risk_map[terrain_mask] = np.minimum(risk_map[terrain_mask] + 1, 3)
    
    return risk_map

def create_risk_visualizations(seg_map, slope_map, risk_map, dem, output_dir="outputs"):
    """
    Create comprehensive visualizations
    """
    print("\nCreating visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original satellite image (if available)
    try:
        with rasterio.open("data/processed/sentinel_rgb.tiff") as src:
            rgb = src.read().transpose(1,2,0)
            # Normalize for display
            rgb_norm = (rgb - rgb.min()) / (rgb.max() - rgb.min())
            axes[0,0].imshow(rgb_norm)
            axes[0,0].set_title("Satellite Image")
            axes[0,0].axis('off')
    except:
        axes[0,0].text(0.5, 0.5, "Satellite Image\n(Not Available)", 
                      ha='center', va='center', transform=axes[0,0].transAxes)
        axes[0,0].set_title("Satellite Image")
    
    # DEM
    im1 = axes[0,1].imshow(dem, cmap='terrain')
    axes[0,1].set_title("Digital Elevation Model")
    plt.colorbar(im1, ax=axes[0,1], label='Elevation (m)')
    axes[0,1].axis('off')
    
    # Slope
    im2 = axes[0,2].imshow(slope_map, cmap='hot')
    axes[0,2].set_title("Slope Map")
    plt.colorbar(im2, ax=axes[0,2], label='Slope (°)')
    axes[0,2].axis('off')
    
    # Terrain Segmentation
    im3 = axes[1,0].imshow(seg_map, cmap='tab10')
    axes[1,0].set_title("Terrain Segmentation")
    plt.colorbar(im3, ax=axes[1,0], label='Terrain Type')
    axes[1,0].axis('off')
    
    # Risk Map
    im4 = axes[1,1].imshow(risk_map, cmap='RdYlGn_r', vmin=0, vmax=3)
    axes[1,1].set_title("Traversability Risk Map")
    plt.colorbar(im4, ax=axes[1,1], label='Risk Level')
    axes[1,1].axis('off')
    
    # Risk Statistics
    risk_levels = ['Low', 'Medium', 'High', 'Very High']
    risk_counts = [np.sum(risk_map == i) for i in range(4)]
    risk_percentages = [count/np.prod(risk_map.shape)*100 for count in risk_counts]
    
    axes[1,2].bar(risk_levels, risk_percentages, color=['green', 'yellow', 'orange', 'red'])
    axes[1,2].set_title("Risk Distribution")
    axes[1,2].set_ylabel("Percentage of Area (%)")
    axes[1,2].tick_params(axis='x', rotation=45)
    
    # Add percentage labels on bars
    for i, (count, pct) in enumerate(zip(risk_counts, risk_percentages)):
        axes[1,2].text(i, pct + 0.5, f'{pct:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/risk_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Risk analysis visualization saved to {output_dir}/risk_analysis.png")

def print_risk_summary(risk_map):
    """
    Print comprehensive risk summary
    """
    print("\n" + "="*50)
    print("TRAVERSABILITY RISK SUMMARY")
    print("="*50)
    
    total_pixels = np.prod(risk_map.shape)
    risk_levels = ['Low', 'Medium', 'High', 'Very High']
    colors = ['🟢', '🟡', '🟠', '🔴']
    
    for i, (level, color) in enumerate(zip(risk_levels, colors)):
        count = np.sum(risk_map == i)
        percentage = (count / total_pixels) * 100
        print(f"{color} {level} Risk: {count:,} pixels ({percentage:.1f}%)")
    
    # Calculate overall traversability score
    safe_area = np.sum(risk_map <= 1)  # Low + Medium risk
    overall_safety = (safe_area / total_pixels) * 100
    
    print(f"\nOverall Traversability Score: {overall_safety:.1f}% safe")
    
    if overall_safety >= 80:
        print("🟢 Area is generally safe for traversal")
    elif overall_safety >= 60:
        print("🟡 Area has moderate traversal challenges")
    elif overall_safety >= 40:
        print("🟠 Area has significant traversal challenges")
    else:
        print("🔴 Area is very challenging for traversal")

if __name__ == "__main__":
    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)
    
    # Load segmentation map
    print("Loading terrain segmentation...")
    seg_map = np.load("outputs/segmentation.npy")
    print(f"Segmentation map shape: {seg_map.shape}")
    
    # Compute slope from DEM
    slope_map, dem = compute_slope_from_dem("data/processed/dem.tiff")
    
    # Analyze terrain-slope relationship
    analyze_terrain_slope_relationship(seg_map, slope_map)
    
    # Assign risk scores
    risk_map = assign_risk_scores(seg_map, slope_map)
    
    # Save results
    np.save("outputs/slope.npy", slope_map)
    np.save("outputs/risk_map.npy", risk_map)
    print("\nResults saved:")
    print("  - outputs/slope.npy")
    print("  - outputs/risk_map.npy")
    
    # Create visualizations
    create_risk_visualizations(seg_map, slope_map, risk_map, dem)
    
    # Print risk summary
    print_risk_summary(risk_map)
    
    print("\nRisk estimation completed!")