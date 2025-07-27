import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

def print_pipeline_summary():
    """
    Print a comprehensive summary of the terrain traversability pipeline results
    """
    print("="*70)
    print("TERRAIN TRAVERSABILITY ESTIMATION PIPELINE - FINAL SUMMARY")
    print("="*70)
    print(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check outputs
    outputs = {
        "Segmentation Map": "outputs/segmentation.npy",
        "Slope Map": "outputs/slope.npy", 
        "Risk Map": "outputs/risk_map.npy",
        "Segmentation Visualization": "outputs/segmentation_vis.png",
        "Risk Analysis Visualization": "outputs/risk_analysis.png"
    }
    
    print("📁 GENERATED OUTPUTS:")
    print("-" * 40)
    for name, path in outputs.items():
        if os.path.exists(path):
            size = os.path.getsize(path)
            if size > 1024*1024:
                size_str = f"{size/(1024*1024):.1f} MB"
            else:
                size_str = f"{size/1024:.1f} KB"
            print(f"✅ {name}: {size_str}")
        else:
            print(f"❌ {name}: Not found")
    
    print()
    
    # Load and analyze results
    try:
        seg_map = np.load("outputs/segmentation.npy")
        risk_map = np.load("outputs/risk_map.npy")
        
        print("🔍 TERRAIN SEGMENTATION RESULTS:")
        print("-" * 40)
        unique_terrain, counts = np.unique(seg_map, return_counts=True)
        total_pixels = np.prod(seg_map.shape)
        
        for terrain_id, count in zip(unique_terrain, counts):
            if terrain_id == 0:  # Background
                continue
            percentage = (count / total_pixels) * 100
            print(f"Terrain Type {terrain_id}: {count:,} pixels ({percentage:.1f}%)")
        
        print()
        
        print("⚠️  TRAVERSABILITY RISK ASSESSMENT:")
        print("-" * 40)
        risk_levels = ['Low', 'Medium', 'High', 'Very High']
        colors = ['🟢', '🟡', '🟠', '🔴']
        
        for i, (level, color) in enumerate(zip(risk_levels, colors)):
            count = np.sum(risk_map == i)
            percentage = (count / total_pixels) * 100
            print(f"{color} {level} Risk: {count:,} pixels ({percentage:.1f}%)")
        
        # Overall safety score
        safe_area = np.sum(risk_map <= 1)  # Low + Medium risk
        overall_safety = (safe_area / total_pixels) * 100
        
        print(f"\n🎯 OVERALL TRAVERSABILITY SCORE: {overall_safety:.1f}% safe")
        
        if overall_safety >= 80:
            print("🟢 RECOMMENDATION: Area is generally safe for traversal")
        elif overall_safety >= 60:
            print("🟡 RECOMMENDATION: Area has moderate traversal challenges")
        elif overall_safety >= 40:
            print("🟠 RECOMMENDATION: Area has significant traversal challenges")
        else:
            print("🔴 RECOMMENDATION: Area is very challenging for traversal")
        
        print()
        
        print("📊 TECHNICAL DETAILS:")
        print("-" * 40)
        print(f"Image Resolution: {seg_map.shape[0]} x {seg_map.shape[1]} pixels")
        print(f"Total Area: {total_pixels:,} pixels")
        print(f"Terrain Types Identified: {len(unique_terrain) - 1}")  # Exclude background
        print(f"Risk Levels: 4 (Low, Medium, High, Very High)")
        
        print()
        
        print("🎨 VISUALIZATIONS CREATED:")
        print("-" * 40)
        print("• Terrain Segmentation Map (segmentation_vis.png)")
        print("• Comprehensive Risk Analysis (risk_analysis.png)")
        print("  - Satellite Image")
        print("  - Digital Elevation Model")
        print("  - Slope Map")
        print("  - Terrain Segmentation")
        print("  - Risk Map")
        print("  - Risk Distribution Statistics")
        
        print()
        
        print("💡 NEXT STEPS:")
        print("-" * 40)
        print("1. Review the visualizations in the outputs/ directory")
        print("2. Use the risk_map.npy for path planning algorithms")
        print("3. Integrate with robot navigation systems")
        print("4. Validate results with ground truth data")
        print("5. Fine-tune parameters based on specific robot capabilities")
        
        print()
        print("="*70)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        
    except Exception as e:
        print(f"❌ Error loading results: {e}")

if __name__ == "__main__":
    print_pipeline_summary() 