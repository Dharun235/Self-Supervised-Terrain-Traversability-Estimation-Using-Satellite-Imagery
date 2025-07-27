import rasterio
import numpy as np
import os
import sys
from pathlib import Path

def check_file_exists(file_path):
    """Check if a file exists and is readable."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"Cannot read file: {file_path}")

def validate_bands(bands_data):
    """Validate that all bands have the same shape."""
    shapes = [band.shape for band in bands_data]
    if len(set(shapes)) > 1:
        raise ValueError(f"All bands must have the same shape. Found shapes: {shapes}")
    return True

def main():
    # Define band paths - Landsat 8 bands for RGB visualization
    band_paths = {
        'B4': r'data/raw/LC08_L1GT_139211_20140920_20200910_02_T2_B4.TIF',  # Red
        'B3': r'data/raw/LC08_L1GT_139211_20140920_20200910_02_T2_B3.TIF',  # Green
        'B2': r'data/raw/LC08_L1GT_139211_20140920_20200910_02_T2_B2.TIF',  # Blue
    }
    
    output_path = 'data/raw/sentinel_rgb.tiff'
    
    try:
        # Check if all input files exist
        print("Checking input files...")
        for band, path in band_paths.items():
            check_file_exists(path)
            print(f"✓ {band}: {path}")
        
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Read bands
        print("\nReading bands...")
        bands = []
        profile = None
        
        # Read bands in RGB order (B4=Red, B3=Green, B2=Blue)
        for band in ['B4', 'B3', 'B2']:
            with rasterio.open(band_paths[band]) as src:
                band_data = src.read(1)
                bands.append(band_data)
                
                if profile is None:
                    profile = src.profile.copy()
                
                print(f"✓ Read {band}: shape {band_data.shape}, dtype {band_data.dtype}")
        
        # Validate bands
        validate_bands(bands)
        
        # Stack bands for RGB (Red, Green, Blue order)
        print("\nStacking bands...")
        rgb = np.stack(bands, axis=0)  # Shape: (3, height, width)
        
        # Update profile for RGB output
        profile.update(
            count=3,
            dtype=rgb.dtype,
            photometric='rgb'
        )
        
        # Write output
        print(f"\nWriting RGB image to: {output_path}")
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(rgb)
        
        print(f"✓ Successfully created RGB image: {output_path}")
        print(f"  Shape: {rgb.shape}")
        print(f"  Data type: {rgb.dtype}")
        print(f"  Value range: [{rgb.min():.2f}, {rgb.max():.2f}]")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except PermissionError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()