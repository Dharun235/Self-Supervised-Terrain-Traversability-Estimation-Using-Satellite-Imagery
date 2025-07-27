# terrain_traversability/preprocess.py
import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from skimage.transform import resize
import cv2


def read_raster(path):
    """
    Reads a raster (GeoTIFF) file and returns the array and profile.
    Works for both DEM and 3-band RGB images (from Sentinel-2 or Landsat).
    """
    with rasterio.open(path) as src:
        arr = src.read(out_dtype='float32')
        profile = src.profile
    return arr, profile


def save_raster(path, arr, profile):
    """
    Saves a numpy array as a raster (GeoTIFF) file with the given profile.
    """
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(arr)


def align_and_crop(sat_path, dem_path, out_sat, out_dem):
    """
    Aligns and crops the DEM to match the satellite RGB image.
    The satellite image can be from Sentinel-2 or Landsat (must be 3-band RGB GeoTIFF).
    Saves the processed files to the specified output paths.
    """
    sat, sat_prof = read_raster(sat_path)
    dem, dem_prof = read_raster(dem_path)
    # Resize DEM to match satellite shape
    dem_resized = resize(dem[0], sat.shape[1:], preserve_range=True, anti_aliasing=True)
    dem_resized = dem_resized[np.newaxis, ...]
    # Save processed files
    save_raster(out_sat, sat, sat_prof)
    dem_prof.update({'count': 1, 'height': sat.shape[1], 'width': sat.shape[2]})
    save_raster(out_dem, dem_resized, dem_prof)


if __name__ == "__main__":
    # The input RGB image can be from Sentinel-2 or Landsat (must be 3-band RGB GeoTIFF named sentinel_rgb.tiff)
    sat_path = "data/raw/sentinel_rgb.tiff"
    dem_path = "data/raw/dem.tif"  # Place your DEM here
    out_sat = "data/processed/sentinel_rgb.tiff"
    out_dem = "data/processed/dem.tiff"
    align_and_crop(sat_path, dem_path, out_sat, out_dem)