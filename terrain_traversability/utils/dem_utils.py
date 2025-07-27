# terrain_traversability/utils/dem_utils.py
import numpy as np

def compute_slope(dem, cellsize=30):
    # Simple finite difference slope calculation
    dzdx = np.gradient(dem, axis=1) / cellsize
    dzdy = np.gradient(dem, axis=0) / cellsize
    slope = np.sqrt(dzdx**2 + dzdy**2)
    slope_deg = np.degrees(np.arctan(slope))
    return slope_deg