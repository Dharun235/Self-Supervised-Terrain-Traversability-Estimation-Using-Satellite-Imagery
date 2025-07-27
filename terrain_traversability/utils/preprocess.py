import rasterio
import numpy as np
import cv2
import os

def load_satellite_image(path):
    with rasterio.open(path) as src:
        image = src.read([4, 3, 2])  # RGB
        image = np.transpose(image, (1, 2, 0))
        image = (image - image.min()) / (image.max() - image.min())
        return (image * 255).astype(np.uint8)

def tile_image(image, tile_size=128):
    tiles = []
    h, w, _ = image.shape
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = image[y:y+tile_size, x:x+tile_size]
            if tile.shape[0] == tile_size and tile.shape[1] == tile_size:
                tiles.append(tile)
    return tiles
