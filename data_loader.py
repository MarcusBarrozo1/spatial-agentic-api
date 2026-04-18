"""
Module: data_loader
Description: Extracts multi-band raster data and vector ground truth, 
             stacking and generating synchronous X and Y tensors for U-Net.
"""

import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio.mask import mask
from rasterio.features import rasterize

def extract_training_tensors(raster_paths, vector_path, patch_size=256):
    """
    Reads geographic data, crops multiple bands, stacks them, normalizes, 
    and generates patches for deep learning.
    
    Args:
        raster_paths (list of str): Paths to the satellite image bands (e.g., ['B2.tif', 'B3.tif']).
        vector_path (str): Path to the shapefile with ground truth polygons.
        patch_size (int): Dimension of the output square patches.
        
    Returns:
        tuple: (X_tensor, Y_tensor) ready for TensorFlow training.
    """
    
    # 1. LOAD VECTOR AND ALIGN CRS
    vector = gpd.read_file(vector_path)
    
    # Use the first raster to establish the reference CRS
    with rio.open(raster_paths[0]) as ref_raster:
        if vector.crs != ref_raster.crs:
            vector = vector.to_crs(ref_raster.crs)
            
    geometry = [geom for geom in vector.geometry]

    # 2. EXTRACT, CROP AND STACK BANDS (X)
    cropped_bands = []
    out_transform = None
    
    for path in raster_paths:
        with rio.open(path) as src:
            # Crop each band using the vector geometry
            out_image, out_transform = mask(dataset=src, shapes=geometry, crop=True)
            # out_image usually has shape (1, height, width). We take index 0.
            cropped_bands.append(out_image[0]) 
            
    # Stack the bands into shape (channels, height, width)
    stacked_image = np.stack(cropped_bands, axis=0)
    channels, height, width = stacked_image.shape

    # 3. GENERATE BINARY MASK (Y)
    out_mask = rasterize(
        [(geom, 1) for geom in geometry],
        out_shape=(height, width),
        transform=out_transform,
        fill=0,
        dtype='uint8'
    )
    
    # Expand mask dims from (H, W) to (1, H, W)
    out_mask = np.expand_dims(out_mask, axis=0)

    # 4. PATCH GENERATION (Synchronized X and Y)
    y_steps = height // patch_size
    x_steps = width // patch_size

    x_patches = []
    y_patches = []

    for i in range(y_steps):
        for j in range(x_steps):
            y_start = i * patch_size
            y_end = (i + 1) * patch_size
            x_start = j * patch_size
            x_end = (j + 1) * patch_size

            # Slice the multi-band image and the mask
            img_patch = stacked_image[:, y_start:y_end, x_start:x_end]
            msk_patch = out_mask[:, y_start:y_end, x_start:x_end]
            
            x_patches.append(img_patch)
            y_patches.append(msk_patch)

    # 5. TENSOR FORMATTING & NORMALIZATION
    tensor_x = np.array(x_patches).transpose(0, 2, 3, 1) # (Batch, H, W, Channels)
    tensor_y = np.array(y_patches).transpose(0, 2, 3, 1) # (Batch, H, W, 1)

    # Convert to float32 and normalize Sentinel-2 data (divide by 10000.0)
    tensor_x = tensor_x.astype(np.float32) / 10000.0
    tensor_x = tensor_x.clip(0.0, 1.0)
    
    tensor_y = tensor_y.astype(np.float32)

    print(f"[DataLoader] X Tensor created: {tensor_x.shape} (Batches, H, W, Bands)")
    print(f"[DataLoader] Y Tensor created: {tensor_y.shape} (Batches, H, W, Classes)")

    return tensor_x, tensor_y

# Optional debugging block
if __name__ == "__main__":
    # Define your multi-band stack here (e.g., Blue, Green, Red, NIR)
    bands = [
        'sample_data/B2_S2.tif',
        'sample_data/B3_S2.tif', 
        'sample_data/B4_S2.tif',
        'sample_data/B8_S2.tif'
    ]
    
    X, Y = extract_training_tensors(
        raster_paths=bands,
        vector_path='sample_data/sample_area.shp'
    )