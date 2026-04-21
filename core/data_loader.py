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
import random

from core.spatial_utils import calculate_ndvi

def extract_training_tensors(raster_paths, vector_path, patch_size=256, class_column='category', negative_ratio=2):
    """
    Reads geographic data, crops multiple bands, stacks them, normalizes, 
    and generates patches for deep learning.
    
    Args:
        raster_paths (list of str): Paths to the satellite image bands (e.g., ['B2.tif', 'B3.tif']).
        vector_path (str): Path to the shapefile with ground truth polygons.
        patch_size (int): Dimension of the output square patches.
        class_column (str): The attribute column in the shapefile defining 1 (Pivot) and 0 (Background/Hard Negative).
        negative_ratio (int): How many pure background patches to keep for every 1 positive patch.
        
    Returns:
        tuple: (X_tensor, Y_tensor) ready for TensorFlow training.
    """
    
    # 1. LOAD VECTOR AND ALIGN CRS
    vector = gpd.read_file(vector_path)
    
    if class_column not in vector.columns:
        raise ValueError(f"[Data Loader] Class column '{class_column}' not found in vector data.")

    # Use the first raster to establish the reference CRS
    with rio.open(raster_paths[0]) as ref_raster:
        if vector.crs != ref_raster.crs:
            vector = vector.to_crs(ref_raster.crs)
            
    out_meta = ref_raster.meta.copy()
    height, width = ref_raster.height, ref_raster.width
    out_transform = ref_raster.transform

    # 2. RASTERIZE GROUND TRUTH
    print("[DataLoader] Rasterizing Ground Truth masks...")
    shapes = ((geom, value) for geom, value in zip(vector.geometry, vector[class_column]))
    
    out_mask = rasterize(
        shapes,
        out_shape = (height, width),
        transform = out_transform,
        fill = 0,
        dtype = 'uint8'
    )
    out_mask = np.expand_dims(out_mask, axis=0) # (1, H, W)

    # 3. EXTRACT AND STACK BANDS
    cropped_bands = []
    for path in raster_paths:
        with rio.open(path) as src:
            img = src.read(1)
            cropped_bands.append(img)
            
    stacked_image = np.stack(cropped_bands, axis=0) # (Channels, H, W)

    # 4. TILING
    print(f"[DataLoader] Tiling {height}x{width} image into {patch_size}x{patch_size} patches...")
    y_steps = height // patch_size
    x_steps = width // patch_size

    x_patches_all = []
    y_patches_all = []

    for i in range(y_steps):
        for j in range(x_steps):
            y_start = i * patch_size
            y_end = (i + 1) * patch_size
            x_start = j * patch_size
            x_end = (j + 1) * patch_size

            img_patch = stacked_image[:, y_start:y_end, x_start:x_end]
            msk_patch = out_mask[:, y_start:y_end, x_start:x_end]
            
            x_patches_all.append(img_patch)
            y_patches_all.append(msk_patch)

    # 5. SMART SAMPLING (Balancing the dataset)
    print("[DataLoader] Balancing dataset to prevent Class Imbalance...")
    positives = []
    negatives = []
    
    for idx, mask in enumerate(y_patches_all):
        if np.sum(mask) > 0: 
            positives.append(idx)
        else:
            negatives.append(idx)
            
    print(f"[DataLoader] Found {len(positives)} positive patches and {len(negatives)} negative patches.")
    
    # Limita os negativos baseando-se na quantidade de positivos
    num_negatives_to_keep = min(len(negatives), len(positives) * negative_ratio)
    sampled_negatives = random.sample(negatives, num_negatives_to_keep) if num_negatives_to_keep > 0 else []
    
    selected_indices = positives + sampled_negatives
    random.shuffle(selected_indices) # Embaralha para o modelo não viciar na ordem
    
    final_x = [x_patches_all[i] for i in selected_indices]
    final_y = [y_patches_all[i] for i in selected_indices]

    # 6. TENSOR FORMATTING & NORMALIZATION
    tensor_x = np.array(final_x).transpose(0, 2, 3, 1) # (Batch, H, W, Channels)
    tensor_y = np.array(final_y).transpose(0, 2, 3, 1) # (Batch, H, W, 1)

    tensor_x = tensor_x.astype(np.float32) / 10000.0
    tensor_x = tensor_x.clip(0.0, 1.0)
    tensor_y = tensor_y.astype(np.float32)

    print(f"[DataLoader] Final Training X Tensor: {tensor_x.shape}")
    print(f"[DataLoader] Final Training Y Tensor: {tensor_y.shape}")

    return tensor_x, tensor_y

if __name__ == "__main__":
    bands = [
        'sample_data/B2_S2.tif',
        'sample_data/B3_S2.tif', 
        'sample_data/B4_S2.tif',
        'sample_data/B8_S2.tif'
    ]
    X, Y = extract_training_tensors(bands, 'sample_data/sample_area.shp', class_column='category')