"""
Module: inference
Description: Loads the trained U-Net model, handles dynamic image sizes via Tiling/Padding,
             performs semantic segmentation, and exports a georeferenced GeoTIFF.
"""

import os
import numpy as np
import rasterio as rio
from rasterio.mask import mask
import geopandas as gpd

# MLOps Silencer
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import keras

import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from core.spatial_utils import calculate_ndvi

# --- CONFIGURATION ---
MODEL_PATH = 'saved_models/spatial_unet_v1.keras'
NEW_RASTER_BANDS = [
    'sample_data/B2_S2.tif',
    'sample_data/B3_S2.tif',
    'sample_data/B4_S2.tif',
    'sample_data/B8_S2.tif'
]
OUTPUT_TIFF = 'predictions/predicted_mask.tif'
PATCH_SIZE = 256

def run_inference(raster_bands, output_path, model_path=MODEL_PATH):
    print("[Inference] Starting inference pipeline...")
    
    if not os.path.exists(model_path):
        print(f"[FATAL ERROR] Model not found at {model_path}")
        return
        
    print("[Inference] Loading trained U-Net model...")

    custom_objects = {
        'Functional': tf.keras.Model
    }

    try:
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(model_path, compile=False)
        print("[Inference] Model loaded successfully with custom scope.")
    except Exception as e:
        print(f"[Inference] Failed to load model: {e}")
        raise e
    
    print("[Inference] Extracting spatial data...")
        
    cropped_bands = []
    out_meta = None

    for idx, path in enumerate(raster_bands):
        with rio.open(path) as src:
            img = src.read(1) # Lê a banda inteira
            cropped_bands.append(img)
            if idx == 0:
                out_meta = src.meta.copy()
            
    stacked_image = np.stack(cropped_bands, axis=0) # (Bands, H, W)
    tensor_x = stacked_image.transpose(1, 2, 0).astype(np.float32) / 10000.0
    tensor_x = tensor_x.clip(0.0, 1.0) # Shape is now (H, W, Bands)
    
    original_h, original_w, num_bands = tensor_x.shape
    print(f"[Inference] Full image shape: {original_h}x{original_w}")

    # --- 1. PADDING ---
    pad_h = (PATCH_SIZE - (original_h % PATCH_SIZE)) % PATCH_SIZE
    pad_w = (PATCH_SIZE - (original_w % PATCH_SIZE)) % PATCH_SIZE
    
    padded_x = np.pad(tensor_x, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
    new_h, new_w = padded_x.shape[0], padded_x.shape[1]
    
    # --- 2. TILING ---
    patches = []
    for i in range(0, new_h, PATCH_SIZE):
        for j in range(0, new_w, PATCH_SIZE):
            patches.append(padded_x[i:i+PATCH_SIZE, j:j+PATCH_SIZE, :])
            
    batch_x = np.array(patches) # Shape: (N_patches, 256, 256, 4)

    print(f"[Inference] Executing prediction on {batch_x.shape[0]} patches...")

    # --- 3. PREDICTION ---
    prediction_tensor = model.predict(batch_x, verbose=1)
    predicted_classes = np.argmax(prediction_tensor, axis=-1).astype('uint8')

    # --- 4. STITCHING ---
    reconstructed_mask = np.zeros((new_h, new_w), dtype='uint8')
    patch_idx = 0
    for i in range(0, new_h, PATCH_SIZE):
        for j in range(0, new_w, PATCH_SIZE):
            reconstructed_mask[i:i+PATCH_SIZE, j:j+PATCH_SIZE] = predicted_classes[patch_idx]
            patch_idx += 1

    # --- 5. CROPPING ---
    final_mask = reconstructed_mask[:original_h, :original_w] # Retorna para 678x643

    red_full = stacked_image[2]
    nir_full = stacked_image[3]
    ndvi_full = calculate_ndvi(red_full, nir_full)

    pivot_pixels = ndvi_full[final_mask == 1]

    mean_ndvi = float(np.mean(pivot_pixels)) if len(pivot_pixels) > 0 else 0.0
    print(f"[Inference] Análise Agronômica: NDVI Médio dos Pivôs = {mean_ndvi:.3f}")

    print(f"[Inference] Saving prediction to {output_path}...")
    
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
            print("[Inference] Arquivo antigo removido para sobrescrita.")
        except PermissionError:
            print("[Inference] Aviso: Falha de permissão ao remover. Tentando sobrescrever diretamente...")
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    out_meta.update({
        "driver": "GTiff",
        "height": original_h,
        "width": original_w,
        "count": 1,
        "dtype": 'uint8'
    })

    with rio.open(output_path, "w", **out_meta) as dest:
        dest.write(final_mask, 1)

    print("[Inference] Pipeline completed successfully!")

    return {
        "output_path": output_path,
        "mean_ndvi": mean_ndvi,
        "detected_pixels": len(pivot_pixels)
    }

if __name__ == "__main__":
    run_inference()