"""
Module: trainer
Description: Orchestrates data loading, model instantiation, and training loops.
             Saves the trained model weights for production inference.
"""

import os

# --- MLOps Silencer ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

from data_loader import extract_training_tensors

from convolution_model import build_unet_architecture

# --- CONFIGURATION & HYPERPARAMETERS ---
RASTER_BANDS = [
    'sample_data/B2_S2.tif',
    'sample_data/B3_S2.tif',
    'sample_data/B4_S2.tif',
    'sample_data/B8_S2.tif'
]
VECTOR_TRUTH = 'sample_data/sample_area.shp'

PATCH_SIZE = 256
EPOCHS = 50
BATCH_SIZE = 8
MODEL_SAVE_PATH = 'saved_models/spatial_unet_v1.keras'

def run_training_pipeline():
    """
    Executes the end-to-end MLOps training pipeline.
    """
    print("[Trainer] Step 1: Initiating MLOps Training Pipeline...")

    # 1. DATA EXTRACTION
    print("[Trainer] Loading and formatting spatial data...")
    X_train, Y_train = extract_training_tensors(
        raster_paths = RASTER_BANDS,
        vector_path = VECTOR_TRUTH, 
        patch_size = PATCH_SIZE
    )

    num_bands = X_train.shape[-1]
    input_shape = (PATCH_SIZE, PATCH_SIZE, num_bands)

    # 2. BUILD MODEL
    print("[Trainer] Step 2: Building U-Net model with input shape {input_shape}...")
    model = build_unet_architecture(input_shape = input_shape, num_classes=2)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )

    # 3. CONFIGURE CALLBACKS
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath = MODEL_SAVE_PATH,
            save_best_only = True,
            monitor = 'loss',
            mode = 'min',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor = 'val_loss',
            patience = 10,
            restore_best_weights = True,
            verbose = 1
        )
    ]

    # 4. TRAINING
    print("[Trainer] Step 3: Starting training loop...")
    history = model.fit(
        x = X_train, 
        y = Y_train,
        batch_size = BATCH_SIZE,
        epochs = EPOCHS,
        callbacks = callbacks,
        validation_split = 0.2 # 20% of patches reserved for internal validation
    )

    print(f"[Trainer] Pipeline execution finished. Model saved to {MODEL_SAVE_PATH}")
    return history

if __name__ == "__main__":
    import sys    
    print("[Trace] 8. Calling main function...", flush=True)
    
    try:
        # Verifica se os arquivos realmente existem antes de deixar o Rasterio explodir a máquina
        if not os.path.exists(VECTOR_TRUTH):
            print(f"[FATAL] Shapefile not found at: {VECTOR_TRUTH}")
            sys.exit(1)
            
        for band in RASTER_BANDS:
            if not os.path.exists(band):
                print(f"[FATAL] Raster band not found at: {band}")
                sys.exit(1)

        print("[Trace] 9. Files located successfully on disk. Starting pipeline...", flush=True)
        run_training_pipeline()
        
    except Exception as e:
        print(f"\n[PYTHON ERROR CAUGHT]: {e}")