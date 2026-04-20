"""
Module: trainer
Description: Orchestrates data loading, model instantiation, and training loops.
             Saves the trained model weights for production inference.
"""

import os

# --- MLOps Silencer ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
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
        patch_size = PATCH_SIZE,
        class_column = 'category'
    )

    num_bands = X_train.shape[-1]
    input_shape = (PATCH_SIZE, PATCH_SIZE, num_bands)

    # 1.5 SYNCHRONIZED DATA AUGMENTATION
    print("[Trainer] Step 1.5: Applying Synchronized Data Augmentation...")
    x_aug, y_aug = [], []
    
    for x_patch, y_patch in zip(X_train, Y_train):
        # 1. Original
        x_aug.append(x_patch)
        y_aug.append(y_patch)
        # 2. 90 degrees
        x_aug.append(np.rot90(x_patch, 1, axes=(0, 1)))
        y_aug.append(np.rot90(y_patch, 1, axes=(0, 1)))
        # 3. 180 degrees
        x_aug.append(np.rot90(x_patch, 2, axes=(0, 1)))
        y_aug.append(np.rot90(y_patch, 2, axes=(0, 1)))
        # 4. Flip Up-Down
        x_aug.append(np.flipud(x_patch))
        y_aug.append(np.flipud(y_patch))
        
    X_train = np.array(x_aug)
    Y_train = np.array(y_aug)
    print(f"[Trainer] Augmented dataset size: X={X_train.shape}, Y={Y_train.shape}")

    def weighted_sparse_categorical_crossentropy(weights):
        weights_tensor = tf.constant(weights, dtype=tf.float32)

        def loss(y_true, y_pred):
            base_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
            
            y_true_squeezed = tf.squeeze(tf.cast(y_true, tf.int32), axis=-1)
            
            pixel_weights = tf.gather(weights_tensor, y_true_squeezed)
            
            return base_loss * pixel_weights

        return loss

    # 2. BUILD MODEL
    print(f"[Trainer] Step 2: Building U-Net model with input shape {input_shape}...")
    model = build_unet_architecture(input_shape=input_shape, num_classes=3)

    class_weights_list = [1.0, 50.0, 50.0] 

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=weighted_sparse_categorical_crossentropy(class_weights_list),
        metrics=['accuracy']
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