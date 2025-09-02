# ==============================================================================
# train_uci_har_egetcn_like.py
#
# Implements and trains a model inspired by the EGTCN paper for a fair
# comparison on the UCI HAR dataset.
#
# Key Features:
# 1. Uses ONLY the raw inertial time-series data, same as the paper.
# 2. Implements an Encoder-Decoder-Classifier architecture.
# 3. The Encoder uses a combination of:
#    - Multitemporal Convolution (MTC) for local patterns.
#    - Self-Attention (as a stand-in for MGC) for global patterns.
# 4. Employs a Dual-Objective Function (DOF) to train for both segmentwise
#    and framewise classification simultaneously.
#
# To Run:
# (tf_venv) sandeep@AcerNitro:~/tf_project/UCI_HAR$ python train_uci_har_egetcn_like.py
# ==============================================================================

# --- Block 1: Setup and Initial Configuration ---
print("--- Block 1: Setup and Initial Configuration ---")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, BatchNormalization, Add,
                                     Conv1D, GlobalAveragePooling1D, Activation,
                                     Concatenate, MaxPooling1D, UpSampling1D,
                                     MultiHeadAttention, LayerNormalization)

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# --- Main Configuration ---
DATASET_PATH = Path('./data/UCI HAR Dataset/')
OUTPUT_DIR = Path('./results_egetcn_like/')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Model & Training Parameters ---
WINDOW_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 75
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 15

# --- EGTCN-like Parameters ---
# Loss weights for the Dual-Objective Function (DOF), inspired by Table IV in the paper for UCI-HAR
ALPHA = 1.0  # Weight for segmentwise loss
BETA = 0.8   # Weight for framewise loss

NUM_ENCODER_BLOCKS = 3
NUM_FILTERS = 64
KERNEL_SIZE = 7

# --- GPU Check ---
# (Identical to previous script, included for completeness)
print("\n--- GPU Check ---")
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print(f"Found {len(gpu_devices)} GPU(s).")
    for gpu in gpu_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("!!! No GPU found. Training will use CPU. !!!")
print("-" * 25)

# ==============================================================================
# --- Block 2: Data Loading and Preparation ---
# ==============================================================================
print("\n--- Block 2: Loading and Preparing Data ---")

def load_data():
    """Loads all necessary data for the EGTCN-like model."""
    # Load activity labels
    activity_labels_df = pd.read_csv(DATASET_PATH / 'activity_labels.txt', header=None, delim_whitespace=True)
    activity_labels = activity_labels_df[1].tolist()
    
    # Load segmentwise numerical labels (0-indexed)
    y_train_segment = pd.read_csv(DATASET_PATH / 'train/y_train.txt', header=None).values.flatten() - 1
    y_test_segment = pd.read_csv(DATASET_PATH / 'test/y_test.txt', header=None).values.flatten() - 1
    
    # Create framewise labels by repeating the segment label for each time step
    y_train_frame = np.repeat(y_train_segment[:, np.newaxis], WINDOW_SIZE, axis=1)
    y_test_frame = np.repeat(y_test_segment[:, np.newaxis], WINDOW_SIZE, axis=1)
    
    # Load inertial signals
    signals = ['body_acc_x', 'body_acc_y', 'body_acc_z',
               'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
               'total_acc_x', 'total_acc_y', 'total_acc_z']
    
    def load_signals_subset(subset):
        data = [pd.read_csv(DATASET_PATH / subset / 'Inertial Signals' / f'{s}_{subset}.txt', header=None, delim_whitespace=True).values for s in signals]
        return np.stack(data, axis=-1)

    X_train_inertial = load_signals_subset('train')
    X_test_inertial = load_signals_subset('test')
    
    return (X_train_inertial, y_train_segment, y_train_frame,
            X_test_inertial, y_test_segment, y_test_frame,
            activity_labels)

def plot_confusion_matrix(y_true, y_pred, labels, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("EGTCN-like Model Confusion Matrix (Segmentwise)")
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()
    print(f"Confusion matrix saved to {OUTPUT_DIR / filename}")


# ==============================================================================
# --- Block 3: EGTCN-like Model Definition ---
# ==============================================================================
print("\n--- Block 3: Defining the EGTCN-like Model Architecture ---")

def attention_block(inputs):
    """Self-Attention block to capture global temporal relationships (stand-in for MGC)."""
    # Using MultiHeadAttention to find relationships between time steps
    x = LayerNormalization()(inputs)
    x = MultiHeadAttention(num_heads=8, key_dim=NUM_FILTERS // 8)(x, x)
    x = Dropout(0.2)(x)
    return Add()([inputs, x])

def mtc_block(inputs, filters, kernel_size, stride):
    """Multitemporal Convolution (MTC) block for local feature extraction."""
    # Project input channels
    x = Conv1D(filters // 2, 1, padding='same', activation='relu')(inputs)
    
    # Parallel convolutions with different dilations
    conv_d1 = Conv1D(filters // 2, kernel_size, dilation_rate=1, padding='causal')(x)
    conv_d2 = Conv1D(filters // 2, kernel_size, dilation_rate=2, padding='causal')(x)
    
    # Concatenate and process
    x = Concatenate()([conv_d1, conv_d2])
    x = BatchNormalization()(x)
    x = Activation('mish')(x)
    
    # Reduce sequence length
    if stride > 1:
        x = MaxPooling1D(pool_size=stride)(x)
        
    return x

def build_egetcn_like_model(input_shape, num_classes):
    """Builds the full Encoder-Decoder-Classifier model."""
    input_layer = Input(shape=input_shape, name='inertial_input')
    x = input_layer
    
    # --- Encoder ---
    # Stacking blocks to learn hierarchical features
    for i in range(NUM_ENCODER_BLOCKS):
        x = attention_block(x)
        # Stride is applied in the first two blocks to reduce length
        stride = 2 if i < 2 else 1
        x = mtc_block(x, NUM_FILTERS * (2**i), KERNEL_SIZE, stride)

    encoder_output = x
    
    # --- Decoder (Framewise Prediction) ---
    # Upsample to original sequence length for per-frame classification
    decoder_x = UpSampling1D(size=4)(encoder_output) # 2*2=4 to reverse the two strides of 2
    framewise_output = Conv1D(num_classes, 1, activation='softmax', name='framewise_output')(decoder_x)
    
    # --- Classifier (Segmentwise Prediction) ---
    classifier_x = GlobalAveragePooling1D()(encoder_output)
    classifier_x = Dropout(0.4)(classifier_x)
    classifier_x = Dense(128, activation='relu')(classifier_x)
    segmentwise_output = Dense(num_classes, activation='softmax', name='segmentwise_output')(classifier_x)
    
    model = Model(inputs=input_layer, outputs=[segmentwise_output, framewise_output])
    
    # Define losses and weights for the two outputs
    losses = {
        'segmentwise_output': 'sparse_categorical_crossentropy',
        'framewise_output': 'sparse_categorical_crossentropy',
    }
    loss_weights = {
        'segmentwise_output': ALPHA,
        'framewise_output': BETA,
    }
    
    # --- *** START OF FIX *** ---
    # Specify metrics for each output using a dictionary
    metrics = {
        'segmentwise_output': 'accuracy',
        'framewise_output': 'accuracy',
    }
    
    model.compile(optimizer='adam', 
                  loss=losses, 
                  loss_weights=loss_weights, 
                  metrics=metrics) # Use the new metrics dictionary here
    # --- *** END OF FIX *** ---

    return model

# ==============================================================================
# --- Block 4: Main Training and Evaluation ---
# ==============================================================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("--- Starting EGTCN-like Model Training ---")
    print("="*50)

    # --- Load and prepare data ---
    (X_train, y_train_seg, y_train_frame,
     X_test, y_test_seg, y_test_frame,
     activity_labels) = load_data()
    
    num_classes = len(activity_labels)

    # --- Scale inertial data ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    print("Inertial signal scaling complete.")

    # --- Build the model ---
    model = build_egetcn_like_model(
        input_shape=(WINDOW_SIZE, X_train.shape[-1]),
        num_classes=num_classes
    )
    model.summary()

    # --- Callbacks ---
    checkpoint_path = OUTPUT_DIR / 'egetcn_like_model.keras'
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        # Monitor the weighted sum of losses
        monitor='val_loss',
        save_best_only=True,
        mode='min'
    )
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True)

    # --- Train the model ---
    print("\nTraining the model with dual outputs...")
    history = model.fit(
        X_train_scaled,
        {'segmentwise_output': y_train_seg, 'framewise_output': y_train_frame},
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[model_checkpoint, reduce_lr, early_stopping],
        verbose=1
    )

    # --- Evaluate the best model ---
    print("\nEvaluating EGTCN-like model on test data...")
    best_model = tf.keras.models.load_model(checkpoint_path)
    
    # Evaluation returns total loss, and then loss & metric for each output
    results = best_model.evaluate(
        X_test_scaled,
        {'segmentwise_output': y_test_seg, 'framewise_output': y_test_frame},
        verbose=0
    )
    
    print("\n--- Final Performance Report ---")
    print(f"Total Test Loss: {results[0]:.4f}")
    print(f"Segmentwise Test Accuracy: {results[3]:.4f}") # Accuracy for segmentwise_output
    print(f"Framewise Test Accuracy: {results[4]:.4f}") # Accuracy for framewise_output

    # --- Generate Classification Report for Segmentwise Prediction ---
    y_pred_list = best_model.predict(X_test_scaled)
    y_pred_segment = np.argmax(y_pred_list[0], axis=1) # Predictions are in the first element of the list
    
    print("\nSegmentwise Classification Report:")
    print(classification_report(y_test_seg, y_pred_segment, target_names=activity_labels))
    plot_confusion_matrix(y_test_seg, y_pred_segment, activity_labels, "egetcn_like_confusion_matrix.png")

    print("\n" + "="*50)
    print("--- SCRIPT COMPLETE ---")
    print(f"All models and plots saved in: {OUTPUT_DIR}")
    print("="*50)