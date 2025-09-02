# ==============================================================================
# verify_final_model_on_test_set.py
#
# Description:
#   This script evaluates the performance of the FINAL aggregated model from the
#   federated learning server against the definitive test set from training.
#
#   This helps determine if the FL process introduced any regression or
#   improvements on the original, clean validation data.
#
#   Steps:
#   1. Reconstructs the TCN Keras model architecture.
#   2. Loads the final aggregated weights ('final_full_state_model.npy')
#      from the '../server_deployment/' directory.
#   3. Loads the definitive test set from '../definitive_test_set/'.
#   4. Runs inference and generates a final classification report.
#
# Instructions:
#   1. Place this script inside the 'export_and_verify' folder.
#   2. Ensure the final model file is in '../server_deployment/'.
#   3. Ensure the definitive test set is in '../definitive_test_set/'.
#   4. Run the script:
#      python verify_final_model_on_test_set.py
#
# ==============================================================================

import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Add, Conv1D, BatchNormalization, Activation, SpatialDropout1D, GlobalAveragePooling1D, Dense
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuration ---
FINAL_MODEL_WEIGHTS_FILENAME = "final_full_state_model.npy"

# --- Define Paths ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_SOURCE_DIR = BASE_DIR / ".." / "model_source"
TEST_SET_DIR = BASE_DIR / ".." / "definitive_test_set"
SERVER_DEPLOYMENT_DIR = BASE_DIR / ".." / "server_deployment"

# File paths for assets
LABELS_PATH = MODEL_SOURCE_DIR / "labels.json"
FINAL_WEIGHTS_PATH = SERVER_DEPLOYMENT_DIR / FINAL_MODEL_WEIGHTS_FILENAME
X_TEST_PATH = TEST_SET_DIR / "X_test_scaled.npy"
Y_TEST_PATH = TEST_SET_DIR / "y_test.npy"


# ==============================================================================
#                 MODEL ARCHITECTURE (Copied from Training Notebook)
# ==============================================================================
WINDOW_SIZE, NUM_FEATURES, KERNEL_SIZE, NUM_FILTERS = 60, 6, 7, 64
NUM_TCN_BLOCKS, SPATIAL_DROPOUT_RATE, FINAL_DROPOUT_RATE = 5, 0.15, 0.3
DILATION_RATES = [2**i for i in range(NUM_TCN_BLOCKS)]

def residual_block(x, dilation_rate, nb_filters, kernel_size, dropout_rate=0.0):
    prev_x = x
    conv1 = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='same')(x)
    conv1 = BatchNormalization()(conv1); conv1 = Activation('relu')(conv1); conv1 = SpatialDropout1D(dropout_rate)(conv1)
    conv2 = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='same')(conv1)
    conv2 = BatchNormalization()(conv2); conv2 = Activation('relu')(conv2); conv2 = SpatialDropout1D(dropout_rate)(conv2)
    if prev_x.shape[-1] != conv2.shape[-1]: prev_x = Conv1D(nb_filters, 1, padding='same')(prev_x)
    return Add()([prev_x, conv2])

def build_tcn_model(input_shape, num_classes):
    input_layer = Input(shape=input_shape)
    x = input_layer
    for rate in DILATION_RATES:
        x = residual_block(x, rate, NUM_FILTERS, KERNEL_SIZE, SPATIAL_DROPOUT_RATE)
    x = GlobalAveragePooling1D()(x)
    # The training notebook summary shows a Dropout layer before the final Dense layer.
    # It seems to have been omitted in the code block but present in the output summary.
    # We will add it here to match the parameter count and architecture.
    x = tf.keras.layers.Dropout(FINAL_DROPOUT_RATE)(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=input_layer, outputs=output_layer)

def unflatten_weights(model, flat_weights):
    new_weights = []
    start = 0
    for layer in model.layers:
        layer_weights = layer.get_weights()
        if not layer_weights:
            continue
        for i, weight_matrix in enumerate(layer_weights):
            shape = weight_matrix.shape
            size = np.prod(shape)
            end = start + size
            new_weight = flat_weights[start:end].reshape(shape)
            layer_weights[i] = new_weight
            start = end
        new_weights.extend(layer_weights)
    return new_weights
# ==============================================================================

def main():
    """Main function to run the verification process."""
    print("--- Verifying FINAL Aggregated FL Model on Definitive Test Set ---")

    # 1. Load labels
    print(f"\nLoading labels from: {LABELS_PATH}")
    with open(LABELS_PATH, 'r') as f:
        labels_map_int_to_str = json.load(f)
    num_classes = len(labels_map_int_to_str)
    class_names = sorted(labels_map_int_to_str.values())

    # 2. Load the definitive test set
    print(f"Loading test set data from: {X_TEST_PATH}")
    if not X_TEST_PATH.exists() or not Y_TEST_PATH.exists():
        print(f"\nFATAL: Test set files not found in '{TEST_SET_DIR}'.")
        return
    X_test = np.load(X_TEST_PATH)
    y_test_int = np.load(Y_TEST_PATH)
    y_test_str = np.array([labels_map_int_to_str[str(i)] for i in y_test_int])
    print(f"Test data loaded. Shape: {X_test.shape}")

    # 3. Build model and load final aggregated weights
    print("\nBuilding TCN model architecture...")
    model = build_tcn_model(input_shape=(WINDOW_SIZE, NUM_FEATURES), num_classes=num_classes)
    
    print(f"Loading final aggregated weights from: {FINAL_WEIGHTS_PATH}")
    if not FINAL_WEIGHTS_PATH.exists():
        print(f"\nFATAL: Final model weights file not found at '{FINAL_WEIGHTS_PATH}'")
        return
    final_flat_weights = np.load(FINAL_WEIGHTS_PATH)
    
    print("Un-flattening weights and setting them on the model...")
    structured_weights = unflatten_weights(model, final_flat_weights)
    model.set_weights(structured_weights)
    print("Final aggregated weights have been loaded into the model.")

    # 4. Run model predictions
    print("\nRunning model inference on the test set...")
    predictions_proba = model.predict(X_test)
    y_pred_int = np.argmax(predictions_proba, axis=1)
    y_pred_str = [labels_map_int_to_str[str(i)] for i in y_pred_int]

    # 5. Generate and print the classification report
    print("\n" + "="*55)
    print(" CLASSIFICATION REPORT FOR FINAL FEDERATED MODEL")
    print("="*55)
    report = classification_report(y_test_str, y_pred_str, labels=class_names, digits=4)
    print(report)
    
    # 6. Generate and display a confusion matrix
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test_str, y_pred_str, labels=class_names)
    print(cm)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix for Final FL Model on Definitive Test Set')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    output_plot_path = BASE_DIR / "cm_final_model_on_test_set.png"
    plt.savefig(output_plot_path)
    print(f"\nConfusion matrix plot saved to: {output_plot_path}")
    print("\n--- Verification Complete ---")

if __name__ == '__main__':
    main()