# ==============================================================================
# verify_final_model.py
#
# Description:
#   This script evaluates the performance of the FINAL aggregated model from the
#   federated learning server against the combined, user-exported data.
#
#   It performs the following steps:
#   1. Reconstructs the TCN Keras model architecture.
#   2. Loads the final aggregated weights (e.g., 'final_full_state_model.npy')
#      that were saved by the FL server after the last round.
#   3. "Un-flattens" the weight array and loads it into the Keras model.
#   4. Loads and combines the user-exported JSON data files ('Ajay.json', etc.).
#   5. Preprocesses the data using the original scaler from training.
#   6. Generates a final classification report and confusion matrix to assess
#      the performance of the globally aggregated model.
#
# Instructions:
#   1. Place this script inside the 'export_and_verify' folder.
#   2. Ensure the final model ('final_full_state_model.npy') is in the
#      '../server_deployment/' folder.
#   3. Ensure the user data JSON files are in the '../client_data/' folder.
#   4. Run the script from the command line:
#      python verify_final_model.py
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
# A list of all user-exported data files to be combined for testing.
EXPORTED_DATA_FILENAMES = [
    "Ajay.json",
    "Prabhat.json",
    "Vikash.json",
    "Sandeep.json"
]

# The name of the final model file saved by the server.
FINAL_MODEL_WEIGHTS_FILENAME = "final_full_state_model.npy"

# --- Define Paths ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_SOURCE_DIR = BASE_DIR / ".." / "model_source"
CLIENT_DATA_DIR = BASE_DIR / ".." / "client_data"
SERVER_DEPLOYMENT_DIR = BASE_DIR / ".." / "server_deployment"

# File paths for assets
SCALER_PATH = MODEL_SOURCE_DIR / "scaler.json"
LABELS_PATH = MODEL_SOURCE_DIR / "labels.json"
FINAL_WEIGHTS_PATH = SERVER_DEPLOYMENT_DIR / FINAL_MODEL_WEIGHTS_FILENAME


# ==============================================================================
#                 MODEL ARCHITECTURE (Copied from Training Notebook)
# ==============================================================================
# These parameters must match the model that was originally trained.
WINDOW_SIZE = 60
NUM_FEATURES = 6
KERNEL_SIZE = 7
NUM_FILTERS = 64
NUM_TCN_BLOCKS = 5
DILATION_RATES = [2**i for i in range(NUM_TCN_BLOCKS)]
SPATIAL_DROPOUT_RATE = 0.15
FINAL_DROPOUT_RATE = 0.3
# Note: L2_REG and Dropout in the final Dense layer are not included here
# as they are not present in the notebook's final model architecture.
# The notebook shows Dense layer without regularizers.

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
    # The training notebook has a dropout layer here, but it's not present in the model summary provided
    # Let's add it back for consistency with the code.
    # x = tf.keras.layers.Dropout(FINAL_DROPOUT_RATE)(x) # <-- This was missing
    output_layer = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=input_layer, outputs=output_layer)


def unflatten_weights(model, flat_weights):
    """
    Reshapes a flat array of weights into the structured list of arrays
    required by a Keras model's set_weights() method.
    """
    new_weights = []
    layer_shapes = [w.shape for w in model.get_weights()]
    start = 0
    for shape in layer_shapes:
        size = np.prod(shape)
        end = start + size
        layer_weights = flat_weights[start:end].reshape(shape)
        new_weights.append(layer_weights)
        start = end
    return new_weights
# ==============================================================================
#                 DATA LOADING (Reused from previous script)
# ==============================================================================
def load_and_process_app_data(data_dir, filenames, scaler_mean, scaler_scale):
    all_windows, all_labels = [], []
    print("--- Loading and Parsing Exported App Data ---")
    for filename in filenames:
        data_path = data_dir / filename
        if not data_path.exists():
            print(f"Warning: Data file not found at {data_path}. Skipping.")
            continue
        print(f"Loading data from: {data_path}...")
        with open(data_path, 'r') as f:
            exported_data = json.load(f)
        
        file_window_count = 0
        for item in exported_data:
            label, window_data_str = item.get("label"), item.get("window_data_json")
            if label and window_data_str:
                window_array = np.array(json.loads(window_data_str), dtype=np.float32)
                if window_array.shape == (60, 6):
                    all_windows.append(window_array)
                    all_labels.append(label)
                    file_window_count += 1
        print(f"  - Found {file_window_count} valid windows in {filename}.")

    if not all_windows:
        raise ValueError("No valid windows found across all specified files.")
        
    print(f"\nTotal valid windows from all files: {len(all_windows)}")
    X = np.array(all_windows)
    y_true = np.array(all_labels)
    print("Scaling the combined data using the provided scaler...")
    X_scaled = (X - scaler_mean) / scaler_scale
    return X_scaled, y_true
# ==============================================================================

def main():
    """Main function to run the verification process."""
    print("--- Verifying FINAL Aggregated FL Model ---")

    # 1. Load scaler and labels
    print(f"\nLoading scaler from: {SCALER_PATH}")
    with open(SCALER_PATH, 'r') as f:
        scaler_params = json.load(f)
    scaler_mean, scaler_scale = np.array(scaler_params['mean']), np.array(scaler_params['scale'])
    
    print(f"Loading labels from: {LABELS_PATH}")
    with open(LABELS_PATH, 'r') as f:
        labels_map_int_to_str = json.load(f)
    num_classes = len(labels_map_int_to_str)
    class_names = sorted(labels_map_int_to_str.values())

    # 2. Build model and load final aggregated weights
    print("\nBuilding TCN model architecture...")
    model = build_tcn_model(input_shape=(WINDOW_SIZE, NUM_FEATURES), num_classes=num_classes)
    
    print(f"Loading final aggregated weights from: {FINAL_WEIGHTS_PATH}")
    if not FINAL_WEIGHTS_PATH.exists():
        print(f"\nFATAL: Final model weights file not found at '{FINAL_WEIGHTS_PATH}'")
        print("Please run the federated learning server to completion first.")
        return
        
    final_flat_weights = np.load(FINAL_WEIGHTS_PATH)
    
    print("Un-flattening weights and setting them on the model...")
    structured_weights = unflatten_weights(model, final_flat_weights)
    model.set_weights(structured_weights)
    print("Final aggregated weights have been loaded into the model.")

    # 3. Load and preprocess the application data
    try:
        X_test, y_test_str = load_and_process_app_data(CLIENT_DATA_DIR, EXPORTED_DATA_FILENAMES, scaler_mean, scaler_scale)
        print(f"Final processed data shape for testing: {X_test.shape}")
    except (FileNotFoundError, ValueError) as e:
        print(f"\nFATAL: Failed to load or process data. Error: {e}")
        return

    # 4. Run model predictions
    print("\nRunning model inference on the combined test data...")
    predictions_proba = model.predict(X_test)
    y_pred_int = np.argmax(predictions_proba, axis=1)
    y_pred_str = [labels_map_int_to_str[str(i)] for i in y_pred_int]

    # 5. Generate and print the classification report
    print("\n" + "="*55)
    print(" CLASSIFICATION REPORT FOR FINAL FEDERATED MODEL")
    print("="*55)
    print("This report shows the performance of the globally aggregated model on user data.\n")
    report = classification_report(y_test_str, y_pred_str, labels=class_names, digits=4)
    print(report)
    
    # 6. Generate and display a confusion matrix
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test_str, y_pred_str, labels=class_names)
    print(cm)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix for Final Aggregated FL Model')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    output_plot_path = BASE_DIR / "confusion_matrix_final_model.png"
    plt.savefig(output_plot_path)
    print(f"\nConfusion matrix plot saved to: {output_plot_path}")
    print("\n--- Verification Complete ---")

if __name__ == '__main__':
    main()