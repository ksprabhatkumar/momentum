# ==============================================================================
# verify_app_export.py
#
# Description:
#   This script validates the performance of the trained TCN model against
#   a combined dataset of labeled data exported from the Momentum Android app.
#
#   It performs the following steps:
#   1. Loads the pre-trained Keras model, scaler, and labels from the
#      '../model_source/' directory.
#   2. Loads and combines data from the specified JSON files in the
#      '../client_data/' directory.
#   3. Parses the nested JSON structure of the exported data.
#   4. Preprocesses the data by applying the same scaling transformation used
#      during training.
#   5. Runs inference with the model to get predictions.
#   6. Generates and prints a detailed classification report and a confusion
#      matrix to evaluate the model's performance on the combined,
#      user-labeled dataset.
#
# Instructions:
#   1. Place this script inside the 'export_and_verify' folder.
#   2. Ensure the following files are in the '../client_data/' folder:
#      - Ajay.json
#      - Prabhat.json
#      - Vikash.json
#   3. Ensure the model, scaler, and labels are in the '../model_source/' folder.
#   4. Run the script from the command line:
#      python verify_app_export.py
#
# ==============================================================================

import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuration ---
# The script will load and combine all files listed here.
EXPORTED_DATA_FILENAMES = [
    "Ajay.json",
    "Prabhat.json",
    "Vikash.json",
    "Sandeep.json"
]

# --- Define Paths ---
# The script is in 'export_and_verify', so we go up one level ('..')
# to access sibling directories like 'model_source' and 'client_data'.
BASE_DIR = Path(__file__).resolve().parent
MODEL_SOURCE_DIR = BASE_DIR / ".." / "model_source"
CLIENT_DATA_DIR = BASE_DIR / ".." / "client_data"

# File paths for model, scaler, and labels
MODEL_PATH = MODEL_SOURCE_DIR / "tcn_v2_ABCDE.keras"
SCALER_PATH = MODEL_SOURCE_DIR / "scaler.json"
LABELS_PATH = MODEL_SOURCE_DIR / "labels.json"


def load_and_process_app_data(data_dir, filenames, scaler_mean, scaler_scale):
    """
    Loads and processes labeled data from multiple exported JSON files.

    Args:
        data_dir (Path): Path to the directory containing the data files.
        filenames (list): A list of filenames to load and combine.
        scaler_mean (np.ndarray): The mean values from the scaler.
        scaler_scale (np.ndarray): The scale (std dev) values from the scaler.

    Returns:
        A tuple of (X_scaled, y_true) where:
        - X_scaled is a NumPy array of scaled window data.
        - y_true is a NumPy array of corresponding labels.
    """
    all_windows = []
    all_labels = []

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
            label = item.get("label")
            window_data_str = item.get("window_data_json")

            if label and window_data_str:
                # The 'window_data_json' field is a string, so it must be parsed again
                window_array = np.array(json.loads(window_data_str), dtype=np.float32)

                # Validate the shape of the window data
                if window_array.shape == (60, 6):
                    all_windows.append(window_array)
                    all_labels.append(label)
                    file_window_count += 1
                else:
                    print(f"  - Warning: Skipping a window with incorrect shape: {window_array.shape}")
        print(f"  - Found {file_window_count} valid windows in {filename}.")

    if not all_windows:
        raise ValueError("No valid windows with shape (60, 6) were found across all specified files.")

    print(f"\nTotal valid windows from all files: {len(all_windows)}")
    
    X = np.array(all_windows)
    y_true = np.array(all_labels)

    print("Scaling the combined data using the provided scaler...")
    # Apply the scaling transformation: (data - mean) / scale
    X_scaled = (X - scaler_mean) / scaler_scale
    
    return X_scaled, y_true


def main():
    """Main function to run the verification process."""
    print("--- Starting Verification of Exported App Data ---")

    # 1. Load scaler and labels
    print(f"\nLoading scaler from: {SCALER_PATH}")
    with open(SCALER_PATH, 'r') as f:
        scaler_params = json.load(f)
    scaler_mean = np.array(scaler_params['mean'])
    scaler_scale = np.array(scaler_params['scale'])
    
    print(f"Loading labels from: {LABELS_PATH}")
    with open(LABELS_PATH, 'r') as f:
        labels_map_int_to_str = json.load(f)
    labels_map_str_to_int = {v: int(k) for k, v in labels_map_int_to_str.items()}
    class_names = sorted(labels_map_str_to_int.keys())

    # 2. Load the trained Keras model
    print(f"Loading Keras model from: {MODEL_PATH}")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"\nFATAL: Could not load the Keras model. Error: {e}")
        return

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

    # Convert integer predictions back to string labels ('A', 'B', etc.)
    y_pred_str = [labels_map_int_to_str[str(i)] for i in y_pred_int]

    # 5. Generate and print the classification report
    print("\n" + "="*50)
    print("      CLASSIFICATION REPORT ON EXPORTED APP DATA")
    print("="*50)
    print("This report shows how well the model performed on the combined user-labeled data.\n")
    report = classification_report(y_test_str, y_pred_str, labels=class_names, digits=4)
    print(report)
    
    # 6. Generate and display a confusion matrix for better visualization
    print("\n--- Confusion Matrix ---")
    print("Rows: True Labels, Columns: Predicted Labels\n")
    cm = confusion_matrix(y_test_str, y_pred_str, labels=class_names)
    print(cm)

    # Plotting the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix on Combined Exported App Data')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    # Save the plot
    output_plot_path = BASE_DIR / "confusion_matrix_app_data.png"
    plt.savefig(output_plot_path)
    print(f"\nConfusion matrix plot saved to: {output_plot_path}")
    print("\n--- Verification Complete ---")

if __name__ == '__main__':
    main()