# ==============================================================================
# verify_initial_model_on_test_set.py
#
# Description:
#   This script validates the performance of the INITIAL pre-trained Keras
#   model against the definitive test set created during the original training.
#
#   Its purpose is to replicate the benchmark evaluation from the training
#   notebook to confirm that the correct base model is being used.
#
#   Steps:
#   1. Loads the pre-trained Keras model from '../model_source/'.
#   2. Loads the scaled test data (X_test_scaled.npy) and labels (y_test.npy)
#      from the '../definitive_test_set/' directory.
#   3. Runs inference and generates a classification report.
#
# Instructions:
#   1. Place this script inside the 'export_and_verify' folder.
#   2. Ensure the definitive test set files are in '../definitive_test_set/'.
#   3. Ensure the initial Keras model is in '../model_source/'.
#   4. Run the script:
#      python verify_initial_model_on_test_set.py
#
# ==============================================================================

import numpy as np
import tensorflow as tf
import json
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# --- Define Paths ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_SOURCE_DIR = BASE_DIR / ".." / "model_source"
TEST_SET_DIR = BASE_DIR / ".." / "definitive_test_set"

# File paths for model, labels, and the definitive test set
MODEL_PATH = MODEL_SOURCE_DIR / "tcn_v2_ABCDE.keras"
LABELS_PATH = MODEL_SOURCE_DIR / "labels.json"
X_TEST_PATH = TEST_SET_DIR / "X_test_scaled.npy"
Y_TEST_PATH = TEST_SET_DIR / "y_test.npy"


def main():
    """Main function to run the verification process."""
    print("--- Verifying INITIAL Keras Model on Definitive Test Set ---")

    # 1. Load labels
    print(f"\nLoading labels from: {LABELS_PATH}")
    with open(LABELS_PATH, 'r') as f:
        labels_map_int_to_str = json.load(f)
    class_names = sorted(labels_map_int_to_str.values())

    # 2. Load the definitive test set
    print(f"Loading test set data from: {X_TEST_PATH}")
    if not X_TEST_PATH.exists() or not Y_TEST_PATH.exists():
        print(f"\nFATAL: Test set files not found in '{TEST_SET_DIR}'.")
        print("Please ensure 'X_test_scaled.npy' and 'y_test.npy' are present.")
        return
    X_test = np.load(X_TEST_PATH)
    y_test_int = np.load(Y_TEST_PATH)
    y_test_str = np.array([labels_map_int_to_str[str(i)] for i in y_test_int])
    print(f"Test data loaded. Shape: {X_test.shape}")

    # 3. Load the initial Keras model
    print(f"Loading Keras model from: {MODEL_PATH}")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"\nFATAL: Could not load the Keras model. Error: {e}")
        return

    # 4. Run model predictions
    print("\nRunning model inference on the test set...")
    predictions_proba = model.predict(X_test)
    y_pred_int = np.argmax(predictions_proba, axis=1)
    y_pred_str = [labels_map_int_to_str[str(i)] for i in y_pred_int]

    # 5. Generate and print the classification report
    print("\n" + "="*50)
    print("      CLASSIFICATION REPORT (INITIAL MODEL)")
    print("="*50)
    print("This report should match the benchmark from the training notebook.\n")
    report = classification_report(y_test_str, y_pred_str, labels=class_names, digits=4)
    print(report)
    
    # 6. Generate and display a confusion matrix
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test_str, y_pred_str, labels=class_names)
    print(cm)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix for Initial Model on Definitive Test Set')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    output_plot_path = BASE_DIR / "cm_initial_model_on_test_set.png"
    plt.savefig(output_plot_path)
    print(f"\nConfusion matrix plot saved to: {output_plot_path}")
    print("\n--- Verification Complete ---")

if __name__ == '__main__':
    main()