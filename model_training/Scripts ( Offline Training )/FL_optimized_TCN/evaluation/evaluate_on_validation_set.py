# evaluation/evaluate_on_validation_set.py
# Evaluates the final exported TFLite model on the hold-out validation set.

import numpy as np
import tensorflow as tf
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuration ---
print("--- [1/4] Setting up configuration ---")

# --- Path Setup ---
SCRIPT_DIR = Path(__file__).parent.resolve()
EXPORT_DIR = SCRIPT_DIR / ".." / "export" / "generated_assets"
VALIDATION_DATA_DIR = SCRIPT_DIR / ".." / "validation_data"
EVALUATION_RESULTS_DIR = SCRIPT_DIR / "evaluation_results"
EVALUATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Asset and Data Paths ---
TFLITE_MODEL_PATH = EXPORT_DIR / "tcn_fl_optimized_trainable.tflite"
SCALER_PATH = EXPORT_DIR / "scaler.json"
LABELS_PATH = EXPORT_DIR / "labels.json"
VALIDATION_DATA_PATH = VALIDATION_DATA_DIR / "validation_windows.json"
# --- NEW: Path to the initial weights for model initialization ---
INITIAL_WEIGHTS_PATH = EXPORT_DIR / "initial_weights.bin"

# --- 2. Load Assets and TFLite Model ---
print("--- [2/4] Loading TFLite model and assets ---")

# Load scaler and labels
with open(SCALER_PATH, 'r') as f:
    scaler_dict = json.load(f)
    mean = np.array(scaler_dict['mean'], dtype=np.float32)
    scale = np.array(scaler_dict['scale'], dtype=np.float32)
with open(LABELS_PATH, 'r') as f:
    labels_map = json.load(f)
    idx_to_label = {int(k): v for k, v in labels_map.items()}
    label_to_idx = {v: int(k) for k, v in labels_map.items()}
    class_names = [idx_to_label[i] for i in sorted(idx_to_label.keys())]

# Load TFLite interpreter
interpreter = tf.lite.Interpreter(model_path=str(TFLITE_MODEL_PATH))
interpreter.allocate_tensors()
infer_signature = interpreter.get_signature_runner('infer')
# --- NEW: Get the set_weights signature runner ---
set_weights_signature = interpreter.get_signature_runner('set_weights_flat')
print("✅ Model and assets loaded successfully.")

# ==============================================================================
#                 THE CRITICAL FIX: INITIALIZE THE MODEL'S VARIABLES
# ==============================================================================
print("--- Initializing model with baseline weights ---")
if not INITIAL_WEIGHTS_PATH.exists():
    raise FileNotFoundError(f"Initial weights file not found at {INITIAL_WEIGHTS_PATH}. Please run the export script first.")

initial_weights = np.fromfile(INITIAL_WEIGHTS_PATH, dtype=np.float32)
set_weights_signature(flat_weights=initial_weights)
print("✅ Model variables initialized.")
# ==============================================================================


# --- 3. Load and Preprocess Validation Data ---
print("--- [3/4] Loading and preprocessing validation data ---")
with open(VALIDATION_DATA_PATH, 'r') as f:
    validation_data = json.load(f)

y_true = []
y_pred = []

for item in validation_data:
    raw_window = np.array(item['window'], dtype=np.float32)
    true_label_str = item['label']
    
    scaled_window = (raw_window - mean) / scale
    input_tensor = np.expand_dims(scaled_window, axis=0).astype(np.float32)
    
    # This will now succeed because the variables have been initialized.
    output = infer_signature(x_input=input_tensor)
    logits = output['logits'][0]
    
    predicted_idx = np.argmax(logits)
    
    y_pred.append(predicted_idx)
    y_true.append(label_to_idx[true_label_str])

print(f"✅ Inference complete on {len(validation_data)} validation samples.")

# --- 4. Report Results ---
print("--- [4/4] Generating evaluation report ---")

accuracy = accuracy_score(y_true, y_pred)
print(f"\nOverall Accuracy on Validation Set: {accuracy:.4f} ({accuracy:.2%})\n")

report_str = classification_report(y_true, y_pred, target_names=class_names)
print("Classification Report:")
print(report_str)

with open(EVALUATION_RESULTS_DIR / 'validation_report.txt', 'w') as f:
    f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
    f.write(report_str)
print("\nClassification report saved to evaluation_results/")

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix on Hold-Out Validation Set')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(EVALUATION_RESULTS_DIR / 'validation_confusion_matrix.png')
print("Confusion matrix plot saved to evaluation_results/")
plt.show()