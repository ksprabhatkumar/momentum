# evaluation/evaluate_initial_distribution.py
# Evaluates the initial, centrally-trained Keras model against the partitioned client data
# to understand the baseline performance and data distribution before FL begins.

import numpy as np
import tensorflow as tf
import json
from pathlib import Path
import random
import joblib

print("--- [1/4] Setting up configuration ---")
# --- Path Setup ---
SCRIPT_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = SCRIPT_DIR / ".." / "results"
UNSEEN_DATA_DIR = SCRIPT_DIR / ".." / "unseen_client_data"
EVALUATION_RESULTS_DIR = SCRIPT_DIR / "evaluation_results"
EVALUATION_RESULTS_DIR.mkdir(exist_ok=True)

# --- Asset and Data Paths ---
KERAS_MODEL_PATH = RESULTS_DIR / "fl_optimized_model.keras"
SCALER_PATH = RESULTS_DIR / "scaler.joblib"
LABEL_ENCODER_PATH = RESULTS_DIR / "label_encoder.joblib"
CLIENT_DATA_PATH = UNSEEN_DATA_DIR / "sample_data_for_app.json"
NUM_CLIENTS = 4 # Must match the simulation script

print("--- [2/4] Loading model, assets, and partitioning data ---")
# --- Load Global Assets ---
model = tf.keras.models.load_model(KERAS_MODEL_PATH)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

# --- Load and Partition Client Data (same logic as simulation) ---
with open(CLIENT_DATA_PATH, 'r') as f:
    all_client_data_raw = json.load(f)
# Use a fixed seed to ensure the partitions are the same as in the simulation
random.seed(42)
random.shuffle(all_client_data_raw)
client_data_partitions_raw = np.array_split(all_client_data_raw, NUM_CLIENTS)

# --- Re-usable Preprocessing Function (from simulation) ---
label_map_from_encoder = {i: cls for i, cls in enumerate(label_encoder.classes_)}
def preprocess_data(data, scaler_obj, encoder_obj):
    processed = []
    for item in data:
        raw_window = np.array(item['window'], dtype=np.float32)
        scaled_window = scaler_obj.transform(raw_window)
        label_value = item['label']
        char_label = label_map_from_encoder[label_value] if isinstance(label_value, int) else label_value
        encoded_label = encoder_obj.transform([char_label])[0]
        processed.append((scaled_window, encoded_label))
    return processed

client_datasets = [preprocess_data(p.tolist(), scaler, label_encoder) for p in client_data_partitions_raw]
print(f"Distributed data among {NUM_CLIENTS} clients. Sizes: {[len(d) for d in client_datasets]}")


print("\n--- [3/4] Evaluating initial model on each client's data partition ---")
# --- Evaluate and Report ---
evaluation_results = []
for i, client_data in enumerate(client_datasets):
    client_id = i + 1
    
    # Prepare data for model.evaluate
    client_x = np.array([d[0] for d in client_data], dtype=np.float32)
    client_y = np.array([d[1] for d in client_data], dtype=np.int32)
    
    loss, accuracy = model.evaluate(client_x, client_y, verbose=0)
    
    # Get a label distribution
    labels, counts = np.unique(client_y, return_counts=True)
    label_dist = {label_encoder.classes_[l]: int(c) for l, c in zip(labels, counts)}
    
    print(f"\n- Client {client_id}:")
    print(f"  - Initial Accuracy: {accuracy:.2%}")
    print(f"  - Initial Loss:     {loss:.4f}")
    print(f"  - Label Distribution: {label_dist}")
    
    evaluation_results.append({
        "client_id": client_id,
        "initial_accuracy": accuracy,
        "initial_loss": loss,
        "label_distribution": label_dist
    })

print("\n--- [4/4] Saving detailed report ---")
report_path = EVALUATION_RESULTS_DIR / "initial_client_distribution_report.json"
with open(report_path, 'w') as f:
    json.dump(evaluation_results, f, indent=4)

print(f"\n✅ Detailed report saved to: {report_path}")