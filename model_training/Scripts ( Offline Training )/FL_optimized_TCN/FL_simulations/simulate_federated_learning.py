# FL_simulations/simulate_federated_learning.py
# A script to simulate a multi-client federated learning process using the actual TFLite artifact.

import numpy as np
import tensorflow as tf
import json
from pathlib import Path
import random
import joblib

# --- 1. Configuration ---
print("--- [1/6] Setting up FL TFLite simulation configuration ---")
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# --- FL Hyperparameters ---
NUM_CLIENTS = 4
NUM_ROUNDS = 5
LOCAL_EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE_FL = 1e-4

# --- Path Setup ---
SCRIPT_DIR = Path(__file__).parent.resolve()
EXPORT_DIR = SCRIPT_DIR / ".." / "export" / "generated_assets"
VALIDATION_DATA_DIR = SCRIPT_DIR / ".." / "validation_data"
CLIENT_DATA_DIR = SCRIPT_DIR / ".." / "unseen_client_data"
SIMULATION_RESULTS_DIR = SCRIPT_DIR / "simulation_results"
SIMULATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Asset and Data Paths ---
TFLITE_MODEL_PATH = EXPORT_DIR / "tcn_fl_optimized_trainable.tflite"
SCALER_PATH = EXPORT_DIR / "scaler.json"
LABEL_ENCODER_PATH = SCRIPT_DIR / ".." / "results" / "label_encoder.joblib"
# --- !!! IMPORTANT FIX #1: Point to the balanced data file ---
CLIENT_DATA_PATH = CLIENT_DATA_DIR / "sample_data_for_app.json"
VALIDATION_DATA_PATH = VALIDATION_DATA_DIR / "validation_windows.json"

# --- 2. TFLite Harness and Data Loading ---
print("--- [2/6] Defining TFLite Harness and loading data ---")

class TFLiteHarness:
    """A Python equivalent of the Android TFLiteHarness."""
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.infer = self.interpreter.get_signature_runner('infer')
        self.train_step = self.interpreter.get_signature_runner('train_step')
        self.get_weights = self.interpreter.get_signature_runner('get_weights_flat')
        self.set_weights = self.interpreter.get_signature_runner('set_weights_flat')

    def evaluate(self, x_val, y_val):
        """Evaluates the model on a validation set."""
        if len(x_val) == 0: return 0.0, 0.0
        logits = self.infer(x_input=x_val)['logits']
        preds = np.argmax(logits, axis=1)
        accuracy = np.mean(preds == y_val)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        loss = loss_fn(y_val, logits).numpy()
        return loss, accuracy

# --- Load assets ---
with open(SCALER_PATH, 'r') as f:
    scaler_dict = json.load(f)
    scaler_mean = np.array(scaler_dict['mean'], dtype=np.float32)
    scaler_scale = np.array(scaler_dict['scale'], dtype=np.float32)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

def preprocess_data(data, mean, scale, encoder):
    """
    Final, correct preprocessing function.
    - If label is an int (from app data), it's already encoded. Use it directly.
    - If label is a string (from validation data), transform it to its integer equivalent.
    """
    processed = []
    for item in data:
        raw_window = np.array(item['window'], dtype=np.float32)
        scaled_window = (raw_window - mean) / scale
        
        label_value = item['label']
        encoded_label = -1 # Default to an invalid label

        if isinstance(label_value, int):
            # Case 1: The label is already an integer (e.g., 3).
            # This is the correct encoded format. Use it directly.
            encoded_label = label_value
        else:
            # Case 2: The label is a string (e.g., "D").
            # Use the encoder to transform it into its integer index.
            encoded_label = encoder.transform([label_value])[0]

        if encoded_label == -1:
            raise ValueError(f"Could not process label: {label_value}")

        processed.append((scaled_window, encoded_label))
    return processed

# --- Load and partition client data ---
with open(CLIENT_DATA_PATH, 'r') as f:
    all_client_data_raw = json.load(f)
random.shuffle(all_client_data_raw)
client_data_partitions_raw = np.array_split(all_client_data_raw, NUM_CLIENTS)
client_datasets = [preprocess_data(p.tolist(), scaler_mean, scaler_scale, label_encoder) for p in client_data_partitions_raw]
print(f"Distributed BALANCED data among {NUM_CLIENTS} clients. Sizes: {[len(d) for d in client_datasets]}")

# Load and preprocess the global validation set
with open(VALIDATION_DATA_PATH, 'r') as f:
    validation_data_raw = json.load(f)
global_validation_set = preprocess_data(validation_data_raw, scaler_mean, scaler_scale, label_encoder)
val_x = np.array([d[0] for d in global_validation_set], dtype=np.float32)
val_y = np.array([d[1] for d in global_validation_set], dtype=np.int32)


# --- 3. Define the TFLite Client Training Logic ---
print("--- [3/6] Defining TFLite client update function ---")
def tflite_client_update(client_id, client_data, global_weights_flat, harness):
    print(f"\n  >> Client {client_id} starting TFLite training...")
    harness.set_weights(flat_weights=global_weights_flat)
    
    random.shuffle(client_data)
    train_x = np.array([d[0] for d in client_data], dtype=np.float32)
    train_y = np.array([d[1] for d in client_data], dtype=np.int32)
    
    pre_loss, pre_acc = harness.evaluate(train_x, train_y)
    print(f"  Client {client_id} | Pre-Train -> Loss: {pre_loss:.4f}, Accuracy: {pre_acc:.2%}")
    
    for epoch in range(LOCAL_EPOCHS):
        for i in range(0, len(train_x), BATCH_SIZE):
            x_batch = train_x[i:i+BATCH_SIZE]
            y_batch = train_y[i:i+BATCH_SIZE]
            if len(x_batch) == 0: continue
            harness.train_step(x_input=x_batch, y_batch=y_batch)

    post_loss, post_acc = harness.evaluate(train_x, train_y)
    print(f"  Client {client_id} | Post-Train -> Loss: {post_loss:.4f}, Accuracy: {post_acc:.2%}")
    
    return harness.get_weights(dummy_input=tf.constant(0.0))['weights'], len(train_x)

# --- 4. Run Federated Learning Simulation ---
print("\n--- [4/6] Starting TFLite Federated Learning Simulation ---")
global_harness = TFLiteHarness(TFLITE_MODEL_PATH)
global_weights_flat = global_harness.get_weights(dummy_input=tf.constant(0.0))['weights']
global_performance_history = []

for round_num in range(1, NUM_ROUNDS + 1):
    print(f"\n{'='*20} ROUND {round_num}/{NUM_ROUNDS} {'='*20}")
    
    client_updates_flat = []
    total_samples = 0
    # Use a new harness for each client to simulate isolation
    for i in range(NUM_CLIENTS):
        client_harness = TFLiteHarness(TFLITE_MODEL_PATH)
        weights_flat, n_samples = tflite_client_update(i + 1, client_datasets[i], global_weights_flat, client_harness)
        client_updates_flat.append((weights_flat, n_samples))
        total_samples += n_samples
        
    # Federated Averaging
    if total_samples == 0: continue
    aggregated_weights_flat = np.zeros_like(global_weights_flat)
    for weights_flat, n_samples in client_updates_flat:
        aggregated_weights_flat += weights_flat * (n_samples / total_samples)
    global_weights_flat = aggregated_weights_flat
    
    print("\n  >> Server: Aggregated client weights.")
    global_harness.set_weights(flat_weights=global_weights_flat)
    
    loss, acc = global_harness.evaluate(val_x, val_y)
    global_performance_history.append({'round': round_num, 'loss': loss, 'accuracy': acc})
    print(f"  >> Global Model | Round {round_num} Eval -> Loss: {loss:.4f}, Accuracy: {acc:.2%}")

# --- 5 & 6. Reporting ---
print("\n--- [5/6] FL Simulation Complete. Generating Report ---")
import pandas as pd
performance_df = pd.DataFrame(global_performance_history)
print(performance_df.to_string(index=False))

performance_df.to_csv(SIMULATION_RESULTS_DIR / 'tflite_simulation_performance.csv', index=False)
print(f"\nPerformance log saved to '{SIMULATION_RESULTS_DIR / 'tflite_simulation_performance.csv'}'")
print("\n--- [6/6] Simulation finished. ---")