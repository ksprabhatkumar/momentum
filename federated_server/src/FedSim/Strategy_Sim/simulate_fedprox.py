# FedSim/Strategy_Sim/simulate_fedprox.py (FINAL - Correct Evaluation Logic)

import json
import numpy as np
import tensorflow as tf
from pathlib import Path
import random
from sklearn.metrics import accuracy_score, f1_score
import os

# Suppress TensorFlow informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Simulation Hyperparameters ---
NUM_ROUNDS = 5
CLIENTS_PER_ROUND = 4
LOCAL_EPOCHS = 1
FEDPROX_MU = 0.01

# --- Define Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent
MODELS_DIR = SCRIPT_DIR.parent / "Models"
CLIENT_DATA_DIR = BASE_DIR / "client_data"
TEST_SET_DIR = BASE_DIR / "definitive_test_set"

# --- Model Building Functions ---
from tensorflow.keras.layers import Input, Add, Conv1D, BatchNormalization, Activation, SpatialDropout1D, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.models import Model

def build_tcn_model(input_shape, num_classes):
    # This architecture must match the one used for training
    def residual_block(x, dilation_rate):
        prev_x = x; num_filters = 64; kernel_size = 7; spatial_dropout_rate = 0.15
        conv1 = Conv1D(filters=num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='same')(x)
        conv1 = BatchNormalization()(conv1); conv1 = Activation('relu')(conv1); conv1 = SpatialDropout1D(spatial_dropout_rate)(conv1)
        conv2 = Conv1D(filters=num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='same')(conv1)
        conv2 = BatchNormalization()(conv2); conv2 = Activation('relu')(conv2); conv2 = SpatialDropout1D(spatial_dropout_rate)(conv2)
        if prev_x.shape[-1] != conv2.shape[-1]: prev_x = Conv1D(num_filters, 1, padding='same')(prev_x)
        return Add()([prev_x, conv2])
    input_layer = Input(shape=input_shape)
    x = input_layer
    for rate in [2 ** i for i in range(5)]: x = residual_block(x, rate)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.3)(x)
    output_layer = Dense(num_classes, activation="softmax")(x)
    return Model(inputs=input_layer, outputs=output_layer)

def load_all_client_data_and_scale():
    """Loads, scales, and prepares client data, returning it as separate X/y arrays."""
    print("1. Loading scaler, labels, and all client data...")
    with open(MODELS_DIR / "scaler.json", 'r') as f: scaler_params = json.load(f)
    scaler_mean = np.array(scaler_params['mean'], dtype=np.float32)
    scaler_scale = np.array(scaler_params['scale'], dtype=np.float32)
    with open(MODELS_DIR / "labels.json", 'r') as f: labels_map_str_to_int = {v: int(k) for k, v in json.load(f).items()}
    
    clients = []
    for json_file in sorted(CLIENT_DATA_DIR.glob("*.json")):
        client_name = json_file.stem
        with open(json_file, 'r') as f: data = json.load(f)
        
        client_windows, client_labels = [], []
        for window in data:
            window_data_str, label_str = window.get("window_data_json"), window.get("label")
            if window_data_str and label_str:
                scaled_window = (np.array(json.loads(window_data_str), dtype=np.float32) - scaler_mean) / scaler_scale
                label_int = labels_map_str_to_int.get(label_str)
                if label_int is not None and scaled_window.shape == (60, 6):
                    client_windows.append(scaled_window)
                    client_labels.append(label_int)

        if client_windows:
            clients.append({
                "name": client_name, 
                "X_local": np.array(client_windows),
                "y_local": np.array(client_labels)
            })
            print(f"  > Loaded and scaled {len(client_windows)} samples for client '{client_name}'")
    return clients

def evaluate_model(flat_weights, eval_model_template, X_data, y_data):
    """A general function to evaluate a set of flat weights on any given dataset, returning accuracy and F1 score."""
    # Extract only the Keras model weights (excluding optimizer state)
    model_weights_size = sum(np.prod(w.shape) for w in eval_model_template.get_weights())
    model_weights_flat = flat_weights[:model_weights_size]
    
    # Unflatten and set the weights
    structured_weights = []
    start_idx = 0
    for w_template in eval_model_template.get_weights():
        size = np.prod(w_template.shape)
        chunk = model_weights_flat[start_idx : start_idx + size].reshape(w_template.shape)
        structured_weights.append(chunk)
        start_idx += size
    eval_model_template.set_weights(structured_weights)

    y_pred = np.argmax(eval_model_template.predict(X_data, verbose=0), axis=1)
    
    # Calculate metrics
    acc = accuracy_score(y_data, y_pred)
    # Use 'macro' average for F1 to treat all classes equally. zero_division=0 prevents errors for classes with no predictions.
    f1 = f1_score(y_data, y_pred, average='macro', zero_division=0)
    
    return acc, f1

def main():
    print("--- Starting FedProx Simulation with Correct Evaluation ---")
    print(f"Hyperparameters: Rounds={NUM_ROUNDS}, Clients/Round={CLIENTS_PER_ROUND}, Epochs={LOCAL_EPOCHS}, mu={FEDPROX_MU}\n")

    # 1. Load all datasets
    all_clients = load_all_client_data_and_scale()
    if not all_clients: print("FATAL: No client data found. Exiting."); return
        
    X_test = np.load(TEST_SET_DIR / "X_test_scaled.npy")
    y_test = np.load(TEST_SET_DIR / "y_test.npy")
    print(f"Definitive test set loaded. Shape: {X_test.shape}\n")

    # 2. Initialize models
    print("2. Initializing models...")
    interpreter = tf.lite.Interpreter(model_path=str(MODELS_DIR / "fedprox_trainable.tflite"))
    interpreter.allocate_tensors()
    get_weights = interpreter.get_signature_runner('get_weights_flat')
    set_weights = interpreter.get_signature_runner('set_weights_flat')
    train_step = interpreter.get_signature_runner('train_step')
    snapshot_weights = interpreter.get_signature_runner('snapshot_weights')
    
    eval_model = build_tcn_model(input_shape=(X_test.shape[1], X_test.shape[2]), num_classes=5)
    
    initial_weights = np.fromfile(MODELS_DIR / "initial_weights_fedprox.bin", dtype=np.float32)
    set_weights(flat_weights=initial_weights)
    print("TFLite model initialized with clean weights.\n")

    # 3. Main simulation loop
    global_weights = initial_weights
    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"--- Round {round_num}/{NUM_ROUNDS} ---")
        
        selected_clients = random.sample(all_clients, min(CLIENTS_PER_ROUND, len(all_clients)))
        print(f"Selected clients: {[c['name'] for c in selected_clients]}")
        
        client_updates = []
        for client in selected_clients:
            print(f"  -> Processing client '{client['name']}' ({len(client['X_local'])} samples)...")
            set_weights(flat_weights=global_weights)
            
            # Pre-Train Evaluation on LOCAL data
            pre_train_acc, pre_train_f1 = evaluate_model(global_weights, eval_model, client['X_local'], client['y_local'])
            print(f"    > Pre-Train Local -> Acc: {pre_train_acc:.2%}, F1 (Macro): {pre_train_f1:.2f}")
            
            # Local Training
            snapshot_weights(dummy_input=np.float32(0.0))
            for _ in range(LOCAL_EPOCHS):
                for x_sample, y_sample in zip(client['X_local'], client['y_local']):
                    train_step(x_input=np.expand_dims(x_sample, axis=0), 
                               y_batch=np.array([y_sample], dtype=np.int32),
                               mu=np.float32(FEDPROX_MU))
            
            updated_weights = get_weights(dummy_input=np.float32(0.0))['weights']
            client_updates.append((updated_weights, len(client['X_local'])))

            # Post-Train Evaluation on LOCAL data
            post_train_acc, post_train_f1 = evaluate_model(updated_weights, eval_model, client['X_local'], client['y_local'])
            print(f"    > Post-Train Local -> Acc: {post_train_acc:.2%}, F1 (Macro): {post_train_f1:.2f}")

        # Aggregation
        total_samples = sum(num_samples for _, num_samples in client_updates)
        new_global_weights = np.zeros_like(global_weights)
        for weights, num_samples in client_updates:
            new_global_weights += weights * (num_samples / total_samples)
        global_weights = new_global_weights
        
        # Final GLOBAL Evaluation for the round on the TEST SET
        print("  Aggregating and evaluating global model...")
        round_acc, round_f1 = evaluate_model(global_weights, eval_model, X_test, y_test)
        print(f"✅ Round {round_num} Complete. Aggregated Test Set -> Accuracy: {round_acc:.2%}, F1 (Macro): {round_f1:.2f}\n")

if __name__ == "__main__":
    main()