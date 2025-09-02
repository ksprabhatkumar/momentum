# /server_deployment/custom_fl_server_with_test_set_eval.py
# FINAL VERSION: A simplified server that performs ground-truth evaluation on a
# definitive test set after each round and plots the results.

import numpy as np
from flask import Flask, request, jsonify, send_from_directory, Response
from waitress import serve
from pathlib import Path
import base64
import time
import threading
import json

# --- NEW: Imports for Model Building and Evaluation ---
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Add, Conv1D, BatchNormalization, Activation, SpatialDropout1D, GlobalAveragePooling1D, Dense
from sklearn.metrics import accuracy_score

# Matplotlib must be configured to a non-GUI backend for server use
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =================================================================================
#                            EXPERIMENT HYPERPARAMETERS
# =================================================================================
HOST = "0.0.0.0"
PORT = 8080
NUM_EXPECTED_CLIENTS = 4 # As per your client logs
NUM_ROUNDS = 3
LOCAL_EPOCHS = 1
# =================================================================================

# --- Asset Paths ---
SCRIPT_DIR = Path(__file__).parent
INITIAL_WEIGHTS_PATH = SCRIPT_DIR / "initial_weights.bin"
DEPLOYMENT_ASSETS_DIR = SCRIPT_DIR / "deployment_assets"
DEPLOYMENT_ASSETS_DIR.mkdir(exist_ok=True)

# --- Paths for Metrics, Plots, and Test Data ---
PLOTS_DIR = SCRIPT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
METRICS_FILE = SCRIPT_DIR / "fl_metrics_history.json"
TEST_SET_DIR = SCRIPT_DIR / ".." / "definitive_test_set" # Relative path to test set
X_TEST_PATH = TEST_SET_DIR / "X_test_scaled.npy"
Y_TEST_PATH = TEST_SET_DIR / "y_test.npy"


# --- Server State (Thread-Safe) ---
SERVER_STATE_LOCK = threading.Lock()
SERVER_STATE = {
    "global_model_weights": None, "initial_weights": None, "current_round": 1,
    "server_status": "INITIALIZING", "clients_in_round": {}, "client_updates_in_round": [],
}
METRICS_LOCK = threading.Lock()
METRICS_HISTORY = {"rounds": []}

# --- NEW: Global variables to hold the definitive test set in memory ---
X_TEST_GLOBAL = None
Y_TEST_GLOBAL = None

app = Flask(__name__)

# ==============================================================================
#      MODEL ARCHITECTURE & HELPERS (Copied from your verification script)
# ==============================================================================
def residual_block(x, dilation_rate):
    prev_x = x; num_filters = 64; kernel_size = 7; spatial_dropout_rate = 0.15
    conv1 = Conv1D(filters=num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='same')(x)
    conv1 = BatchNormalization()(conv1); conv1 = Activation('relu')(conv1); conv1 = SpatialDropout1D(spatial_dropout_rate)(conv1)
    conv2 = Conv1D(filters=num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='same')(conv1)
    conv2 = BatchNormalization()(conv2); conv2 = Activation('relu')(conv2); conv2 = SpatialDropout1D(spatial_dropout_rate)(conv2)
    if prev_x.shape[-1] != conv2.shape[-1]: prev_x = Conv1D(num_filters, 1, padding='same')(prev_x)
    return Add()([prev_x, conv2])

def build_tcn_model(input_shape, num_classes):
    input_layer = Input(shape=input_shape)
    x = input_layer
    for rate in [2 ** i for i in range(5)]: x = residual_block(x, rate)
    x = GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=input_layer, outputs=output_layer)

def unflatten_weights_for_model(model, flat_weights):
    """Reconstructs the structured list of weights for a Keras model from a flat numpy array."""
    structured_weights = []
    start_idx = 0
    for layer in model.layers:
        layer_weight_shapes = [w.shape for w in layer.get_weights()]
        if not layer_weight_shapes:
            continue
        
        layer_weights_list = []
        for shape in layer_weight_shapes:
            size = np.prod(shape)
            chunk = flat_weights[start_idx : start_idx + size]
            layer_weights_list.append(chunk.reshape(shape))
            start_idx += size
        
        # This is required because model.set_weights needs the exact list structure
        for i in range(len(layer.get_weights())):
             structured_weights.append(layer_weights_list[i])
             
    return structured_weights

# =================================================================================
#                   METRICS, PLOTTING & EVALUATION
# =================================================================================

def evaluate_and_plot_metrics(round_num, global_weights_flat):
    """
    Evaluates the current global model on the definitive test set and generates plots.
    """
    global X_TEST_GLOBAL, Y_TEST_GLOBAL
    if X_TEST_GLOBAL is None or Y_TEST_GLOBAL is None:
        print("⚠️ Test set not loaded. Skipping evaluation.")
        return

    print(f"\nEvaluating Global Model for Round {round_num} on Definitive Test Set...")
    try:
        # 1. Build model and load the newly aggregated weights
        model = build_tcn_model(input_shape=(X_TEST_GLOBAL.shape[1], X_TEST_GLOBAL.shape[2]), num_classes=5)
        structured_weights = unflatten_weights_for_model(model, global_weights_flat)
        model.set_weights(structured_weights)
        
        # 2. Run prediction and calculate accuracy
        y_pred_probs = model.predict(X_TEST_GLOBAL, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        test_accuracy = accuracy_score(Y_TEST_GLOBAL, y_pred)
        
        print(f"✅ True Test Set Accuracy: {test_accuracy:.2%}")

        # 3. Save this true metric to history
        with METRICS_LOCK:
            round_data = next((r for r in METRICS_HISTORY["rounds"] if r["round"] == round_num), None)
            if round_data is not None:
                round_data["aggregates"] = {"test_set_accuracy": test_accuracy}
            
            # 4. Generate and save the plot using all historical true accuracies
            rounds_with_data = sorted([r for r in METRICS_HISTORY["rounds"] if "aggregates" in r], key=lambda x: x["round"])
            round_numbers = [r["round"] for r in rounds_with_data]
            accuracies = [r["aggregates"]["test_set_accuracy"] * 100 for r in rounds_with_data]
            
            plt.figure(figsize=(8, 4))
            plt.plot(round_numbers, accuracies, marker='o', linestyle='-')
            plt.title("Global Model Accuracy on Definitive Test Set")
            plt.xlabel("Federated Round")
            plt.ylabel("Accuracy (%)")
            plt.xticks(round_numbers)
            plt.grid(True)
            plt.ylim(0, 101) # Set y-axis from 0 to 100
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / "test_set_accuracy_plot.png")
            plt.close()

        METRICS_FILE.write_text(json.dumps(METRICS_HISTORY, indent=2))
        print(f"📈 Plot updated and saved to '{PLOTS_DIR.resolve()}'")

    except Exception as e:
        print(f"❌ Error during test set evaluation: {e}")

# =================================================================================
#                       SERVER STATE & WEIGHT MANAGEMENT
# =================================================================================
# --- No changes in this section ---
def load_initial_weights(is_reset=False):
    # ... (function is unchanged)
    with SERVER_STATE_LOCK:
        action = "Resetting" if is_reset else "Loading initial"
        print(f"--- {action} model state from binary file ---")
        try:
            if SERVER_STATE["initial_weights"] is None:
                if not INITIAL_WEIGHTS_PATH.exists():
                    raise FileNotFoundError(f"'{INITIAL_WEIGHTS_PATH.name}' not found.")
                weight_bytes = INITIAL_WEIGHTS_PATH.read_bytes()
                SERVER_STATE["initial_weights"] = np.frombuffer(weight_bytes, dtype=np.float32)

            SERVER_STATE["global_model_weights"] = SERVER_STATE["initial_weights"].copy()
            SERVER_STATE["current_round"] = 1
            SERVER_STATE["clients_in_round"].clear()
            SERVER_STATE["client_updates_in_round"].clear()
            SERVER_STATE["server_status"] = "WAITING_FOR_CLIENTS"
            
            print(f"✅ State reset. Array shape: {SERVER_STATE['global_model_weights'].shape}")
            print(f"Server is now WAITING for {NUM_EXPECTED_CLIENTS} client(s) to join Round 1.")
            return True
        except Exception as e:
            print(f"❌ CRITICAL ERROR: Could not load initial state. {e}")
            SERVER_STATE["server_status"] = "ERROR"
            return False

# =================================================================================
#                   FEDERATED LEARNING ENDPOINTS (with corrected logic)
# =================================================================================

@app.route('/client-check-in', methods=['POST'])
def client_check_in():
    with SERVER_STATE_LOCK:
        client_id = request.json.get('client_id')
        status = SERVER_STATE["server_status"]
        current_round = SERVER_STATE["current_round"]
        
        # Prevent clients who have already submitted from re-joining the same round
        if SERVER_STATE["clients_in_round"].get(client_id) == "submitted":
            return jsonify({"action": "WAIT"})

        if status == "AGGREGATING": return jsonify({"action": "WAIT"})
        if current_round > NUM_ROUNDS or status == "COMPLETE": return jsonify({"action": "COMPLETE"})

        if status == "WAITING_FOR_CLIENTS":
            if client_id not in SERVER_STATE["clients_in_round"]:
                SERVER_STATE["clients_in_round"][client_id] = "joined"
        
        if len(SERVER_STATE["clients_in_round"]) >= NUM_EXPECTED_CLIENTS:
            weights_b64 = base64.b64encode(SERVER_STATE["global_model_weights"].tobytes()).decode('utf-8')
            return jsonify({
                "action": "TRAIN", "round": current_round, 
                "data": {"weights": weights_b64, "config": {"local_epochs": LOCAL_EPOCHS}}
            })

        return jsonify({"action": "WAIT"})

@app.route('/submit-weights', methods=['POST'])
def submit_weights():
    with SERVER_STATE_LOCK:
        client_id = request.json.get('client_id')
        if SERVER_STATE["clients_in_round"].get(client_id) != "joined":
            return jsonify({"status": "error", "message": "Client not in 'joined' state."}), 400

        weights_b64 = request.json.get('weights'); metrics = request.json.get('metrics', {})
        weights_array = np.frombuffer(base64.b64decode(weights_b64), dtype=np.float32)
        num_samples = metrics.get('num_samples', 1)
        SERVER_STATE["client_updates_in_round"].append((weights_array, num_samples))
        SERVER_STATE["clients_in_round"][client_id] = "submitted" # Mark as submitted

        print(f"[{time.ctime()}] Received submission from '{client_id}' ({len(SERVER_STATE['client_updates_in_round'])}/{NUM_EXPECTED_CLIENTS})")

        if len(SERVER_STATE["client_updates_in_round"]) >= NUM_EXPECTED_CLIENTS:
            round_num = SERVER_STATE['current_round']
            print(f"✅ All submissions received for Round {round_num}. Aggregating...")
            
            total_samples = sum(item[1] for item in SERVER_STATE["client_updates_in_round"]) or 1
            weighted_updates = [item[0] * (item[1] / total_samples) for item in SERVER_STATE["client_updates_in_round"]]
            new_global_weights = np.sum(weighted_updates, axis=0)
            SERVER_STATE["global_model_weights"] = new_global_weights
            
            # --- MODIFIED: Call the new evaluation function ---
            evaluate_and_plot_metrics(round_num, new_global_weights)
            
            SERVER_STATE["current_round"] += 1
            SERVER_STATE["clients_in_round"].clear(); SERVER_STATE["client_updates_in_round"].clear()

            if SERVER_STATE["current_round"] > NUM_ROUNDS:
                SERVER_STATE["server_status"] = "COMPLETE"
                np.save(SCRIPT_DIR / "final_full_state_model.npy", new_global_weights)
                print("🏁 All rounds complete! Final model saved.")
            else:
                SERVER_STATE["server_status"] = "WAITING_FOR_CLIENTS"
                print(f"Server now WAITING for clients for Round {SERVER_STATE['current_round']}.")
        
        return jsonify({"status": "success"})

# ... (Other endpoints like /reset_demo, /model_info etc. are unchanged)
# =================================================================================
#                   DASHBOARD & PLOTTING ENDPOINTS
# =================================================================================
@app.route('/dashboard')
def dashboard():
    timestamp = int(time.time())
    return Response(f"""
    <html>
        <head><title>FL Dashboard</title><meta http-equiv="refresh" content="15"></head>
        <body style="font-family: sans-serif; text-align: center;">
            <h1>Federated Learning Ground-Truth Dashboard</h1>
            <p>(Auto-refreshes every 15 seconds)</p>
            <h2>Accuracy on Definitive Test Set</h2>
            <img src="/plots/test_set_accuracy_plot.png?v={timestamp}" alt="Plot not generated yet.">
        </body>
    </html>
    """, mimetype='text/html')

@app.route('/plots/<filename>')
def serve_plot(filename): return send_from_directory(PLOTS_DIR, filename)

# =================================================================================
#                                 SERVER STARTUP
# =================================================================================
if __name__ == '__main__':
    # --- NEW: Load the test set on startup ---
    try:
        print("--- Loading definitive test set into memory ---")
        X_TEST_GLOBAL = np.load(X_TEST_PATH)
        Y_TEST_GLOBAL = np.load(Y_TEST_PATH)
        print(f"✅ Test set loaded successfully. Shape: {X_TEST_GLOBAL.shape}")
    except FileNotFoundError:
        print(f"❌ FATAL: Could not find test set files in '{TEST_SET_DIR}'.")
        print("   Evaluation on test set will be disabled.")
        X_TEST_GLOBAL, Y_TEST_GLOBAL = None, None

    if load_initial_weights():
        print("\n--- FL Server with Test Set Evaluation ---")
        print(f" > Total Rounds:         {NUM_ROUNDS}")
        print(f" > Clients per Round:    {NUM_EXPECTED_CLIENTS}")
        print(f"-----------------------------------")
        print(f"\n🚀 Server starting on http://{HOST}:{PORT}")
        print(f"   - View the live dashboard at http://127.0.0.1:{PORT}/dashboard")
        serve(app, host=HOST, port=PORT, threads=8)