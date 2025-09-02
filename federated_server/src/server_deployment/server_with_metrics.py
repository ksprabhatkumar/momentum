# /3_server_deployment/custom_fl_server_with_plotting.py
# A simplified server that adds metrics and plotting to the original, working FL logic.

import numpy as np
from flask import Flask, request, jsonify, send_from_directory, Response
from waitress import serve
from pathlib import Path
import base64
import time
import threading
import json

# Matplotlib must be configured to a non-GUI backend for server use
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =================================================================================
#                            EXPERIMENT HYPERPARAMETERS
# =================================================================================
HOST = "0.0.0.0"
PORT = 8080
NUM_EXPECTED_CLIENTS = 2
NUM_ROUNDS = 3
LOCAL_EPOCHS = 1
# =================================================================================

# --- Asset Paths ---
SCRIPT_DIR = Path(__file__).parent
INITIAL_WEIGHTS_PATH = SCRIPT_DIR / "initial_weights.bin"
DEPLOYMENT_ASSETS_DIR = SCRIPT_DIR / "deployment_assets"
DEPLOYMENT_ASSETS_DIR.mkdir(exist_ok=True)

# --- NEW: Paths for Metrics and Plots ---
PLOTS_DIR = SCRIPT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
METRICS_FILE = SCRIPT_DIR / "fl_metrics_history.json"

# --- Server State (Thread-Safe) ---
SERVER_STATE_LOCK = threading.Lock()
SERVER_STATE = {
    "global_model_weights": None,
    "initial_weights": None,
    "current_round": 1,
    "server_status": "INITIALIZING",
    "clients_in_round": {},
    "client_updates_in_round": [],
}

# --- NEW: Metrics History State (Thread-Safe) ---
METRICS_LOCK = threading.Lock()
METRICS_HISTORY = {"rounds": []}

app = Flask(__name__)


# =================================================================================
#                   NEW: METRICS & PLOTTING FUNCTIONS
# =================================================================================

def load_metrics_history():
    """Loads metrics from a JSON file at startup for persistence."""
    global METRICS_HISTORY
    with METRICS_LOCK:
        if METRICS_FILE.exists():
            try:
                METRICS_HISTORY = json.loads(METRICS_FILE.read_text())
                print("✅ Metrics history loaded from file.")
            except json.JSONDecodeError:
                print(f"⚠️ Could not read {METRICS_FILE.name}. Starting fresh.")
                METRICS_HISTORY = {"rounds": []}
        else:
             METRICS_HISTORY = {"rounds": []}


def save_metrics_history():
    """Saves the current metrics to the JSON file."""
    with METRICS_LOCK:
        METRICS_FILE.write_text(json.dumps(METRICS_HISTORY, indent=2))

def compute_and_plot_round_metrics(round_num):
    """
    Calculates aggregate metrics for the completed round and regenerates all plots.
    This is called right after the model weights are aggregated.
    """
    with METRICS_LOCK:
        # Find the data for the just-completed round
        current_round_data = next((r for r in METRICS_HISTORY["rounds"] if r["round"] == round_num), None)
        if not current_round_data or not current_round_data.get("clients"):
            return

        clients = current_round_data["clients"]
        total_samples = sum(c.get("num_samples", 1) for c in clients)
        if total_samples == 0: total_samples = 1

        # Calculate weighted average accuracy and loss
        post_acc = sum(c.get("post_eval_accuracy", 0.0) * c.get("num_samples", 1) for c in clients) / total_samples
        post_loss = sum(c.get("post_eval_loss", 0.0) * c.get("num_samples", 1) for c in clients) / total_samples
        
        current_round_data["aggregates"] = {
            "weighted_post_accuracy": post_acc,
            "weighted_post_loss": post_loss
        }
        print(f"📊 Metrics for Round {round_num}: Accuracy={post_acc:.2%}, Loss={post_loss:.4f}")
        
        # --- Now, generate plots using the entire history ---
        rounds_with_data = sorted([r for r in METRICS_HISTORY["rounds"] if "aggregates" in r], key=lambda x: x["round"])
        
        round_numbers = [r["round"] for r in rounds_with_data]
        accuracies = [r["aggregates"]["weighted_post_accuracy"] * 100 for r in rounds_with_data]
        losses = [r["aggregates"]["weighted_post_loss"] for r in rounds_with_data]
        
        # Accuracy Plot
        plt.figure(figsize=(8, 4))
        plt.plot(round_numbers, accuracies, marker='o', linestyle='-')
        plt.title("Model Accuracy per Round")
        plt.xlabel("Federated Round")
        plt.ylabel("Weighted Avg. Accuracy (%)")
        plt.xticks(round_numbers)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "accuracy_plot.png")
        plt.close()

        # Loss Plot
        plt.figure(figsize=(8, 4))
        plt.plot(round_numbers, losses, marker='o', linestyle='-')
        plt.title("Model Loss per Round")
        plt.xlabel("Federated Round")
        plt.ylabel("Weighted Avg. Loss")
        plt.xticks(round_numbers)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "loss_plot.png")
        plt.close()

    save_metrics_history()
    print(f"📈 Plots updated and saved to '{PLOTS_DIR.resolve()}'")


# =================================================================================
#                       SERVER STATE & WEIGHT MANAGEMENT
# =================================================================================
# --- This section is identical to your original custom_fl_server.py ---

def load_initial_weights(is_reset=False):
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
#                 MODEL MANAGEMENT ENDPOINTS (from central_server.py)
# =================================================================================
# --- Unchanged ---
@app.route('/model_info', methods=['GET'])
def get_model_info():
    try:
        return send_from_directory(DEPLOYMENT_ASSETS_DIR, 'model_info.json')
    except FileNotFoundError:
        return jsonify({"error": "model_info.json not found in deployment_assets"}), 404

@app.route('/download_model', methods=['GET'])
def download_model():
    try:
        return send_from_directory(DEPLOYMENT_ASSETS_DIR, 'inference_model.tflite')
    except FileNotFoundError:
        return jsonify({"error": "inference_model.tflite not found in deployment_assets"}), 404

@app.route('/download_scaler_for_app', methods=['GET'])
def download_scaler_for_app():
    try:
        return send_from_directory(DEPLOYMENT_ASSETS_DIR, 'scaler.json')
    except FileNotFoundError:
        return jsonify({"error": "scaler.json not found in deployment_assets"}), 404

# =================================================================================
#                   CUSTOM FEDERATED LEARNING ENDPOINTS
# =================================================================================
# --- These routes now include metric collection but retain the original logic ---

@app.route('/client-check-in', methods=['POST'])
def client_check_in():
    with SERVER_STATE_LOCK:
        client_id = request.json.get('client_id')
        status = SERVER_STATE["server_status"]
        current_round = SERVER_STATE["current_round"]
        
        # --- NEW: Check if this client has already submitted for this round ---
        client_status_in_round = SERVER_STATE["clients_in_round"].get(client_id)
        if client_status_in_round == "submitted":
            return jsonify({"action": "WAIT"}) # Force client to wait for next round

        if status == "AGGREGATING":
            return jsonify({"action": "WAIT"})
            
        if current_round > NUM_ROUNDS or status == "COMPLETE":
            return jsonify({"action": "COMPLETE"})

        if status == "WAITING_FOR_CLIENTS":
            # --- MODIFIED: Only mark as 'joined' if not already present ---
            if client_id not in SERVER_STATE["clients_in_round"]:
                SERVER_STATE["clients_in_round"][client_id] = "joined"
                print(f" > Client '{client_id}' joined Round {current_round}.")
        
        # Count how many clients have AT LEAST joined (not yet submitted)
        num_joined_clients = len(SERVER_STATE["clients_in_round"])

        if num_joined_clients >= NUM_EXPECTED_CLIENTS:
            weights_bytes = SERVER_STATE["global_model_weights"].tobytes()
            weights_b64 = base64.b64encode(weights_bytes).decode('utf-8')
            training_config = {"local_epochs": LOCAL_EPOCHS}
            
            return jsonify({
                "action": "TRAIN", 
                "round": current_round, 
                "data": { "weights": weights_b64, "config": training_config }
            })

        return jsonify({"action": "WAIT"})


@app.route('/submit-weights', methods=['POST'])
def submit_weights():
    with SERVER_STATE_LOCK:
        client_id = request.json.get('client_id')
        
        # --- MODIFIED: Check against the client's status, not just existence ---
        if SERVER_STATE["clients_in_round"].get(client_id) != "joined":
            return jsonify({"status": "error", "message": "Client not in a 'joined' state for this round."}), 400

        # ... (rest of the weight decoding is the same)
        weights_b64 = request.json.get('weights')
        metrics = request.json.get('metrics', {})
        weights_bytes = base64.b64decode(weights_b64)
        weights_array = np.frombuffer(weights_bytes, dtype=np.float32)
        num_samples = metrics.get('num_samples', 1)
        SERVER_STATE["client_updates_in_round"].append((weights_array, num_samples))
        
        # --- NEW: Mark this client as having submitted ---
        SERVER_STATE["clients_in_round"][client_id] = "submitted"

        
        # --- NEW: Collect rich metrics from this client ---
        with METRICS_LOCK:
            round_num = SERVER_STATE['current_round']
            round_data = next((r for r in METRICS_HISTORY["rounds"] if r["round"] == round_num), None)
            if round_data is None:
                round_data = {"round": round_num, "clients": []}
                METRICS_HISTORY["rounds"].append(round_data)
            
            # Add all metrics from the client payload
            client_metric_data = {"client_id": client_id, **metrics}
            round_data["clients"].append(client_metric_data)
        
        print(f"[{time.ctime()}] Received submission from '{client_id}' ({len(SERVER_STATE['client_updates_in_round'])}/{NUM_EXPECTED_CLIENTS})")

        # --- Aggregation logic is IDENTICAL to your original script ---
        if len(SERVER_STATE["client_updates_in_round"]) >= NUM_EXPECTED_CLIENTS:
            current_round_num = SERVER_STATE['current_round']
            print(f"✅ All submissions received for Round {current_round_num}. Aggregating...")
            
            total_samples = sum(item[1] for item in SERVER_STATE["client_updates_in_round"])
            if total_samples == 0: total_samples = 1
            
            weighted_updates = [item[0] * (item[1] / total_samples) for item in SERVER_STATE["client_updates_in_round"]]
            new_global_weights = np.sum(weighted_updates, axis=0)
            SERVER_STATE["global_model_weights"] = new_global_weights
            print(" > Global model updated via Federated Averaging.")

           
            compute_and_plot_round_metrics(current_round_num)
            
            SERVER_STATE["current_round"] += 1
            SERVER_STATE["clients_in_round"].clear()
            SERVER_STATE["client_updates_in_round"].clear()

            if SERVER_STATE["current_round"] > NUM_ROUNDS:
                SERVER_STATE["server_status"] = "COMPLETE"
                print("🏁 All federated rounds are complete!")
                final_model_path = SCRIPT_DIR / "final_full_state_model.npy"
                np.save(final_model_path, new_global_weights)
                print(f"✅ Final model state saved to '{final_model_path.name}'")
            else:
                SERVER_STATE["server_status"] = "WAITING_FOR_CLIENTS"
                print(f"Server now WAITING for clients to join Round {SERVER_STATE['current_round']}.")
        
        return jsonify({"status": "success"})


@app.route('/reset_demo', methods=['POST'])
def reset_demo():
    # --- Unchanged ---
    if load_initial_weights(is_reset=True):
        return jsonify({"message": "Server state reset successfully."}), 200
    else:
        return jsonify({"message": "Failed to reset server state."}), 500


@app.route('/dashboard')
def dashboard():
    """A simple HTML page to view the plots."""
  
    timestamp = int(time.time())
    return Response(f"""
    <html>
        <head>
            <title>FL Server Dashboard</title>
            <meta http-equiv="refresh" content="15">
        </head>
        <body style="font-family: sans-serif; text-align: center;">
            <h1>Federated Learning Dashboard</h1>
            <p>(This page auto-refreshes every 15 seconds)</p>
            <h2>Accuracy per Round</h2>
            <img src="/plots/accuracy_plot.png?v={timestamp}" alt="Accuracy Plot not available yet.">
            <h2>Loss per Round</h2>
            <img src="/plots/loss_plot.png?v={timestamp}" alt="Loss Plot not available yet.">
            <p><a href="/metrics">View Raw Metrics JSON</a></p>
        </body>
    </html>
    """, mimetype='text/html')

@app.route('/plots/<filename>')
def serve_plot(filename):
    """Serves the plot images."""
    return send_from_directory(PLOTS_DIR, filename)

@app.route('/metrics')
def serve_metrics():
    """Serves the raw metrics history."""
    with METRICS_LOCK:
        return jsonify(METRICS_HISTORY)


if __name__ == '__main__':
    load_metrics_history()
    if load_initial_weights():
        print("\n--- FL Experiment Configuration ---")
        print(f" > Total Rounds:         {NUM_ROUNDS}")
        print(f" > Clients per Round:    {NUM_EXPECTED_CLIENTS}")
        print(f" > Local Epochs:         {LOCAL_EPOCHS}")
        print(f"-----------------------------------")
        print(f"\n🚀 Custom FL Server with Plotting starting on http://{HOST}:{PORT}")
        print(f"   - View the live dashboard at http://127.0.0.1:{PORT}/dashboard")
        serve(app, host=HOST, port=PORT)