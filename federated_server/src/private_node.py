# private_node_streaming.py
# A simple, real-time streaming server for live predictions from the Momentum app.

from flask import Flask, request, jsonify, render_template_string, Response
import time
import json
import queue
import threading

# --- Configuration ---
HOST = "0.0.0.0"
PORT = 5000

# --- In-Memory, Thread-Safe Queue ---
predictions_queue = queue.Queue()

app = Flask(__name__)

# =================================================================================
#               BACKEND API & FRONTEND UI (THESE ARE UNCHANGED)
# =================================================================================

@app.route('/live_prediction', methods=['POST'])
def receive_live_prediction():
    try:
        payload = request.get_json()
        prediction = payload.get('prediction')
        if prediction:
            data_packet = {
                "prediction": prediction,
                "time": time.strftime("%H:%M:%S", time.localtime())
            }
            predictions_queue.put(data_packet)
            return jsonify({"status": "ok"}), 200
        else:
            return jsonify({"error": "Missing 'prediction' key in payload"}), 400
    except Exception as e:
        print(f"Error processing /live_prediction: {e}")
        return jsonify({"error": "Invalid JSON payload"}), 400

@app.route('/stream')
def stream():
    def event_stream():
        print("A web client connected to the stream.")
        try:
            while True:
                data = predictions_queue.get()
                sse_formatted_data = f"data: {json.dumps(data)}\n\n"
                yield sse_formatted_data
        except GeneratorExit:
            print("A web client disconnected from the stream.")
    return Response(event_stream(), mimetype='text/event-stream')

@app.route('/')
def index():
    return render_template_string("""
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Momentum Live Monitor</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; line-height: 1.6; background-color: #121212; color: #e0e0e0; max-width: 900px; margin: auto; padding: 20px; }
            .container { background-color: #1e1e1e; padding: 25px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.3); }
            h1 { color: #bb86fc; border-bottom: 2px solid #373737; padding-bottom: 10px; }
            #log-container { margin-top: 20px; height: 60vh; background-color: #000; border: 1px solid #373737; border-radius: 5px; overflow-y: scroll; padding: 15px; font-family: "SF Mono", "Fira Code", "Source Code Pro", monospace; font-size: 14px; }
            .log-entry { padding: 4px 0; border-bottom: 1px solid #2a2a2a; }
            .log-entry:last-child { border-bottom: none; }
            .time { color: #03dac6; margin-right: 15px; }
            .prediction { color: #cf6679; font-weight: bold; }
        </style>
      </head>
      <body>
        <div class="container">
            <h1>Momentum Live Prediction Stream</h1>
            <p>This page will automatically display live predictions sent from the Momentum app. Make sure the "Share Live Predictions" toggle is enabled in the app's settings.</p>
            <div id="log-container">
                <div class="log-entry"><span class="time">--:--:--</span><span>Waiting for connection from app...</span></div>
            </div>
        </div>
        <script>
            const logContainer = document.getElementById('log-container');
            const activityMap = { "A": "Walking", "B": "Jogging", "C": "Using Stairs", "D": "Sitting", "E": "Standing" };
            const eventSource = new EventSource('/stream');
            eventSource.onmessage = function(event) {
                const data = JSON.parse(event.data);
                const entryDiv = document.createElement('div');
                entryDiv.className = 'log-entry';
                const timeSpan = document.createElement('span');
                timeSpan.className = 'time';
                timeSpan.textContent = `[${data.time}]`;
                const predictionSpan = document.createElement('span');
                predictionSpan.className = 'prediction';
                const activityName = activityMap[data.prediction] || data.prediction;
                predictionSpan.textContent = `Prediction: ${activityName}`;
                entryDiv.appendChild(timeSpan);
                entryDiv.appendChild(predictionSpan);
                logContainer.appendChild(entryDiv);
                logContainer.scrollTop = logContainer.scrollHeight;
            };
            eventSource.onerror = function(err) {
                console.error("EventSource failed:", err);
                const errorDiv = document.createElement('div');
                errorDiv.className = 'log-entry';
                errorDiv.style.color = 'red';
                errorDiv.textContent = 'Connection to server lost. Attempting to reconnect...';
                logContainer.appendChild(errorDiv);
                logContainer.scrollTop = logContainer.scrollHeight;
            };
        </script>
      </body>
    </html>
    """)

if __name__ == '__main__':
    print(f"--- Private Node Live Stream Server ---")
    print(f" > Listening on http://{HOST}:{PORT}")
    print(" > Open your web browser to this machine's IP address at port 5000.")
    print(" > Example: http://192.18.1.10:5000")
    print(" > Point the Momentum app's 'Private Debug Node IP' to this machine's IP.")
    from waitress import serve
    
    # ==============================================================================
    # THE FIX: Increase server capacity to handle high-frequency client requests.
    # ==============================================================================
    serve(
        app, 
        host=HOST, 
        port=PORT, 
        threads=8,                  # Increase worker threads to handle more requests in parallel.
        connection_limit=500,       # Increase the max number of simultaneous connections.
        channel_timeout=20          # Close idle connections faster to free up slots.
    )