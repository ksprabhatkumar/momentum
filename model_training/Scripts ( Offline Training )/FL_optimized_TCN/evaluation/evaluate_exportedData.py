# evaluate_unseen_data.py

import numpy as np
import json
from pathlib import Path
import joblib
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Define Paths and Load Artifacts ---
print("--- [1/4] Loading model and preprocessing artifacts ---")
SCRIPT_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = SCRIPT_DIR / ".." / "results"
UNSEEN_DATA_DIR = SCRIPT_DIR / ".." / "unseen_client_data"
EVALUATION_RESULTS_DIR = SCRIPT_DIR / ".." / "evaluation" / "evaluation_results"
EVALUATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Load the trained model, scaler, and label encoder
try:
    model = tf.keras.models.load_model(RESULTS_DIR / 'fl_optimized_model.keras')
    scaler = joblib.load(RESULTS_DIR / 'scaler.joblib')
    label_encoder = joblib.load(RESULTS_DIR / 'label_encoder.joblib')
    print("Model and artifacts loaded successfully.")
except Exception as e:
    print(f"Error loading artifacts: {e}")
    exit()

# --- 2. Load and Parse Unseen Client Data ---
print(f"--- [2/4] Loading and parsing data from '{UNSEEN_DATA_DIR}' ---")

files_to_evaluate = [
    UNSEEN_DATA_DIR / 'Ajay.json',
    UNSEEN_DATA_DIR / 'Prabhat.json',
    UNSEEN_DATA_DIR / 'Vikash.json'
]

all_windows = []
all_true_labels = []
client_names = []

for file_path in files_to_evaluate:
    if not file_path.exists():
        print(f"Warning: File not found, skipping: {file_path}")
        continue

    client_name = file_path.stem
    print(f"Processing data for client: {client_name}")
    with open(file_path, 'r') as f:
        data = json.load(f)

    for record in data:
        window_data = json.loads(record['window_data_json'])
        all_windows.append(window_data)
        all_true_labels.append(record['label'])
        client_names.append(client_name)

if not all_windows:
    print("No data found to evaluate. Exiting.")
    exit()

X_unseen = np.array(all_windows)
y_true_labels = np.array(all_true_labels)
print(f"Loaded {len(X_unseen)} windows from {len(files_to_evaluate)} clients.")

# --- 3. Preprocess Data and Make Predictions ---
print("--- [3/4] Scaling data and making predictions ---")

num_samples, timesteps, features = X_unseen.shape
X_unseen_reshaped = X_unseen.reshape(-1, features)
X_unseen_scaled_reshaped = scaler.transform(X_unseen_reshaped)
X_unseen_scaled = X_unseen_scaled_reshaped.reshape(num_samples, timesteps, features)
print("Data scaling complete.")

predictions_proba = model.predict(X_unseen_scaled)
y_pred_integers = np.argmax(predictions_proba, axis=1)
y_true_integers = label_encoder.transform(y_true_labels)

# --- 4. Evaluate and Report Results ---
print("--- [4/4] Generating evaluation report ---")

overall_accuracy = accuracy_score(y_true_integers, y_pred_integers)
print("\n" + "="*50)
print("      Overall Performance on Unseen Client Data")
print("="*50)
print(f"  - Overall Accuracy: {overall_accuracy:.4f}")
print("-" * 50)

# --- FIX IS HERE ---
# We explicitly provide the labels the model knows about (0, 1, 2, 3, 4)
# This ensures the report is generated correctly even if one label is missing in the test data.
num_classes = len(label_encoder.classes_)
expected_labels = np.arange(num_classes)

print("Classification Report:")
report = classification_report(
    y_true_integers,
    y_pred_integers,
    labels=expected_labels, # <-- THE FIX
    target_names=label_encoder.classes_
)
print(report)

report_path = EVALUATION_RESULTS_DIR / 'unseen_clients_report.txt'
with open(report_path, 'w') as f:
    f.write("Overall Performance on Unseen Client Data\n")
    f.write("="*50 + "\n")
    f.write(f"Overall Accuracy: {overall_accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
print(f"Evaluation report saved to '{report_path}'")

# Also add the `labels` parameter to the confusion matrix for consistency
cm = confusion_matrix(y_true_integers, y_pred_integers, labels=expected_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.title('Confusion Matrix - Unseen Client Data')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

cm_path = EVALUATION_RESULTS_DIR / 'unseen_clients_confusion_matrix.png'
plt.savefig(cm_path)
print(f"Confusion matrix saved to '{cm_path}'")
print("\n--- Evaluation complete ---")