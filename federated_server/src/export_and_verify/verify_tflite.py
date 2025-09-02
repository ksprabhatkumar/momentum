# export_and_verify/verify_tflite.py (FINAL, MODULARIZED VERSION)
import numpy as np
import tensorflow as tf
import json
from pathlib import Path
import random

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --- NEW: Paths relative to this script's location ---
SCRIPT_DIR = Path(__file__).parent
ASSETS_DIR = SCRIPT_DIR / "generated_assets" # Read from the output of export.py
TFLITE_MODEL_PATH = ASSETS_DIR / "tcn_v2_ABCDE_trainable.tflite"
SCALER_PATH = ASSETS_DIR / "scaler.json"
INITIAL_WEIGHTS_PATH = ASSETS_DIR / "initial_weights.bin"
LABELED_DATA_PATH = SCRIPT_DIR / ".." / "client_data" / "sample_data_for_app.json"

def load_and_preprocess_data(scaler_path, data_path):
    print(f"Loading and preprocessing data from {data_path.name}...")
    with open(scaler_path, 'r') as f:
        scaler = json.load(f)
        mean = np.array(scaler['mean'], dtype=np.float32)
        scale = np.array(scaler['scale'], dtype=np.float32)

    with open(data_path, 'r') as f:
        client_data = json.load(f)

    processed_data = []
    for sample in client_data:
        raw_window = np.array(sample['window'], dtype=np.float32)
        scaled_window = (raw_window - mean) / scale
        processed_data.append((scaled_window, sample['label']))
    
    print(f"Loaded and scaled {len(processed_data)} total samples.")
    return processed_data

def evaluate_tflite_model(infer_signature, eval_data):
    if not eval_data:
        return 0.0, 0.0
    total_loss, correct_predictions = 0, 0
    for window, label in eval_data:
        input_tensor = np.expand_dims(window, axis=0).astype(np.float32)
        output = infer_signature(x_input=input_tensor)
        logits = output['logits'][0]
        y_true = tf.constant([label], dtype=tf.int32)
        y_pred = tf.expand_dims(tf.constant(logits), axis=0)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True).numpy()[0]
        total_loss += loss
        predicted_class = np.argmax(tf.nn.softmax(logits).numpy())
        if predicted_class == label:
            correct_predictions += 1
    return total_loss / len(eval_data), correct_predictions / len(eval_data)

def main():
    print(f"Loading TFLite model from: {TFLITE_MODEL_PATH}")
    if not TFLITE_MODEL_PATH.exists():
        print(f"❌ TFLite model not found in '{ASSETS_DIR}'. Please run export.py first.")
        return

    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_MODEL_PATH))
    interpreter.allocate_tensors()
    
    infer_signature = interpreter.get_signature_runner('infer')
    train_step_signature = interpreter.get_signature_runner('train_step')
    set_weights_signature = interpreter.get_signature_runner('set_weights_flat')
    print("✅ TFLite model and signatures loaded successfully.")

    print(f"Loading initial weights from: {INITIAL_WEIGHTS_PATH}")
    if not INITIAL_WEIGHTS_PATH.exists():
        print(f"❌ Initial weights file not found in '{ASSETS_DIR}'. Please run export.py first.")
        return
        
    initial_weights = np.fromfile(INITIAL_WEIGHTS_PATH, dtype=np.float32)
    set_weights_signature(flat_weights=initial_weights)
    print("✅ Model variables initialized with initial weights.")

    all_data = load_and_preprocess_data(SCALER_PATH, LABELED_DATA_PATH)
    
    random.shuffle(all_data)
    round_data = all_data[:1000]
    split_index = int(len(round_data) * 0.8)
    train_data = round_data[:split_index]
    eval_data = round_data[split_index:]

    print(f"\n{'='*20} TFLITE SIMULATION ROUND 1 {'='*20}")
    print(f"Using {len(round_data)} samples: {len(train_data)} for training, {len(eval_data)} for evaluation.")

    pre_loss, pre_acc = evaluate_tflite_model(infer_signature, eval_data)
    print(f"  > Pre-Train Eval -> Loss: {pre_loss:.4f}, Accuracy: {pre_acc*100:.2f}%")

    for window, label in train_data:
        input_tensor = np.expand_dims(window, axis=0).astype(np.float32)
        label_tensor = np.array([label], dtype=np.int32)
        train_step_signature(x_input=input_tensor, y_batch=label_tensor)
        
    print(f"  > Local training on {len(train_data)} samples complete.")

    post_loss, post_acc = evaluate_tflite_model(infer_signature, eval_data)
    print(f"  > Post-Train Eval -> Loss: {post_loss:.4f}, Accuracy: {post_acc*100:.2f}%")

    print(f"\n{'='*20} VERIFICATION COMPLETE {'='*20}")
    print("If the post-train accuracy is high and stable, the TFLite model is correct.")

if __name__ == "__main__":
    main()