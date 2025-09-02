import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, SpatialDropout1D, Add, GlobalAveragePooling1D, Dropout, Dense
import json
from pathlib import Path
import random

# --- Configuration ---
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# --- Paths (relative to Momentum_Backend) ---
ASSETS_DIR = Path("./initialization")
LABELED_DATA_PATH = Path("./sample_data_for_app.json") # The data from your client

# --- Model Building (Identical to export.py) ---
def build_tcn_model(input_shape, num_classes, include_dropout=True):
    def residual_block(x, dilation_rate):
        prev_x = x; num_filters = 64; kernel_size = 7; spatial_dropout_rate = 0.15
        conv1 = Conv1D(filters=num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='same')(x)
        conv1 = BatchNormalization()(conv1); conv1 = Activation('relu')(conv1)
        if include_dropout: conv1 = SpatialDropout1D(spatial_dropout_rate)(conv1)
        conv2 = Conv1D(filters=num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='same')(conv1)
        conv2 = BatchNormalization()(conv2); conv2 = Activation('relu')(conv2)
        if include_dropout: conv2 = SpatialDropout1D(spatial_dropout_rate)(conv2)
        if prev_x.shape[-1] != conv2.shape[-1]: prev_x = Conv1D(num_filters, 1, padding='same')(prev_x)
        return Add()([prev_x, conv2])
    input_layer = Input(shape=input_shape, dtype=tf.float32)
    x = input_layer
    for rate in [2 ** i for i in range(5)]: x = residual_block(x, rate)
    x = GlobalAveragePooling1D()(x)
    if include_dropout: x = Dropout(0.3)(x)
    output_layer = Dense(num_classes, activation="softmax", dtype=tf.float32)(x)
    return Model(inputs=input_layer, outputs=output_layer)

# ====================================================================================
#  THE CRITICAL FIX: A new wrapper that freezes Batch Normalization during training
# ====================================================================================
class BNTrainableWrapper(tf.Module):
    def __init__(self, keras_model_path):
        super().__init__()
        print("Loading Keras model...")
        original_model = tf.keras.models.load_model(keras_model_path, compile=False)

        self.keras_model = build_tcn_model(
            input_shape=original_model.input_shape[1:],
            num_classes=original_model.output_shape[1],
            include_dropout=False
        )
        self.keras_model.set_weights(original_model.get_weights())
        print("Keras model loaded and weights set.")

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4) # Keep the lower learning rate
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    @tf.function
    def train_step(self, x_input, y_batch):
        with tf.GradientTape() as tape:
            # THE FIX: Call the model with training=False.
            # This uses the frozen BN stats but still allows gradients to flow through other layers.
            logits = self.keras_model(x_input, training=False)
            loss = self.loss_fn(y_batch, logits)

        # Gradients are calculated only on trainable variables (which BN stats are not by default)
        trainable_vars = self.keras_model.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        return {'loss': loss}

    @tf.function
    def infer(self, x_input):
        logits = self.keras_model(x_input, training=False)
        return {'logits': logits}


# --- Data and Evaluation Helpers ---
def load_data(scaler_path, data_path):
    print(f"Loading and preprocessing data from {data_path}...")
    with open(scaler_path, 'r') as f:
        scaler = json.load(f)
        mean = np.array(scaler['mean'], dtype=np.float32)
        scale = np.array(scaler['scale'], dtype=np.float32)

    with open(data_path, 'r') as f:
        client_data = json.load(f)

    # Preprocess all data
    processed_data = []
    for sample in client_data:
        raw_window = np.array(sample['window'], dtype=np.float32)
        # Apply scaling
        scaled_window = (raw_window - mean) / scale
        processed_data.append((scaled_window, sample['label']))

    print(f"Loaded and scaled {len(processed_data)} total samples.")
    return processed_data

def evaluate_model(model, eval_data):
    if not eval_data:
        return 0.0, 0.0

    total_loss = 0
    correct_predictions = 0
    
    for window, label in eval_data:
        # Add a batch dimension to the input window
        input_tensor = tf.expand_dims(tf.constant(window), axis=0)
        
        # Get model output (logits)
        output = model.infer(input_tensor)
        logits = output['logits'][0] # Get the first (and only) item in the batch

        # =====================================================================
        # THE FIX: Convert the integer label into a tensor-like object
        # We also need to add a "batch" dimension to match the logits' shape expectation
        # =====================================================================
        y_true_tensor = tf.constant([label], dtype=tf.int32) # e.g., int 2 becomes tensor [2]
        y_pred_tensor = tf.expand_dims(logits, axis=0)      # e.g., shape (5,) becomes (1, 5)

        # Calculate loss using the tensors
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true_tensor, y_pred_tensor, from_logits=True)
        total_loss += loss.numpy()[0] # The loss is now a 1-element array, so we extract the value

        # Calculate accuracy (this part can remain the same)
        probabilities = tf.nn.softmax(logits).numpy()
        predicted_class = np.argmax(probabilities)
        if predicted_class == label:
            correct_predictions += 1
            
    avg_loss = total_loss / len(eval_data)
    accuracy = correct_predictions / len(eval_data)
    return avg_loss, accuracy

def main():
    # --- 1. Load Model and Data ---
    keras_model_path = ASSETS_DIR / "tcn_v2_ABCDE_har_model.keras"
    scaler_path = ASSETS_DIR / "scaler.json"
    
    model_wrapper = BNTrainableWrapper(keras_model_path)
    all_data = load_data(scaler_path, LABELED_DATA_PATH)

    # --- 2. Simulation Loop ---
    num_rounds = 5
    for i in range(1, num_rounds + 1):
        print(f"\n{'='*20} ROUND {i} {'='*20}")

        # --- 3. Shuffle and split data for this round ---
        random.shuffle(all_data)
        round_data = all_data[:1000] # Take at most 1000 samples
        split_index = int(len(round_data) * 0.8)
        train_data = round_data[:split_index]
        eval_data = round_data[split_index:]

        print(f"Using {len(round_data)} samples: {len(train_data)} for training, {len(eval_data)} for evaluation.")

        # --- 4. Pre-Train Evaluation ---
        pre_loss, pre_acc = evaluate_model(model_wrapper, eval_data)
        print(f"  > Pre-Train Eval -> Loss: {pre_loss:.4f}, Accuracy: {pre_acc*100:.2f}%")

        # --- 5. Local Training ---
        for window, label in train_data:
            input_tensor = tf.expand_dims(tf.constant(window), axis=0)
            label_tensor = tf.constant([label], dtype=tf.int32)
            model_wrapper.train_step(input_tensor, label_tensor)
        print(f"  > Local training on {len(train_data)} samples complete.")
        
        # --- 6. Post-Train Evaluation ---
        post_loss, post_acc = evaluate_model(model_wrapper, eval_data)
        print(f"  > Post-Train Eval -> Loss: {post_loss:.4f}, Accuracy: {post_acc*100:.2f}%")

    print(f"\n{'='*20} SIMULATION COMPLETE {'='*20}")

if __name__ == "__main__":
    main()