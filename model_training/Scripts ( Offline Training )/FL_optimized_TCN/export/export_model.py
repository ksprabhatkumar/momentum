# ~/tf_project/FL_optimized_TCN/export/export_model.py

import shutil
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Add, Conv1D, SpatialDropout1D, GlobalAveragePooling1D, Activation, LayerNormalization
from tensorflow.keras.regularizers import l2
import json
import joblib

# --- 1. Path and Configuration Setup ---
print("--- [1/3] Setting up configuration and paths ---")
RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

SCRIPT_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = SCRIPT_DIR / ".." / "results"
OUTPUT_DIR = SCRIPT_DIR / "generated_assets"

KERAS_MODEL_PATH = RESULTS_DIR / "fl_optimized_model.keras"
SCALER_PATH = RESULTS_DIR / "scaler.joblib"
LABEL_ENCODER_PATH = RESULTS_DIR / "label_encoder.joblib"

LEARNING_RATE_FL = 1e-4
NUM_FILTERS_FL = 32
KERNEL_SIZE = 7
DILATION_RATES = [2**i for i in range(5)]
L2_REG_FL = 1e-4

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True)

# --- 2. Build and Verify the FLTrainableWrapper ---
print("--- [2/3] Building and wrapping model for TFLite export ---")
def build_tcn_model_fl_optimized_for_export(input_shape, num_classes):
    def residual_block_fl(x, dilation_rate):
        prev_x = x
        conv1 = Conv1D(filters=NUM_FILTERS_FL, kernel_size=KERNEL_SIZE, dilation_rate=dilation_rate, padding='same', kernel_regularizer=l2(L2_REG_FL))(x)
        conv1 = LayerNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv2 = Conv1D(filters=NUM_FILTERS_FL, kernel_size=KERNEL_SIZE, dilation_rate=dilation_rate, padding='same', kernel_regularizer=l2(L2_REG_FL))(conv1)
        conv2 = LayerNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        if prev_x.shape[-1] != conv2.shape[-1]:
            prev_x = Conv1D(NUM_FILTERS_FL, 1, padding='same')(prev_x)
        return Add()([prev_x, conv2])

    input_layer = Input(shape=input_shape)
    x = input_layer
    for rate in DILATION_RATES:
        x = residual_block_fl(x, rate)
    x = GlobalAveragePooling1D()(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=input_layer, outputs=output_layer)


class FLTrainableWrapper(tf.Module):
    def __init__(self, trained_keras_model):
        super().__init__()
        self.keras_model = build_tcn_model_fl_optimized_for_export(
            input_shape=trained_keras_model.input_shape[1:],
            num_classes=trained_keras_model.output_shape[1]
        )
        self.keras_model.set_weights(trained_keras_model.get_weights())
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_FL)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        trainable_vars = self.keras_model.trainable_variables
        dummy_grads = [tf.zeros_like(var) for var in trainable_vars]
        self.optimizer.apply_gradients(zip(dummy_grads, trainable_vars))
        self.all_variables = self.keras_model.variables + self.optimizer.variables

    @tf.function
    def infer(self, x_input):
        return {'logits': self.keras_model(x_input, training=False)}

    @tf.function
    def train_step(self, x_input, y_batch):
        with tf.GradientTape() as tape:
            logits = self.keras_model(x_input, training=False)
            loss = self.loss_fn(y_batch, logits)
        trainable_vars = self.keras_model.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        return {'loss': tf.reshape(loss, [1])}

    @tf.function
    def get_weights_flat(self, dummy_input):
        reshaped_vars = [tf.reshape(v, [-1]) for v in self.all_variables]
        casted_vars = [tf.cast(v, tf.float32) for v in reshaped_vars]
        return {"weights": tf.concat(casted_vars, axis=0)}

    @tf.function
    def set_weights_flat(self, flat_weights):
        start = 0
        for var in self.all_variables:
            size = tf.reduce_prod(var.shape)
            chunk_float32 = tf.reshape(flat_weights[start:start+size], var.shape)
            chunk_casted = tf.cast(chunk_float32, var.dtype)
            var.assign(chunk_casted)
            start += size
        return {"status": tf.constant("ok", dtype=tf.string)}

    # --- THE FIX: Re-introduce the get_signatures method ---
    def get_signatures(self):
        # Dynamically determine the total size of all variables for the spec
        total_weights_size = sum(tf.reduce_prod(v.shape) for v in self.all_variables)

        x_spec = tf.TensorSpec(shape=[None, 60, 6], dtype=tf.float32, name='x_input')
        y_spec = tf.TensorSpec(shape=[None], dtype=tf.int32, name='y_batch')
        dummy_input_spec = tf.TensorSpec(shape=(), dtype=tf.float32, name='dummy_input')
        flat_weights_spec = tf.TensorSpec(shape=[total_weights_size], dtype=tf.float32, name='flat_weights')

        return {
            'infer': self.infer.get_concrete_function(x_spec),
            'train_step': self.train_step.get_concrete_function(x_spec, y_spec),
            'get_weights_flat': self.get_weights_flat.get_concrete_function(dummy_input_spec),
            'set_weights_flat': self.set_weights_flat.get_concrete_function(flat_weights_spec),
        }

# --- 3. Run Export and Save All Assets ---
print("--- [3/3] Exporting final assets ---")
original_model = load_model(KERAS_MODEL_PATH)
wrapper = FLTrainableWrapper(original_model)

# --- THE FIX: Use the get_signatures method before saving ---
signatures = wrapper.get_signatures()
saved_model_dir = OUTPUT_DIR / 'saved_model'
tf.saved_model.save(wrapper, str(saved_model_dir), signatures=signatures)
print(f"SavedModel created at: {saved_model_dir}")

# Save initial_weights.bin for the server
flat_weights = wrapper.get_weights_flat(tf.constant(0.0))['weights'].numpy()
(OUTPUT_DIR / "initial_weights.bin").write_bytes(flat_weights.tobytes())
print(f"Initial weights saved. Total flat weights: {len(flat_weights)}")

# Convert and save the trainable TFLite model for the app
converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter.experimental_enable_resource_variables = True
tflite_model = converter.convert()
(OUTPUT_DIR / "tcn_fl_optimized_trainable.tflite").write_bytes(tflite_model)
print("Trainable TFLite model saved.")

# Save the scaler.json for the app
scaler = joblib.load(SCALER_PATH)
scaler_dict = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}
with open(OUTPUT_DIR / 'scaler.json', 'w') as f:
    json.dump(scaler_dict, f, indent=4)
print("Scaler JSON saved.")

# Save the labels.json for the app
label_encoder = joblib.load(LABEL_ENCODER_PATH)
labels_dict = {str(i): label for i, label in enumerate(label_encoder.classes_)}
with open(OUTPUT_DIR / 'labels.json', 'w') as f:
    json.dump(labels_dict, f, indent=4)
print("Labels JSON saved.")

print(f"\n✅ Export complete. All assets are in: {OUTPUT_DIR}")