# FedSim/Strategy_Exports/export_fedprox.py (CORRECTED VERSION)

import shutil
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, SpatialDropout1D, Add, GlobalAveragePooling1D, Dropout, Dense
from tensorflow.keras.models import Model

print(f"Using TensorFlow version: {tf.__version__}")

# --- Configuration ---
RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --- Define Paths relative to this script's location ---
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent # Navigate up to Momentum_Backend/
KERAS_MODEL_PATH = BASE_DIR / "model_source" / "tcn_v2_ABCDE.keras"
OUTPUT_DIR = SCRIPT_DIR.parent / "Models" # Save models to FedSim/Models/

# --- Model Architecture (Consistent with your verification scripts) ---
def build_tcn_model(input_shape, num_classes, include_dropout=True):
    # This function is identical to your working version
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

class FLTrainableWrapper(tf.Module):
    def __init__(self, trained_keras_model):
        super().__init__()
        self.keras_model = build_tcn_model(
            input_shape=trained_keras_model.input_shape[1:],
            num_classes=trained_keras_model.output_shape[1],
            include_dropout=False
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        self.keras_model.set_weights(trained_keras_model.get_weights())
        
        dummy_grads = [tf.zeros_like(var) for var in self.keras_model.trainable_variables]
        self.optimizer.apply_gradients(zip(dummy_grads, self.keras_model.trainable_variables))
        
        self.global_weights_snapshot = [
            tf.Variable(tf.identity(var), trainable=False) for var in self.keras_model.trainable_variables
        ]
        self.all_variables = self.keras_model.variables + self.optimizer.variables

    @tf.function
    def infer(self, x_input):
        return {'logits': self.keras_model(x_input, training=False)}

    @tf.function
    def snapshot_weights(self, dummy_input):
        for var, snapshot_var in zip(self.keras_model.trainable_variables, self.global_weights_snapshot):
            snapshot_var.assign(var)
        return {"status": tf.constant("ok")}

    @tf.function
    def train_step(self, x_input, y_batch, mu):
        with tf.GradientTape() as tape:
            logits = self.keras_model(x_input, training=False)
            standard_loss = self.loss_fn(y_batch, logits)
            
            proximal_term = tf.constant(0.0, dtype=tf.float32)
            for var, global_var_snapshot in zip(self.keras_model.trainable_variables, self.global_weights_snapshot):
                proximal_term += tf.reduce_sum(tf.square(var - global_var_snapshot))
            
            total_loss = standard_loss + (mu / 2.0) * proximal_term

        grads = tape.gradient(total_loss, self.keras_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.keras_model.trainable_variables))
        return {'loss': tf.reshape(total_loss, [1])}

    # --- THE FIX IS HERE ---
    @tf.function
    def get_weights_flat(self, dummy_input):
        """Flattens all variables (model + optimizer) into a single 1D tensor."""
        reshaped_vars = [tf.reshape(v, [-1]) for v in self.all_variables]
        # **CRITICAL FIX**: Cast all variables to float32 to handle the int64 optimizer `iterations` counter.
        casted_vars = [tf.cast(v, tf.float32) for v in reshaped_vars]
        return {"weights": tf.concat(casted_vars, axis=0)}

    @tf.function
    def set_weights_flat(self, flat_weights):
        """Sets all variables (model + optimizer) from a single 1D tensor."""
        start = 0
        for var in self.all_variables:
            size = tf.reduce_prod(var.shape)
            chunk_float = tf.reshape(flat_weights[start:start+size], var.shape)
            # Cast the chunk back to the original variable's dtype before assigning.
            chunk_casted = tf.cast(chunk_float, var.dtype)
            var.assign(chunk_casted)
            start += size
        return {"status": tf.constant("ok")}

    def get_signatures(self):
        # This function defines the inputs/outputs for the TFLite model's functions
        x_spec = tf.TensorSpec(shape=[None, 60, 6], dtype=tf.float32, name='x_input')
        y_spec = tf.TensorSpec(shape=[None], dtype=tf.int32, name='y_batch')
        total_size = sum(tf.reduce_prod(v.shape) for v in self.all_variables)
        flat_weights_spec = tf.TensorSpec(shape=[total_size], dtype=tf.float32, name="flat_weights")
        dummy_input_spec = tf.TensorSpec(shape=[], dtype=tf.float32, name="dummy_input")
        mu_spec = tf.TensorSpec(shape=[], dtype=tf.float32, name="mu")

        return {
            'infer': self.infer.get_concrete_function(x_spec),
            'train_step': self.train_step.get_concrete_function(x_spec, y_spec, mu_spec),
            'snapshot_weights': self.snapshot_weights.get_concrete_function(dummy_input_spec),
            'get_weights_flat': self.get_weights_flat.get_concrete_function(dummy_input_spec),
            'set_weights_flat': self.set_weights_flat.get_concrete_function(flat_weights_spec),
        }

def main():
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)
    
    print(f"1. Loading base Keras model from {KERAS_MODEL_PATH}")
    original_trained_model = tf.keras.models.load_model(str(KERAS_MODEL_PATH), compile=False)

    print("2. Wrapping model in FedProx-enabled TF.Module...")
    wrapper = FLTrainableWrapper(original_trained_model)
    signatures = wrapper.get_signatures()

    saved_model_dir = OUTPUT_DIR / 'saved_model_fedprox'
    if saved_model_dir.exists():
        shutil.rmtree(saved_model_dir)
    print(f"3. Saving SavedModel to: {saved_model_dir}")
    tf.saved_model.save(wrapper, str(saved_model_dir), signatures=signatures)
    
    print("4. Saving initial weights for the simulator...")
    flat_weights_array = wrapper.get_weights_flat(tf.constant(0.0))['weights'].numpy()
    initial_weights_path = OUTPUT_DIR / "initial_weights_fedprox.bin"
    initial_weights_path.write_bytes(flat_weights_array.tobytes())
    print(f"   > Saved initial weights to: {initial_weights_path}")
    
    print(f"5. Converting to FedProx-trainable TFLite model...")
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.experimental_enable_resource_variables = True
    tflite_model = converter.convert()
    
    tflite_path = OUTPUT_DIR / "fedprox_trainable.tflite"
    tflite_path.write_bytes(tflite_model)
    print(f"   > Wrote TFLite model to: {tflite_path}")
    print(f"\n✅ FedProx export complete. Assets generated in '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()