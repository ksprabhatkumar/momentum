# ==============================================================================
# train_wisnet_model.py
#
# Implements the WISNet architecture for Human Activity Recognition,
# integrating it into a proven, production-ready data processing and
# training pipeline.
#
# Key Architectural Components Implemented:
# 1.  CNPM Block: The initial Convolved, Normalized, and Pooled block for
#     primary feature extraction.
# 2.  IDBN Block: The Identity and Basic block, a residual-style component with
#     parallel convolutions to capture features at multiple scales.
# 3.  CAS Block: The Channel and Spatial Attention block, inspired by CBAM,
#     to help the network focus on the most salient features.
#
# The data handling, splitting, scaling, and evaluation logic is preserved
# from the reference script to ensure a robust and fair training process.
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import joblib

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, Add, Conv1D,
                                     MaxPooling1D, GlobalAveragePooling1D,
                                     Activation, LayerNormalization, BatchNormalization,
                                     Multiply, Concatenate, Reshape, Permute, Lambda)
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight
from keras import ops

tf.keras.config.enable_unsafe_deserialization()

# --- 1. Path and Configuration Setup ---
print("--- [1/8] Setting up configuration and paths ---")
RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# (Paths and Data File configurations are unchanged)
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / ".." / "HAR_Momentum" / "data"
RESULTS_DIR = SCRIPT_DIR / "results_wisnet" # Saving results to a new directory
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
INPUT_CSV_1 = DATA_DIR / 'resampled_normalized_phone_data.csv'
INPUT_CSV_2 = DATA_DIR / 'combined_collected_data.csv'
ACTIVITIES_FROM_FILE1 = ['B', 'D', 'E']
ACTIVITIES_FROM_FILE2 = ['A', 'C']

# (Hyperparameters are mostly unchanged, adapted for WISNet where necessary)
WINDOW_SIZE = 60
STRIDE = 15
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
LABEL_SMOOTHING = 0.1
EARLY_STOPPING_PATIENCE = 15

# --- 2. GPU Configuration ---
# (Unchanged)
print("--- [2/8] Configuring GPU ---")
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print(f"Found {len(gpu_devices)} GPU(s). Enabling memory growth.")
    try:
        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e: print(f"Could not set memory growth: {e}")
else: print("!!! No GPU found. Training will use CPU. !!!")

# --- 3. Data Loading and Windowing ---
# (Unchanged)
print("--- [3/8] Loading and windowing data ---")
def create_windows(df, window_size, stride):
    windows, labels, subjects = [], [], []
    required_cols = ['x_accel', 'y_accel', 'z_accel', 'x_gyro', 'y_gyro', 'z_gyro']
    for (subject, activity), group_df in df.groupby(['subject', 'activity']):
        data_values = group_df.sort_values('timestamp')[required_cols].values
        if len(data_values) < window_size: continue
        for start in range(0, len(data_values) - window_size + 1, stride):
            windows.append(data_values[start : start + window_size])
            labels.append(activity)
            subjects.append(subject)
    return np.array(windows), np.array(labels), np.array(subjects)

df1 = pd.read_csv(INPUT_CSV_1)[lambda x: x['activity'].isin(ACTIVITIES_FROM_FILE1)]
df2 = pd.read_csv(INPUT_CSV_2)[lambda x: x['activity'].isin(ACTIVITIES_FROM_FILE2)]
combined_df = pd.concat([df1, df2], ignore_index=True)
X, y_raw, subjects = create_windows(combined_df, WINDOW_SIZE, STRIDE)

# --- 4. Group-Aware Data Splitting and Label Encoding ---
# (Unchanged)
print("--- [4/8] Performing group-aware data splitting and label encoding ---")
label_encoder = LabelEncoder()
y_integers = label_encoder.fit_transform(y_raw)
num_classes = len(label_encoder.classes_)

gss_test_split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_SEED)
train_val_idx, test_idx = next(gss_test_split.split(X, y_integers, groups=subjects))
X_train_val, y_train_val, subjects_train_val = X[train_val_idx], y_integers[train_val_idx], subjects[train_val_idx]
X_test, y_test_integers, subjects_test = X[test_idx], y_integers[test_idx], subjects[test_idx]

gss_val_split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_SEED)
train_idx, val_idx = next(gss_val_split.split(X_train_val, y_train_val, groups=subjects_train_val))
X_train, y_train_integers = X_train_val[train_idx], y_train_val[train_idx]
X_val, y_val_integers = X_train_val[val_idx], y_train_val[val_idx]

y_train = to_categorical(y_train_integers, num_classes=num_classes)
y_val = to_categorical(y_val_integers, num_classes=num_classes)
y_test = to_categorical(y_test_integers, num_classes=num_classes)

train_subjects = set(np.unique(subjects_train_val[train_idx]))
val_subjects = set(np.unique(subjects_train_val[val_idx]))
test_subjects = set(np.unique(subjects_test))
assert train_subjects.isdisjoint(val_subjects) and train_subjects.isdisjoint(test_subjects) and val_subjects.isdisjoint(test_subjects)
print("Data splitting successful and verified. No subject overlap.")
print(f"  - Training set:   {len(X_train)} windows from {len(train_subjects)} subjects. Label shape: {y_train.shape}")
print(f"  - Validation set: {len(X_val)} windows from {len(val_subjects)} subjects. Label shape: {y_val.shape}")
print(f"  - Test set:       {len(X_test)} windows from {len(test_subjects)} subjects. Label shape: {y_test.shape}")

# --- 5. Data Scaling (Post-Split) ---
# (Unchanged)
print("--- [5/8] Scaling data based on training set statistics ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X.shape[-1])).reshape(X_train.shape)
X_val_scaled = scaler.transform(X_val.reshape(-1, X.shape[-1])).reshape(X_val.shape)
X_test_scaled = scaler.transform(X_test.reshape(-1, X.shape[-1])).reshape(X_test.shape)
print("Feature scaling complete.")

# --- 6. WISNet Model Definition ---
print("--- [6/8] Defining the WISNet model architecture ---")

def cnpm_block(x, filters=64, kernel_size=5):
    """Convolved Normalized Pooled (CNPM) block."""
    x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)
    return x

def idbn_block_simplified(x, filters=32):
    """
    A simplified Identity and Basic (IDBN) block to reduce model complexity
    and combat overfitting. This is a more standard residual block.
    """
    # Identity part (skip connection)
    shortcut = x

    # Basic Block part with sequential convolutions
    x = Conv1D(filters=filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(filters=filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)

    # Add the shortcut to the output of the convolutions
    res_out = Add()([shortcut, x])
    res_out = Activation('relu')(res_out)
    return res_out

def cas_block(x, ratio=8):
    """
    Channel and Spatial Attention (CAS) block.
    CORRECTED to use the Keras 3 `ops` backend for robust serialization.
    """
    channel_axis = -1
    num_channels = x.shape[channel_axis]

    # Channel Attention Module (unchanged)
    shared_layer_one = Dense(num_channels // ratio, activation='relu', use_bias=True)
    shared_layer_two = Dense(num_channels, use_bias=True)

    avg_pool = GlobalAveragePooling1D()(x)
    avg_pool = Reshape((1, num_channels))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = GlobalAveragePooling1D()(x)
    max_pool = Reshape((1, num_channels))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    channel_attention = Add()([avg_pool, max_pool])
    channel_attention = Activation('sigmoid')(channel_attention)
    channel_attention = Reshape((1, num_channels))(channel_attention)

    channel_refined = Multiply()([x, channel_attention])

    # Spatial Attention Module (CORRECTED to use ops.mean and ops.max)
    spatial_output_shape = (channel_refined.shape[1], 1)

    avg_pool_spatial = Lambda(lambda y: ops.mean(y, axis=channel_axis, keepdims=True),
                              output_shape=spatial_output_shape)(channel_refined)
    max_pool_spatial = Lambda(lambda y: ops.max(y, axis=channel_axis, keepdims=True),
                              output_shape=spatial_output_shape)(channel_refined)
    
    concat_spatial = Concatenate(axis=channel_axis)([avg_pool_spatial, max_pool_spatial])
    spatial_attention = Conv1D(1, kernel_size=7, padding='same', activation='sigmoid')(concat_spatial)
    
    attention_refined = Multiply()([channel_refined, spatial_attention])
    return attention_refined


def build_wisnet_model(input_shape, num_classes):
    """
    Builds the complete WISNet model using the SIMPLIFIED IDBN block.
    """
    input_layer = Input(shape=input_shape)

    # Phase 1: Initial feature extraction
    cnpm_output = cnpm_block(input_layer, filters=32, kernel_size=5)

    # Phase 2: Deeper feature extraction and refinement
    idbn1_output = idbn_block_simplified(cnpm_output, filters=32)
    cas_output = cas_block(idbn1_output)

    # Residual connection
    sum_output = Add()([cnpm_output, cas_output])
    
    # Phase 3: Final feature extraction
    idbn2_output = idbn_block_simplified(sum_output, filters=32)

    # Final Classifier Head
    x = GlobalAveragePooling1D()(idbn2_output)
    x = Dropout(0.5)(x)
    output_layer = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# --- 7. Model Compilation and Training ---
print("--- [7/8] Compiling and training the WISNet model ---")
model = build_wisnet_model((WINDOW_SIZE, X.shape[-1]), num_classes)
optimizer = AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
model.summary()

# Callbacks and Training
checkpoint_path = RESULTS_DIR / 'wisnet_model.keras'
model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1)
class_weights = dict(enumerate(class_weight.compute_class_weight('balanced', classes=np.unique(y_train_integers), y=y_train_integers)))
print(f"Using class weights: {class_weights}")

history = model.fit(
    X_train_scaled, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val_scaled, y_val),
    callbacks=[model_checkpoint, reduce_lr, early_stopping],
    class_weight=class_weights,
    verbose=2
)

# --- 8. Final Evaluation and Saving Artifacts ---
# (Unchanged)
print("\n--- [8/8] Evaluating final model and saving all artifacts ---")
best_model = tf.keras.models.load_model(checkpoint_path)
loss, accuracy = best_model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nFinal Test Set Evaluation:")
print(f"  - Test Loss: {loss:.4f}")
print(f"  - Test Accuracy: {accuracy:.4f}")

y_pred_integers = np.argmax(best_model.predict(X_test_scaled), axis=1)

report_dict = classification_report(y_test_integers, y_pred_integers, target_names=label_encoder.classes_, output_dict=True)
with open(RESULTS_DIR / 'classification_report.json', 'w') as f:
    json.dump(report_dict, f, indent=4)
print("\nClassification Report (Test Set):")
print(classification_report(y_test_integers, y_pred_integers, target_names=label_encoder.classes_))

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1); plt.plot(history.history['accuracy'], label='Train Accuracy'); plt.plot(history.history['val_accuracy'], label='Validation Accuracy'); plt.title('Model Accuracy'); plt.legend(); plt.grid(True)
plt.subplot(1, 2, 2); plt.plot(history.history['loss'], label='Train Loss'); plt.plot(history.history['val_loss'], label='Validation Loss'); plt.title('Model Loss'); plt.legend(); plt.grid(True)
plt.tight_layout(); plt.savefig(RESULTS_DIR / 'accuracy_loss_plot.png'); print("Accuracy/Loss plot saved.")

cm = confusion_matrix(y_test_integers, y_pred_integers)
plt.figure(figsize=(10, 8)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix (Test Set)'); plt.ylabel('True Label'); plt.xlabel('Predicted Label'); plt.savefig(RESULTS_DIR / 'confusion_matrix.png'); print("Confusion Matrix plot saved.")

joblib.dump(scaler, RESULTS_DIR / 'scaler.joblib')
joblib.dump(label_encoder, RESULTS_DIR / 'label_encoder.joblib')
print(f"\nScaler and Label Encoder saved. \n--- Training and evaluation complete! All artifacts saved in '{RESULTS_DIR}' ---")