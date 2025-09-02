# ==============================================================================
# train_model_v5_final_robust.py
#
# Definitive, Production-Ready Training Script for HAR TCN Model
#
# This version provides a robust, version-agnostic solution to the API errors
# encountered previously, ensuring compatibility and correctness.
#
# Key Fixes in this Version:
# 1.  Canonical Label Smoothing: Implemented by switching to the standard
#     `CategoricalCrossentropy` loss function with the `label_smoothing`
#     argument. This is the official and most stable method.
# 2.  One-Hot Encoded Labels: Data labels (y_train, y_val, y_test) are
#     converted to one-hot format to match the requirements of the new loss function.
# 3.  Simplified Final Layer: The model's final activation is reverted to the
#     standard and universal `'softmax'`, removing dependencies on specific or
#     internal Keras/TF function names.
#
# This script retains all previous best practices: Group-Aware Splitting,
# LayerNormalization, L2 Regularization, AdamW, and a correct scaling pipeline.
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
                                     SpatialDropout1D, GlobalAveragePooling1D,
                                     Activation, LayerNormalization)
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.utils import to_categorical # NEW IMPORT

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight

# --- 1. Path and Configuration Setup ---
print("--- [1/7] Setting up configuration and paths ---")
RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# (Paths and Data File configurations are unchanged)
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / ".." / "HAR_Momentum" / "data"
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
INPUT_CSV_1 = DATA_DIR / 'resampled_normalized_phone_data.csv'
INPUT_CSV_2 = DATA_DIR / 'combined_collected_data.csv'
ACTIVITIES_FROM_FILE1 = ['B', 'D', 'E']
ACTIVITIES_FROM_FILE2 = ['A', 'C']

# (Hyperparameters are unchanged)
WINDOW_SIZE = 60
STRIDE = 15
KERNEL_SIZE = 7
DILATION_RATES = [2**i for i in range(5)]
NUM_FILTERS_FL = 64
SPATIAL_DROPOUT_RATE = 0.15
FINAL_DROPOUT_RATE = 0.3
L2_REG_FL = 1e-4
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
LABEL_SMOOTHING = 0.1
EARLY_STOPPING_PATIENCE = 15

# --- 2. GPU Configuration ---
# (Unchanged)
print("--- [2/7] Configuring GPU ---")
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
print("--- [3/7] Loading and windowing data ---")
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
print("--- [4/7] Performing group-aware data splitting and label encoding ---")
label_encoder = LabelEncoder()
y_integers = label_encoder.fit_transform(y_raw)
num_classes = len(label_encoder.classes_)

# (Group splitting logic is unchanged)
gss_test_split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_SEED)
train_val_idx, test_idx = next(gss_test_split.split(X, y_integers, groups=subjects))
X_train_val, y_train_val, subjects_train_val = X[train_val_idx], y_integers[train_val_idx], subjects[train_val_idx]
X_test, y_test_integers, subjects_test = X[test_idx], y_integers[test_idx], subjects[test_idx]

gss_val_split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_SEED)
train_idx, val_idx = next(gss_val_split.split(X_train_val, y_train_val, groups=subjects_train_val))
X_train, y_train_integers = X_train_val[train_idx], y_train_val[train_idx]
X_val, y_val_integers = X_train_val[val_idx], y_train_val[val_idx]

# --- NEW: Convert integer labels to one-hot categorical format ---
y_train = to_categorical(y_train_integers, num_classes=num_classes)
y_val = to_categorical(y_val_integers, num_classes=num_classes)
y_test = to_categorical(y_test_integers, num_classes=num_classes)

# (Verification logic is unchanged)
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
print("--- [5/7] Scaling data based on training set statistics ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X.shape[-1])).reshape(X_train.shape)
X_val_scaled = scaler.transform(X_val.reshape(-1, X.shape[-1])).reshape(X_val.shape)
X_test_scaled = scaler.transform(X_test.reshape(-1, X.shape[-1])).reshape(X_test.shape)
print("Feature scaling complete.")

# --- 6. Robust Model Definition and Training ---
print("--- [6/7] Defining, compiling, and training the robust TCN model ---")
def build_robust_tcn_model(input_shape, num_classes):
    # (Residual block function is unchanged)
    def residual_block(x, dilation_rate):
        prev_x = x
        conv1 = Conv1D(filters=NUM_FILTERS_FL, kernel_size=KERNEL_SIZE, dilation_rate=dilation_rate, padding='same', kernel_regularizer=l2(L2_REG_FL))(x)
        conv1 = LayerNormalization()(conv1); conv1 = Activation('relu')(conv1); conv1 = SpatialDropout1D(SPATIAL_DROPOUT_RATE)(conv1)
        conv2 = Conv1D(filters=NUM_FILTERS_FL, kernel_size=KERNEL_SIZE, dilation_rate=dilation_rate, padding='same', kernel_regularizer=l2(L2_REG_FL))(conv1)
        conv2 = LayerNormalization()(conv2); conv2 = Activation('relu')(conv2); conv2 = SpatialDropout1D(SPATIAL_DROPOUT_RATE)(conv2)
        if prev_x.shape[-1] != conv2.shape[-1]:
            prev_x = Conv1D(NUM_FILTERS_FL, 1, padding='same', kernel_regularizer=l2(L2_REG_FL))(prev_x)
        res_out = Add()([prev_x, conv2]); res_out = Activation('relu')(res_out)
        return res_out
    
    input_layer = Input(shape=input_shape)
    x = input_layer
    for rate in DILATION_RATES:
        x = residual_block(x, rate)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(FINAL_DROPOUT_RATE)(x)
    
    # --- MODIFIED: Reverted to standard, universal softmax activation ---
    output_layer = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=input_layer, outputs=output_layer)

model = build_robust_tcn_model((WINDOW_SIZE, X.shape[-1]), num_classes)

optimizer = AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# --- MODIFIED: Switched to CategoricalCrossentropy with label_smoothing ---
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
model.summary()

# --- Callbacks and Training ---
checkpoint_path = RESULTS_DIR / 'fl_optimized_model.keras'
model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1)
# NOTE: Using integer labels for class weight calculation is correct
class_weights = dict(enumerate(class_weight.compute_class_weight('balanced', classes=np.unique(y_train_integers), y=y_train_integers)))
print(f"Using class weights: {class_weights}")

history = model.fit(
    X_train_scaled, y_train, # Using one-hot encoded y_train
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val_scaled, y_val), # Using one-hot encoded y_val
    callbacks=[model_checkpoint, reduce_lr, early_stopping],
    class_weight=class_weights,
    verbose=2
)

# --- 7. Final Evaluation and Saving Artifacts ---
print("\n--- [7/7] Evaluating final model and saving all artifacts ---")
best_model = tf.keras.models.load_model(checkpoint_path)
# Use one-hot encoded labels for model.evaluate
loss, accuracy = best_model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nFinal Test Set Evaluation:")
print(f"  - Test Loss: {loss:.4f}")
print(f"  - Test Accuracy: {accuracy:.4f}")

# For sklearn metrics, we need integer labels
y_pred_integers = np.argmax(best_model.predict(X_test_scaled), axis=1)

# Save Classification Report as JSON and print
report_dict = classification_report(y_test_integers, y_pred_integers, target_names=label_encoder.classes_, output_dict=True)
with open(RESULTS_DIR / 'classification_report.json', 'w') as f:
    json.dump(report_dict, f, indent=4)
print("\nClassification Report (Test Set):")
print(classification_report(y_test_integers, y_pred_integers, target_names=label_encoder.classes_))

# (Plotting and artifact saving logic is unchanged)
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1); plt.plot(history.history['accuracy'], label='Train Accuracy'); plt.plot(history.history['val_accuracy'], label='Validation Accuracy'); plt.title('Model Accuracy'); plt.legend(); plt.grid(True)
plt.subplot(1, 2, 2); plt.plot(history.history['loss'], label='Train Loss'); plt.plot(history.history['val_loss'], label='Validation Loss'); plt.title('Model Loss'); plt.legend(); plt.grid(True)
plt.tight_layout(); plt.savefig(RESULTS_DIR / 'accuracy_loss_plot.png'); print("Accuracy/Loss plot saved.")

cm = confusion_matrix(y_test_integers, y_pred_integers)
plt.figure(figsize=(10, 8)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix (Test Set)'); plt.ylabel('True Label'); plt.xlabel('Predicted Label'); plt.savefig(RESULTS_DIR / 'confusion_matrix.png'); print("Confusion Matrix plot saved.")

joblib.dump(scaler, RESULTS_DIR / 'scaler.joblib')
joblib.dump(label_encoder, RESULTS_DIR / 'label_encoder.joblib')
print("\nScaler and Label Encoder saved for use in export script.")
print(f"--- Training and evaluation complete! All artifacts saved in '{RESULTS_DIR}' ---")