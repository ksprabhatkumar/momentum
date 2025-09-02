#!/usr/bin/env python3
# train_pamap2.py
#
# Trains a regularized Temporal Convolutional Network (TCN) on the PAMAP2 dataset.
# This script loads data from the 'Protocol' and 'Optional' folders, preprocesses it,
# windows the time-series, and trains the model. It incorporates best practices
# like L2 regularization and robust callbacks inspired by the train_UCI_HAR.py script.
#
# To Run:
# 1. Place this script in your '~/tf_project/pamap/' directory.
# 2. Ensure the PAMAP2_Dataset is unzipped in the same directory.
# 3. Activate your virtual environment: source ~/tf_project/tf_venv/bin/activate
# 4. Run the script: python train_pamap2.py
# ==============================================================================

import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Add, Conv1D, SpatialDropout1D, GlobalAveragePooling1D, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# --- Block 1: Configuration ---
print("--- Block 1: Configuration ---")

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'PAMAP2_Dataset'
OUTPUT_DIR = BASE_DIR / 'results_pamap2'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Data & Windowing Parameters ---
# Activities to use (ID -> Name). We exclude transient (0) and short activities for robust training.
ACTIVITIES_TO_USE = {
    1: 'lying',
    2: 'sitting',
    3: 'standing',
    4: 'walking',
    5: 'running',
    6: 'cycling',
    7: 'Nordic_walking',
    17: 'ironing',
    16: 'vacuum_cleaning',
}
WINDOW_SIZE = 100  # 1 second of data at 100Hz
STRIDE = 50        # 50% overlap

# --- Model Hyperparameters (inspired by training.py & train_UCI_HAR.py) ---
KERNEL_SIZE = 7
NUM_FILTERS = 64
NUM_TCN_BLOCKS = 4
DILATION_RATES = [2**i for i in range(NUM_TCN_BLOCKS)]
L2_REG_STRENGTH = 1e-4
SPATIAL_DROPOUT_RATE = 0.15
FINAL_DROPOUT_RATE = 0.35

# --- Training Parameters ---
BATCH_SIZE = 64
EPOCHS = 100
VALIDATION_SPLIT = 0.2
INITIAL_LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 15
RANDOM_STATE = 42

# --- GPU Setup ---
print("\n--- GPU Check ---")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ Found {len(gpus)} GPU(s). Enabling memory growth.")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Could not set memory growth: {e}")
else:
    print("⚠️ No GPU detected. Training will use CPU.")
print("-" * 25)

# --- Block 2: Data Loading and Preprocessing ---
print("\n--- Block 2: Loading and Preprocessing PAMAP2 Data ---")

def load_pamap2_data(data_dir: Path):
    """Loads, cleans, and preprocesses the raw PAMAP2 data files."""
    # Define column names based on the readme file
    # We only name the columns we will use for simplicity.
    col_names = ['timestamp', 'activity_id', 'hr']
    for sensor_loc in ['hand', 'chest', 'ankle']:
        col_names += [f'temp_{sensor_loc}']
        col_names += [f'acc16g_{axis}_{sensor_loc}' for axis in 'xyz']
        col_names += [f'acc6g_{axis}_{sensor_loc}' for axis in 'xyz']
        col_names += [f'gyro_{axis}_{sensor_loc}' for axis in 'xyz']
        col_names += [f'mag_{axis}_{sensor_loc}' for axis in 'xyz']
        col_names += [f'orient_{i}_{sensor_loc}' for i in range(4)] # Invalid data, will be dropped

    all_dfs = []
    # Iterate over protocol and optional files
    for subdir in ['Protocol', 'Optional']:
        subdir_path = data_dir / subdir
        for f in subdir_path.glob('subject*.dat'):
            subject_id = int(f.stem.replace('subject', ''))
            print(f"Reading {f.name} (Subject {subject_id})...")
            df = pd.read_csv(f, header=None, delim_whitespace=True, names=col_names)
            df['subject_id'] = subject_id
            all_dfs.append(df)

    if not all_dfs:
        raise FileNotFoundError(f"No .dat files found in {data_dir}/Protocol or {data_dir}/Optional. Check paths.")

    full_df = pd.concat(all_dfs, ignore_index=True)

    # --- Data Cleaning ---
    # 1. Filter for activities we want to use
    full_df = full_df[full_df['activity_id'].isin(ACTIVITIES_TO_USE.keys())].copy()
    full_df['activity_name'] = full_df['activity_id'].map(ACTIVITIES_TO_USE)

    # 2. Select only the feature columns we need
    # We use the 16g accelerometer as recommended, plus gyro and magnetometer.
    feature_cols = ['hr']
    for sensor_loc in ['hand', 'chest', 'ankle']:
        feature_cols += [f'acc16g_{axis}_{sensor_loc}' for axis in 'xyz']
        feature_cols += [f'gyro_{axis}_{sensor_loc}' for axis in 'xyz']
        feature_cols += [f'mag_{axis}_{sensor_loc}' for axis in 'xyz']

    final_cols = ['subject_id', 'activity_id', 'activity_name'] + feature_cols
    df_processed = full_df[final_cols]

    # 3. Handle missing values, especially in Heart Rate
    # Forward fill then backward fill is a robust way to handle this
    df_processed.interpolate(method='linear', limit_direction='both', inplace=True)

    # Sanity check for remaining NaNs
    if df_processed.isnull().values.any():
        print("Warning: NaNs still present after interpolation. Filling with 0.")
        df_processed.fillna(0, inplace=True)

    print(f"Data loading complete. Shape: {df_processed.shape}")
    print(f"Activity distribution:\n{df_processed['activity_name'].value_counts()}")
    return df_processed, feature_cols


def create_windows(df, feature_cols, window_size, stride):
    """Creates overlapping windows of time-series data."""
    windows, labels = [], []
    grouped = df.groupby(['subject_id', 'activity_id'])
    print(f"Processing {len(grouped)} subject-activity groups for windowing...")

    for _, group_df in grouped:
        data_values = group_df[feature_cols].values
        activity_id = group_df["activity_id"].iloc[0]

        if len(data_values) < window_size:
            continue

        for start in range(0, len(data_values) - window_size + 1, stride):
            windows.append(data_values[start : start + window_size])
            labels.append(activity_id)

    if not windows:
        raise ValueError("No windows created. Check window_size/stride or dataset lengths.")

    return np.array(windows), np.array(labels)


# --- Block 3: TCN Model Definition (Regularized) ---
print("\n--- Block 3: Defining the Regularized TCN Model ---")

def residual_block(x, dilation_rate, nb_filters, kernel_size, dropout_rate, regularizer):
    prev_x = x
    conv1 = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal', kernel_regularizer=regularizer)(x)
    conv1 = BatchNormalization()(conv1); conv1 = Activation('relu')(conv1); conv1 = SpatialDropout1D(dropout_rate)(conv1)

    conv2 = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal', kernel_regularizer=regularizer)(conv1)
    conv2 = BatchNormalization()(conv2); conv2 = Activation('relu')(conv2); conv2 = SpatialDropout1D(dropout_rate)(conv2)

    if prev_x.shape[-1] != conv2.shape[-1]:
        prev_x = Conv1D(nb_filters, 1, padding='same', kernel_regularizer=regularizer)(prev_x)
    return Add()([prev_x, conv2])

def build_tcn_model(input_shape, num_classes):
    l2_reg = l2(L2_REG_STRENGTH)
    input_layer = Input(shape=input_shape, name='input_layer')
    x = input_layer

    for rate in DILATION_RATES:
        x = residual_block(x, rate, NUM_FILTERS, KERNEL_SIZE, SPATIAL_DROPOUT_RATE, l2_reg)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(FINAL_DROPOUT_RATE)(x)
    output_layer = Dense(num_classes, activation='softmax', name='output_dense', kernel_regularizer=l2_reg)(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    optimizer = Adam(learning_rate=INITIAL_LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Block 4: Plotting Utilities ---
def plot_training_history(history, file_path):
    """Plots and saves the training and validation loss and accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True)
    # Loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    print(f"Training history plot saved to {file_path}")

def plot_confusion_matrix(y_true, y_pred_classes, class_names, file_path):
    """Plots and saves a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    print(f"Confusion matrix saved to {file_path}")


# --- Block 5: Main Training and Evaluation ---
def run_training():
    print("\n" + "="*50)
    print("--- Running PAMAP2 TCN Model Training and Evaluation ---")
    print("="*50)

    # --- Load and prepare data ---
    df, feature_cols = load_pamap2_data(DATA_DIR)
    X, y_raw = create_windows(df, feature_cols, WINDOW_SIZE, STRIDE)
    print(f"Created {X.shape[0]} windows.")

    # --- Label Encoding ---
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    num_classes = len(label_encoder.classes_)
    class_names = [ACTIVITIES_TO_USE[c] for c in label_encoder.classes_]
    print(f"Encoded {num_classes} classes: {class_names}")

    # --- Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    # --- Scaling (Crucial Step!) ---
    print("Fitting StandardScaler on training data...")
    num_samples_train, timesteps, num_features = X_train.shape
    scaler = StandardScaler()
    # Reshape to (samples * timesteps, features) for scaler
    X_train_reshaped = X_train.reshape(-1, num_features)
    X_train_scaled_reshaped = scaler.fit_transform(X_train_reshaped)
    X_train_scaled = X_train_scaled_reshaped.reshape(num_samples_train, timesteps, num_features)

    # Apply the same scaler to the test data
    X_test_reshaped = X_test.reshape(-1, num_features)
    X_test_scaled_reshaped = scaler.transform(X_test_reshaped)
    X_test_scaled = X_test_scaled_reshaped.reshape(X_test.shape[0], timesteps, num_features)
    print("Feature scaling complete.")

    # --- Build the Model ---
    model = build_tcn_model(input_shape=(WINDOW_SIZE, num_features), num_classes=num_classes)
    model.summary()

    # --- Callbacks ---
    checkpoint_path = OUTPUT_DIR / 'best_pamap2_tcn_model.keras'
    model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1)

    # --- Train the Model ---
    print("\nStarting model training...")
    start_time = time.time()
    history = model.fit(
        X_train_scaled, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[model_checkpoint, reduce_lr, early_stopping],
        verbose=1
    )
    training_time = time.time() - start_time
    print(f"Training finished in {training_time / 60:.2f} minutes.")

    # --- Generate and Save Training Plots ---
    plot_training_history(history, OUTPUT_DIR / "tcn_training_history.png")

    # --- Evaluate the Best Model ---
    print("\nLoading best model and evaluating on the test set...")
    best_model = tf.keras.models.load_model(checkpoint_path)
    
    loss, accuracy = best_model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Loss: {loss:.4f}")

    y_pred_probs = best_model.predict(X_test_scaled)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    # --- Save Classification Report and Confusion Matrix ---
    report_str = classification_report(y_test, y_pred_classes, target_names=class_names)
    print("\nClassification Report:")
    print(report_str)
    with open(OUTPUT_DIR / "classification_report.txt", "w") as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        f.write(report_str)

    plot_confusion_matrix(y_test, y_pred_classes, class_names, OUTPUT_DIR / "confusion_matrix.png")


# --- Main Execution Block ---
if __name__ == "__main__":
    run_training()
    print("\n" + "="*50)
    print("--- SCRIPT COMPLETE ---")
    print(f"All models and plots saved in: {OUTPUT_DIR}")
    print("="*50)