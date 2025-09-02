#!/usr/bin/env python3
# simulate_fl_pamap2.py
#
# Simulates Federated Learning (FL) using the FedAvg algorithm for the TCN model
# on the PAMAP2 dataset.
#
# Key Features:
# 1. TCN Model Support: The simulation is built for the single-input TCN model.
# 2. Correct Data Partitioning: Loads all PAMAP2 data, performs a global train/test
#    split, and then partitions the training data by subject to create FL clients.
# 3. Weighted FedAvg: Implements a weighted federated average based on client
#    sample sizes for fair aggregation.
# 4. Centralized Benchmark: Trains the same model on all training data centrally
#    for a direct performance comparison.
# 5. Advanced Evaluation: Tracks the best model, calculates accuracy & F1-score,
#    and generates confusion matrices and performance plots.
#
# To Run:
# 1. Place this script in your '~/tf_project/pamap/' directory.
# 2. Run the centralized training first to have a baseline: python train_pamap2.py
# 3. Run this script: python simulate_fl_pamap2.py
# ==============================================================================

import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Add, Conv1D, SpatialDropout1D, GlobalAveragePooling1D, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

# --- Block 1: Configuration ---
print("--- Block 1: Configuration ---")

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'PAMAP2_Dataset'
OUTPUT_DIR = BASE_DIR / 'results_fl_pamap2_simulation'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Data & Windowing Parameters (from train_pamap2.py) ---
ACTIVITIES_TO_USE = {
    1: 'lying', 2: 'sitting', 3: 'standing', 4: 'walking', 5: 'running',
    6: 'cycling', 7: 'Nordic_walking', 17: 'ironing', 16: 'vacuum_cleaning',
}
WINDOW_SIZE = 100
STRIDE = 50

# --- Federated Learning Parameters ---
COMMUNICATION_ROUNDS = 50
LOCAL_EPOCHS = 3
CLIENTS_PER_ROUND = 5 # Sample 5 out of 9 available subjects each round
BATCH_SIZE = 32

# --- Model Hyperparameters (from train_pamap2.py) ---
KERNEL_SIZE = 7
NUM_FILTERS = 64
NUM_TCN_BLOCKS = 4
DILATION_RATES = [2**i for i in range(NUM_TCN_BLOCKS)]
L2_REG_STRENGTH = 1e-4
SPATIAL_DROPOUT_RATE = 0.15
FINAL_DROPOUT_RATE = 0.35
INITIAL_LEARNING_RATE = 0.001
RANDOM_STATE = 42

# --- GPU Setup ---
print("\n--- GPU Check ---")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ Found {len(gpus)} GPU(s). Enabling memory growth.")
    for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("⚠️ No GPU detected. Training will use CPU.")
print("-" * 25)

# --- Block 2: Data Loading and Partitioning for FL ---
print("\n--- Block 2: Loading and Partitioning PAMAP2 Data for FL Simulation ---")

def load_and_partition_pamap2_data():
    """Loads, windows, scales, and partitions PAMAP2 data for FL simulation."""
    col_names = ['timestamp', 'activity_id', 'hr']
    for loc in ['hand', 'chest', 'ankle']:
        col_names += [f'temp_{loc}'] + [f'acc16g_{ax}_{loc}' for ax in 'xyz'] + \
                     [f'acc6g_{ax}_{loc}' for ax in 'xyz'] + [f'gyro_{ax}_{loc}' for ax in 'xyz'] + \
                     [f'mag_{ax}_{loc}' for ax in 'xyz'] + [f'orient_{i}_{loc}' for i in range(4)]

    all_dfs = []
    for subdir in ['Protocol', 'Optional']:
        for f in (DATA_DIR / subdir).glob('subject*.dat'):
            subject_id = int(f.stem.replace('subject', ''))
            df = pd.read_csv(f, header=None, delim_whitespace=True, names=col_names)
            df['subject_id'] = subject_id
            all_dfs.append(df)
    
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df = full_df[full_df['activity_id'].isin(ACTIVITIES_TO_USE.keys())].copy()
    feature_cols = ['hr'] + [f'acc16g_{ax}_{loc}' for loc in ['hand', 'chest', 'ankle'] for ax in 'xyz'] + \
                   [f'gyro_{ax}_{loc}' for loc in ['hand', 'chest', 'ankle'] for ax in 'xyz'] + \
                   [f'mag_{ax}_{loc}' for loc in ['hand', 'chest', 'ankle'] for ax in 'xyz']
    df_processed = full_df[['subject_id', 'activity_id'] + feature_cols]
    df_processed.interpolate(method='linear', limit_direction='both', inplace=True)

    # Windowing with subject IDs
    windows, labels, subjects = [], [], []
    grouped = df_processed.groupby(['subject_id', 'activity_id'])
    for _, group_df in grouped:
        data_vals = group_df[feature_cols].values
        if len(data_vals) < WINDOW_SIZE: continue
        for start in range(0, len(data_vals) - WINDOW_SIZE + 1, STRIDE):
            windows.append(data_vals[start : start + WINDOW_SIZE])
            labels.append(group_df["activity_id"].iloc[0])
            subjects.append(group_df["subject_id"].iloc[0])
    
    X, y_raw, subjects = np.array(windows), np.array(labels), np.array(subjects)
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    activity_labels = [ACTIVITIES_TO_USE[c] for c in label_encoder.classes_]

    # Global Train/Test Split
    X_train, X_test, y_train, y_test, subjects_train, _ = train_test_split(
        X, y, subjects, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    # Scaling (Fit on train, transform both)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    # Partition scaled training data by subject ID for FL clients
    client_partitions = {}
    client_ids = np.unique(subjects_train)
    for sid in client_ids:
        indices = np.where(subjects_train == sid)[0]
        client_partitions[sid] = (X_train_scaled[indices], y_train[indices])
    
    print(f"Data partitioned into {len(client_ids)} clients.")
    return client_partitions, list(client_ids), X_train_scaled, y_train, X_test_scaled, y_test, activity_labels

# Load all data
client_partitions, CLIENT_IDS, X_train_full, y_train_full, X_test, y_test, ACTIVITY_LABELS = load_and_partition_pamap2_data()

# --- Block 3: TCN Model and FL Algorithm Definition ---
print("\n--- Block 3: Defining TCN Model and FL Logic ---")

# Model definition functions (from train_pamap2.py)
def residual_block(x, dilation_rate, nb_filters, kernel_size, dropout_rate, regularizer):
    prev_x = x
    conv1 = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal', kernel_regularizer=regularizer)(x)
    conv1 = BatchNormalization()(conv1); conv1 = Activation('relu')(conv1); conv1 = SpatialDropout1D(dropout_rate)(conv1)
    conv2 = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal', kernel_regularizer=regularizer)(conv1)
    conv2 = BatchNormalization()(conv2); conv2 = Activation('relu')(conv2); conv2 = SpatialDropout1D(dropout_rate)(conv2)
    if prev_x.shape[-1] != conv2.shape[-1]: prev_x = Conv1D(nb_filters, 1, padding='same', kernel_regularizer=regularizer)(prev_x)
    return Add()([prev_x, conv2])

def build_tcn_model(input_shape, num_classes):
    l2_reg = l2(L2_REG_STRENGTH)
    input_layer = Input(shape=input_shape, name='input_layer')
    x = input_layer
    for rate in DILATION_RATES: x = residual_block(x, rate, NUM_FILTERS, KERNEL_SIZE, SPATIAL_DROPOUT_RATE, l2_reg)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(FINAL_DROPOUT_RATE)(x)
    output_layer = Dense(num_classes, activation='softmax', name='output_dense', kernel_regularizer=l2_reg)(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    optimizer = Adam(learning_rate=INITIAL_LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Weighted Federated Averaging
def weighted_federated_average(client_weights_list, client_sizes):
    total_samples = sum(client_sizes)
    avg_weights = [np.zeros_like(w) for w in client_weights_list[0]]
    for i, client_weights in enumerate(client_weights_list):
        for layer_idx, layer_weights in enumerate(client_weights):
            avg_weights[layer_idx] += (layer_weights * client_sizes[i]) / total_samples
    return avg_weights

# Evaluation and Plotting Helpers
def evaluate_model(model, x_data, y_true):
    y_pred_probs = model.predict(x_data, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    return {"accuracy": acc, "f1_score": f1, "confusion_matrix": cm}

def plot_confusion_matrix(cm, class_names, title, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")

# --- Block 4: Main FL Simulation and Benchmarking ---
if __name__ == "__main__":
    print("\n" + "="*50)
    print("--- Starting PAMAP2 TCN Federated Learning Simulation ---")
    print("="*50)

    input_shape = (WINDOW_SIZE, X_train_full.shape[-1])
    num_classes = len(np.unique(y_train_full))

    # --- 1. Federated Learning Simulation ---
    global_model = build_tcn_model(input_shape, num_classes)
    local_model = build_tcn_model(input_shape, num_classes) # Reusable client model

    fl_accuracies = []
    best_fl_accuracy, best_round = 0.0, 0
    best_fl_weights = None
    start_time = time.time()

    for round_num in range(COMMUNICATION_ROUNDS):
        print(f"\n--- Communication Round {round_num + 1}/{COMMUNICATION_ROUNDS} ---")
        global_weights = global_model.get_weights()
        client_updates, client_sample_sizes = [], []

        selected_client_ids = random.sample(CLIENT_IDS, CLIENTS_PER_ROUND)
        print(f"  Training on {len(selected_client_ids)} selected clients: {selected_client_ids}")

        for client_id in selected_client_ids:
            local_model.set_weights(global_weights)
            X_client, y_client = client_partitions[client_id]
            client_dataset = tf.data.Dataset.from_tensor_slices((X_client, y_client)).shuffle(len(y_client)).batch(BATCH_SIZE)
            local_model.fit(client_dataset, epochs=LOCAL_EPOCHS, verbose=0)
            client_updates.append(local_model.get_weights())
            client_sample_sizes.append(len(y_client))

        new_global_weights = weighted_federated_average(client_updates, client_sample_sizes)
        global_model.set_weights(new_global_weights)

        loss, accuracy = global_model.evaluate(X_test, y_test, verbose=0)
        fl_accuracies.append(accuracy)
        print(f"  Global model test accuracy: {accuracy:.4f}")

        if accuracy > best_fl_accuracy:
            best_fl_accuracy, best_round = accuracy, round_num + 1
            best_fl_weights = global_model.get_weights()
            print(f"  *** New best model found! Acc: {best_fl_accuracy:.4f} at round {best_round} ***")

    fl_duration = time.time() - start_time
    print(f"\nFL Simulation finished in {fl_duration:.2f} seconds.")

    # --- 2. Centralized Training for Benchmark ---
    print("\n" + "="*50)
    print("--- Training Centralized Model for Benchmark ---")
    print("="*50)
    centralized_model = build_tcn_model(input_shape, num_classes)
    centralized_model.fit(X_train_full, y_train_full, epochs=40, batch_size=BATCH_SIZE,
                          validation_data=(X_test, y_test), verbose=1)
    _, centralized_accuracy = centralized_model.evaluate(X_test, y_test, verbose=0)

    # --- 3. Final Report and Plotting ---
    print("\n" + "="*50)
    print("--- Final Simulation Report ---")
    print("="*50)
    print(f"FL Simulation Duration: {fl_duration:.2f} seconds")
    print(f"Federated Model Final Accuracy (Round {COMMUNICATION_ROUNDS}): {fl_accuracies[-1]:.4f}")
    print(f"BEST Federated Model Test Accuracy: {best_fl_accuracy:.4f} (from Round {best_round})")
    print(f"Centralized Model Final Accuracy: {centralized_accuracy:.4f}")

    # --- 4. Detailed Evaluation of the BEST Federated Model ---
    print("\n" + "="*50)
    print(f"--- Detailed Evaluation of Best FL Model (from Round {best_round}) ---")
    print("="*50)

    best_global_model = build_tcn_model(input_shape, num_classes)
    best_global_model.set_weights(best_fl_weights)

    print("\n--- Performance on Global Test Set ---")
    test_metrics = evaluate_model(best_global_model, X_test, y_test)
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}, Weighted F1-Score: {test_metrics['f1_score']:.4f}")
    plot_confusion_matrix(test_metrics['confusion_matrix'], ACTIVITY_LABELS,
                          'Confusion Matrix - Best FL Model on Test Set',
                          OUTPUT_DIR / 'cm_fl_best_model_test_set.png')

    print("\n--- Performance on Combined Client Training Data ---")
    train_metrics = evaluate_model(best_global_model, X_train_full, y_train_full)
    print(f"  Accuracy: {train_metrics['accuracy']:.4f}, Weighted F1-Score: {train_metrics['f1_score']:.4f}")
    plot_confusion_matrix(train_metrics['confusion_matrix'], ACTIVITY_LABELS,
                          'Confusion Matrix - Best FL Model on Client Training Data',
                          OUTPUT_DIR / 'cm_fl_best_model_train_set.png')

    # --- 5. Performance Curve Plot ---
    plt.figure(figsize=(12, 7))
    plt.plot(range(1, COMMUNICATION_ROUNDS + 1), fl_accuracies, marker='o', linestyle='-', label='Federated Model')
    plt.axhline(y=centralized_accuracy, color='r', linestyle='--', label=f'Centralized Benchmark ({centralized_accuracy:.4f})')
    plt.axhline(y=best_fl_accuracy, color='g', linestyle=':', label=f'Best FL Model ({best_fl_accuracy:.4f})')
    plt.title("Federated Learning Performance on PAMAP2 (TCN Model)")
    plt.xlabel("Communication Round")
    plt.ylabel("Global Model Test Accuracy")
    plt.xticks(range(0, COMMUNICATION_ROUNDS + 1, 5))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_path = OUTPUT_DIR / 'fl_pamap2_performance_curve.png'
    plt.savefig(plot_path)
    print(f"\nPerformance plot saved to: {plot_path}")