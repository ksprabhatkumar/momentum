# ==============================================================================
# simulate_fl_hybrid.py
#
# Simulates Federated Learning (FL) for the improved hybrid model using FedAvg.
#
# Key Features & Improvements:
# 1. Hybrid Model Support: The simulation is built for a model with two inputs
#    (time-series for TCN, static features for MLP).
# 2. Weighted FedAvg: Implements a true weighted federated average based on
#    the number of samples per client for more stable and fair aggregation.
# 3. Optimized Simulation: Utilizes client sampling and a reusable client
#    model instance for speed, similar to the provided template.
# 4. Encapsulated Data Pipeline: A single function handles loading, scaling,
#    and partitioning both data types, keeping the main script clean.
# 5. Centralized Benchmark: Trains the same model on all data centrally for a
#    fair performance comparison.
# 6. Advanced Evaluation: Tracks the best model, calculates accuracy, F1-score,
#    and generates confusion matrices for both test and client training data.
# ==============================================================================

# --- Block 1: Setup and Initial Configuration ---
print("--- Block 1: Setup and Initial Configuration ---")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import random
import time
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Add, Concatenate
from tensorflow.keras.layers import Conv1D, SpatialDropout1D, GlobalAveragePooling1D, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# --- Main Configuration ---
DATASET_PATH = Path('./data/UCI HAR Dataset/')
OUTPUT_DIR = Path('./results_fl_hybrid_simulation/')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Federated Learning Parameters ---
COMMUNICATION_ROUNDS = 60
LOCAL_EPOCHS = 4
# The original dataset has 21 subjects in the train set. We'll sample from them.
CLIENTS_PER_ROUND = 15
BATCH_SIZE = 32

# --- Hybrid Model Hyperparameters (from previous script) ---
WINDOW_SIZE = 128
INITIAL_LEARNING_RATE = 0.001
L2_REG_STRENGTH = 1e-4
SPATIAL_DROPOUT_RATE = 0.15
DENSE_DROPOUT_RATE = 0.45
KERNEL_SIZE = 7
NUM_FILTERS = 64
NUM_TCN_BLOCKS = 4
DILATION_RATES = [2**i for i in range(NUM_TCN_BLOCKS)]

# --- GPU Check ---
print("\n--- GPU Check ---")
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print(f"Found {len(gpu_devices)} GPU(s).")
    for gpu in gpu_devices: tf.config.experimental.set_memory_growth(gpu, True)
else: print("!!! No GPU found. Training will use CPU. !!!")
print("-" * 25)


# ==============================================================================
# --- Block 2: Data Loading and Partitioning for FL ---
# ==============================================================================
print("\n--- Block 2: Loading and Partitioning Data for FL Simulation ---")

def load_and_partition_hybrid_data():
    """Loads, scales, and partitions all data needed for the hybrid model FL simulation."""
    # Helper to load inertial signals
    def load_inertial_signals(subset):
        signals = ['body_acc_x', 'body_acc_y', 'body_acc_z', 'body_gyro_x', 'body_gyro_y', 'body_gyro_z', 'total_acc_x', 'total_acc_y', 'total_acc_z']
        data = [pd.read_csv(DATASET_PATH / subset / 'Inertial Signals' / f'{s}_{subset}.txt', header=None, delim_whitespace=True).values for s in signals]
        return np.stack(data, axis=-1)

    # Load all raw data components
    y_train = pd.read_csv(DATASET_PATH / 'train/y_train.txt', header=None).values.flatten() - 1
    y_test = pd.read_csv(DATASET_PATH / 'test/y_test.txt', header=None).values.flatten() - 1
    subjects_train = pd.read_csv(DATASET_PATH / 'train/subject_train.txt', header=None).values.flatten()
    X_train_features = pd.read_csv(DATASET_PATH / 'train/X_train.txt', header=None, delim_whitespace=True).values
    X_test_features = pd.read_csv(DATASET_PATH / 'test/X_test.txt', header=None, delim_whitespace=True).values
    X_train_tcn = load_inertial_signals('train')
    X_test_tcn = load_inertial_signals('test')

    # Load activity labels for confusion matrix plotting
    activity_labels = pd.read_csv(DATASET_PATH / 'activity_labels.txt', header=None, delim_whitespace=True)[1].tolist()

    # --- Pre-computation Scaling (Fit on full train data before splitting) ---
    tcn_scaler = StandardScaler()
    X_train_tcn_scaled = tcn_scaler.fit_transform(X_train_tcn.reshape(-1, X_train_tcn.shape[-1])).reshape(X_train_tcn.shape)
    X_test_tcn_scaled = tcn_scaler.transform(X_test_tcn.reshape(-1, X_test_tcn.shape[-1])).reshape(X_test_tcn.shape)

    feature_scaler = StandardScaler()
    X_train_features_scaled = feature_scaler.fit_transform(X_train_features)
    X_test_features_scaled = feature_scaler.transform(X_test_features)
    print("TCN and Static Feature data scaled.")

    # --- Partition data by subject ID for FL clients ---
    client_data = {}
    client_ids = np.unique(subjects_train)
    for subject_id in client_ids:
        indices = np.where(subjects_train == subject_id)[0]
        # Each client gets their slice of BOTH data types
        client_data[subject_id] = (
            X_train_tcn_scaled[indices],
            X_train_features_scaled[indices],
            y_train[indices]
        )
    print(f"Data partitioned into {len(client_ids)} clients.")

    return client_data, list(client_ids), X_train_tcn_scaled, X_train_features_scaled, y_train, X_test_tcn_scaled, X_test_features_scaled, y_test, activity_labels

# Load all data
client_partitions, CLIENT_IDS, X_train_tcn_full, X_train_feat_full, y_train_full, X_test_tcn, X_test_feat, y_test, ACTIVITY_LABELS = load_and_partition_hybrid_data()


# ==============================================================================
# --- Block 3: Hybrid Model and FL Algorithm Definition ---
# ==============================================================================
print("\n--- Block 3: Defining Hybrid Model and FL Logic ---")

# Model definition functions are copied from the improved centralized script
def residual_block(x, dilation_rate, nb_filters, kernel_size, dropout_rate, regularizer):
    prev_x = x
    conv1 = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal', kernel_regularizer=regularizer)(x)
    conv1 = BatchNormalization()(conv1); conv1 = Activation('relu')(conv1); conv1 = SpatialDropout1D(dropout_rate)(conv1)
    conv2 = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal', kernel_regularizer=regularizer)(conv1)
    conv2 = BatchNormalization()(conv2); conv2 = Activation('relu')(conv2); conv2 = SpatialDropout1D(dropout_rate)(conv2)
    if prev_x.shape[-1] != conv2.shape[-1]: prev_x = Conv1D(nb_filters, 1, padding='same', kernel_regularizer=regularizer)(prev_x)
    return Add()([prev_x, conv2])

def build_hybrid_model(tcn_input_shape, features_input_shape, num_classes):
    l2_reg = l2(L2_REG_STRENGTH)
    input_tcn = Input(shape=tcn_input_shape, name='tcn_input')
    x = input_tcn
    for rate in DILATION_RATES: x = residual_block(x, rate, NUM_FILTERS, KERNEL_SIZE, SPATIAL_DROPOUT_RATE, l2_reg)
    tcn_features = GlobalAveragePooling1D(name='tcn_feature_vector')(x)
    input_features = Input(shape=features_input_shape, name='features_input')
    static_features = BatchNormalization()(input_features)
    static_features = Dense(256, activation='relu', kernel_regularizer=l2_reg)(static_features)
    static_features = Dropout(DENSE_DROPOUT_RATE)(static_features)
    static_features = Dense(128, activation='relu', kernel_regularizer=l2_reg, name='static_feature_vector')(static_features)
    merged = Concatenate(name='merged_features')([tcn_features, static_features])
    final_classifier = BatchNormalization()(merged)
    final_classifier = Dropout(DENSE_DROPOUT_RATE)(final_classifier)
    final_classifier = Dense(128, activation='relu', kernel_regularizer=l2_reg)(final_classifier)
    final_classifier = Dropout(DENSE_DROPOUT_RATE)(final_classifier)
    output_layer = Dense(num_classes, activation='softmax', name='output')(final_classifier)
    model = Model(inputs=[input_tcn, input_features], outputs=output_layer, name='Improved_Hybrid_HAR_Model')
    optimizer = Adam(learning_rate=INITIAL_LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# --- IMPROVEMENT: Weighted Federated Averaging ---
def weighted_federated_average(client_weights_list, client_sizes):
    """Computes the weighted average of client model weights."""
    if not client_weights_list: return None
    total_samples = sum(client_sizes)
    # Initialize avg_weights with zeros, with the same shape as the first client's weights
    avg_weights = [np.zeros_like(w) for w in client_weights_list[0]]
    
    for i in range(len(client_weights_list)):
        client_weights = client_weights_list[i]
        client_size = client_sizes[i]
        for layer_idx, layer_weights in enumerate(client_weights):
            avg_weights[layer_idx] += (layer_weights * client_size) / total_samples
            
    return avg_weights

# --- NEW: Evaluation and Plotting Helper Functions ---
def evaluate_model(model, x_tcn, x_feat, y_true):
    """Calculates accuracy, F1-score, and confusion matrix for a given model and data."""
    y_pred_probs = model.predict([x_tcn, x_feat], verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    
    return {"accuracy": acc, "f1_score": f1, "confusion_matrix": cm}

def plot_confusion_matrix(cm, class_names, title, save_path):
    """Plots and saves a confusion matrix heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close() # Close figure to avoid displaying it in console
    print(f"Confusion matrix saved to: {save_path}")


# ==============================================================================
# --- Block 4: Main FL Simulation and Benchmarking ---
# ==============================================================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("--- Starting Hybrid Model Federated Learning Simulation ---")
    print("="*50)

    tcn_shape = (WINDOW_SIZE, X_train_tcn_full.shape[-1])
    features_shape = (X_train_feat_full.shape[1],)
    num_classes = len(np.unique(y_train_full))

    # --- 1. Federated Learning Simulation ---
    global_model = build_hybrid_model(tcn_shape, features_shape, num_classes)
    local_model = build_hybrid_model(tcn_shape, features_shape, num_classes) # Reusable client model
    
    fl_accuracies = []
    best_fl_accuracy = 0.0
    best_fl_weights = None
    best_round = 0
    start_time = time.time()

    for round_num in range(COMMUNICATION_ROUNDS):
        print(f"\n--- Communication Round {round_num + 1}/{COMMUNICATION_ROUNDS} ---")
        
        global_weights = global_model.get_weights()
        client_updates = []
        client_sample_sizes = []

        selected_client_ids = random.sample(CLIENT_IDS, CLIENTS_PER_ROUND)
        print(f"  Training on {len(selected_client_ids)} selected clients...")

        for client_id in selected_client_ids:
            local_model.set_weights(global_weights)
            
            X_tcn_client, X_feat_client, y_client = client_partitions[client_id]
            
            # Use tf.data for efficient training
            client_dataset = tf.data.Dataset.from_tensor_slices(
                ((X_tcn_client, X_feat_client), y_client)
            ).shuffle(buffer_size=len(y_client)).batch(BATCH_SIZE)

            local_model.fit(client_dataset, epochs=LOCAL_EPOCHS, verbose=0)
            
            client_updates.append(local_model.get_weights())
            client_sample_sizes.append(len(y_client))
        
        # Server aggregation using weighted average
        new_global_weights = weighted_federated_average(client_updates, client_sample_sizes)
        global_model.set_weights(new_global_weights)
        
        # Evaluate global model on the hold-out test set
        loss, accuracy = global_model.evaluate([X_test_tcn, X_test_feat], y_test, verbose=0)
        fl_accuracies.append(accuracy)
        print(f"  Global model test accuracy after round {round_num + 1}: {accuracy:.4f}")

        # Track the best performing model
        if accuracy > best_fl_accuracy:
            best_fl_accuracy = accuracy
            best_fl_weights = global_model.get_weights()
            best_round = round_num + 1
            print(f"  *** New best model found with accuracy: {best_fl_accuracy:.4f} at round {best_round} ***")

    fl_end_time = time.time()
    fl_duration = fl_end_time - start_time
    print(f"\nFL Simulation finished in {fl_duration:.2f} seconds.")

    # --- 2. Centralized Training for Benchmark ---
    print("\n" + "="*50)
    print("--- Training Centralized Model for Benchmark ---")
    print("="*50)
    centralized_model = build_hybrid_model(tcn_shape, features_shape, num_classes)
    
    centralized_model.fit([X_train_tcn_full, X_train_feat_full], y_train_full, 
                          epochs=40, # More epochs for centralized to converge fully
                          batch_size=BATCH_SIZE,
                          validation_data=([X_test_tcn, X_test_feat], y_test),
                          verbose=1)
    
    _, centralized_accuracy = centralized_model.evaluate([X_test_tcn, X_test_feat], y_test, verbose=0)

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
    
    # Load the best model's weights
    best_global_model = build_hybrid_model(tcn_shape, features_shape, num_classes)
    best_global_model.set_weights(best_fl_weights)

    # Evaluate on the Global Test Set
    print("\n--- Performance on Global Test Set ---")
    test_metrics = evaluate_model(best_global_model, X_test_tcn, X_test_feat, y_test)
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Weighted F1-Score: {test_metrics['f1_score']:.4f}")
    plot_confusion_matrix(test_metrics['confusion_matrix'], ACTIVITY_LABELS,
                          title='Confusion Matrix - Best FL Model on Test Set',
                          save_path=OUTPUT_DIR / 'cm_fl_best_model_test_set.png')

    # Evaluate on the combined Client Training Data
    print("\n--- Performance on Combined Client Training Data ---")
    train_metrics = evaluate_model(best_global_model, X_train_tcn_full, X_train_feat_full, y_train_full)
    print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"  Weighted F1-Score: {train_metrics['f1_score']:.4f}")
    plot_confusion_matrix(train_metrics['confusion_matrix'], ACTIVITY_LABELS,
                          title='Confusion Matrix - Best FL Model on Client Training Data',
                          save_path=OUTPUT_DIR / 'cm_fl_best_model_train_set.png')
    
    # --- 5. Performance Curve Plot ---
    plt.figure(figsize=(12, 7))
    plt.plot(range(1, COMMUNICATION_ROUNDS + 1), fl_accuracies, marker='o', linestyle='-', label='Federated Model Accuracy per Round')
    plt.axhline(y=centralized_accuracy, color='r', linestyle='--', label=f'Centralized Benchmark ({centralized_accuracy:.4f})')
    plt.axhline(y=best_fl_accuracy, color='g', linestyle=':', label=f'Best FL Model Accuracy ({best_fl_accuracy:.4f})')
    plt.title("Federated Learning Performance (Hybrid Model)")
    plt.xlabel("Communication Round")
    plt.ylabel("Global Model Test Accuracy")
    plt.xticks(range(0, COMMUNICATION_ROUNDS + 1, 5))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_path = OUTPUT_DIR / 'fl_hybrid_performance_curve.png'
    plt.savefig(plot_path)
    print(f"\nPerformance plot saved to: {plot_path}")