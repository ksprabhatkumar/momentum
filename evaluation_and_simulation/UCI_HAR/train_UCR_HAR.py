# ==============================================================================
# train_uci_har_hybrid_improved.py
#
# Trains an improved hybrid model on the UCI HAR Dataset. This model fuses two branches:
# 1. A regularized TCN branch that learns features from the raw inertial time-series.
# 2. An enhanced MLP branch that learns a better representation from the 561 static features.
# The learned temporal and static feature representations are concatenated and
# fed into a regularized dense classifier head.
#
# Improvements over the original:
# - L2 regularization added to all Conv1D and Dense layers to reduce overfitting.
# - Enhanced static feature branch with its own dense layers to learn a better representation.
# - Scaling (StandardScaler) applied to the static features, a crucial preprocessing step.
# - Explicit Adam optimizer with a defined initial learning rate.
# - More parameters exposed for easier tuning (e.g., dropout rates, L2 strength).
# - Training history (accuracy/loss) is now plotted and saved.
# - Callbacks are tuned for more responsive training.
#
# To Run:
# (tf_venv) sandeep@AcerNitro:~/tf_project/UCI_HAR$ python train_uci_har_hybrid_improved.py
# ==============================================================================

# --- Block 1: Setup and Initial Configuration ---
print("--- Block 1: Setup and Initial Configuration ---")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Add, Concatenate
from tensorflow.keras.layers import Conv1D, SpatialDropout1D, GlobalAveragePooling1D, Activation
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# --- Main Configuration ---
DATASET_PATH = Path('./data/UCI HAR Dataset/')
OUTPUT_DIR = Path('./results_uci_har_hybrid_improved/')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Model & Training Parameters ---
# WINDOW_SIZE is fixed at 128 for the UCI HAR dataset's inertial signals.
WINDOW_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 80 # Increased epochs slightly to allow the regularized model to converge
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 15 # Reduced for more responsive early stopping

# --- Hyperparameters for Tuning ---
INITIAL_LEARNING_RATE = 0.001
L2_REG_STRENGTH = 1e-4 # Strength of L2 regularization
SPATIAL_DROPOUT_RATE = 0.15 # Dropout for TCN feature maps
DENSE_DROPOUT_RATE = 0.45 # Dropout for the final classifier head

# TCN specific parameters
KERNEL_SIZE = 7
NUM_FILTERS = 64
NUM_TCN_BLOCKS = 4
DILATION_RATES = [2**i for i in range(NUM_TCN_BLOCKS)]

# --- GPU Check ---
print("\n--- GPU Check ---")
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print(f"Found {len(gpu_devices)} GPU(s). Enabling memory growth.")
    try:
        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Could not set memory growth: {e}")
else:
    print("!!! No GPU found. Training will use CPU. !!!")
print("-" * 25)

# ==============================================================================
# --- Block 2: Data Loading and Utility Functions ---
# ==============================================================================
print("\n--- Block 2: Loading Data and Defining Utilities ---")

def load_labels_and_features():
    """Loads all necessary data components for the hybrid model."""
    activity_labels_df = pd.read_csv(DATASET_PATH / 'activity_labels.txt', header=None, delim_whitespace=True)
    activity_labels = activity_labels_df[1].tolist()
    
    y_train = pd.read_csv(DATASET_PATH / 'train/y_train.txt', header=None).values.flatten() - 1
    y_test = pd.read_csv(DATASET_PATH / 'test/y_test.txt', header=None).values.flatten() - 1
    
    X_train_features = pd.read_csv(DATASET_PATH / 'train/X_train.txt', header=None, delim_whitespace=True).values
    X_test_features = pd.read_csv(DATASET_PATH / 'test/X_test.txt', header=None, delim_whitespace=True).values

    return y_train, y_test, X_train_features, X_test_features, activity_labels

def load_inertial_signals(subset):
    """Loads and stacks the 9 inertial signals for the TCN input branch."""
    signals_to_load = [
        'body_acc_x', 'body_acc_y', 'body_acc_z',
        'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
        'total_acc_x', 'total_acc_y', 'total_acc_z'
    ]
    signals_data = []
    base_path = DATASET_PATH / subset / 'Inertial Signals'
    
    print(f"Loading inertial signals for '{subset}' set...")
    for signal in signals_to_load:
        filepath = base_path / f'{signal}_{subset}.txt'
        df = pd.read_csv(filepath, header=None, delim_whitespace=True)
        signals_data.append(df.values)
    
    return np.stack(signals_data, axis=-1)

def plot_confusion_matrix(y_true, y_pred, labels, title, filename):
    """Plots and saves a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()
    print(f"Confusion matrix saved to {OUTPUT_DIR / filename}")

def plot_training_history(history, filename):
    """Plots and saves the training and validation loss and accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Plot training & validation accuracy values
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper left')
    ax1.grid(True)
    
    # Plot training & validation loss values
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()
    print(f"Training history plot saved to {OUTPUT_DIR / filename}")

# ==============================================================================
# --- Block 3: Hybrid Model Definition (Improved) ---
# ==============================================================================
print("\n--- Block 3: Defining the Improved Hybrid TCN + Features Model ---")

def residual_block(x, dilation_rate, nb_filters, kernel_size, dropout_rate, regularizer):
    prev_x = x
    conv1 = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal', kernel_regularizer=regularizer)(x)
    conv1 = BatchNormalization()(conv1); conv1 = Activation('relu')(conv1); conv1 = SpatialDropout1D(dropout_rate)(conv1)
    
    conv2 = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal', kernel_regularizer=regularizer)(conv1)
    conv2 = BatchNormalization()(conv2); conv2 = Activation('relu')(conv2); conv2 = SpatialDropout1D(dropout_rate)(conv2)
    
    if prev_x.shape[-1] != conv2.shape[-1]:
        prev_x = Conv1D(nb_filters, 1, padding='same', kernel_regularizer=regularizer)(prev_x)
    return Add()([prev_x, conv2])

def build_hybrid_model(tcn_input_shape, features_input_shape, num_classes):
    l2_reg = l2(L2_REG_STRENGTH)

    # --- TCN Branch for Time-Series Data ---
    input_tcn = Input(shape=tcn_input_shape, name='tcn_input')
    x = input_tcn
    for rate in DILATION_RATES:
        x = residual_block(x, rate, NUM_FILTERS, KERNEL_SIZE, SPATIAL_DROPOUT_RATE, l2_reg)
    tcn_features = GlobalAveragePooling1D(name='tcn_feature_vector')(x)

    # --- Enhanced Feature Branch for Pre-computed Data ---
    # This branch now has its own small MLP to learn a better representation
    # of the static features before fusion.
    input_features = Input(shape=features_input_shape, name='features_input')
    static_features = BatchNormalization()(input_features)
    static_features = Dense(256, activation='relu', kernel_regularizer=l2_reg)(static_features)
    static_features = Dropout(DENSE_DROPOUT_RATE)(static_features)
    static_features = Dense(128, activation='relu', kernel_regularizer=l2_reg, name='static_feature_vector')(static_features)

    # --- Concatenate (Fuse) the two branches ---
    merged = Concatenate(name='merged_features')([tcn_features, static_features])

    # --- Classifier Head ---
    final_classifier = BatchNormalization()(merged)
    final_classifier = Dropout(DENSE_DROPOUT_RATE)(final_classifier)
    final_classifier = Dense(128, activation='relu', kernel_regularizer=l2_reg)(final_classifier)
    final_classifier = Dropout(DENSE_DROPOUT_RATE)(final_classifier)
    output_layer = Dense(num_classes, activation='softmax', name='output')(final_classifier)
    
    # --- Create and Compile the Model ---
    model = Model(inputs=[input_tcn, input_features], outputs=output_layer, name='Improved_Hybrid_HAR_Model')
    optimizer = Adam(learning_rate=INITIAL_LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# ==============================================================================
# --- Block 4: Training and Evaluation ---
# ==============================================================================
def run_hybrid_training():
    print("\n" + "="*50)
    print("--- Running Improved Hybrid Model Training and Evaluation ---")
    print("="*50)

    # --- Load all data components ---
    y_train, y_test, X_train_features, X_test_features, activity_labels = load_labels_and_features()
    X_train_tcn = load_inertial_signals('train')
    X_test_tcn = load_inertial_signals('test')
    num_classes = len(activity_labels)

    # --- Preprocessing: Scale both TCN and Static Feature data ---
    # Scale TCN (inertial) data
    num_samples_train, timesteps, num_features_tcn = X_train_tcn.shape
    tcn_scaler = StandardScaler()
    X_train_tcn_scaled = tcn_scaler.fit_transform(X_train_tcn.reshape(-1, num_features_tcn)).reshape(X_train_tcn.shape)
    X_test_tcn_scaled = tcn_scaler.transform(X_test_tcn.reshape(-1, num_features_tcn)).reshape(X_test_tcn.shape)
    print("Inertial signal (TCN) scaling complete.")

    # Scale static feature data
    feature_scaler = StandardScaler()
    X_train_features_scaled = feature_scaler.fit_transform(X_train_features)
    X_test_features_scaled = feature_scaler.transform(X_test_features)
    print("Static features scaling complete.")

    # --- Build the Hybrid Model ---
    model = build_hybrid_model(
        tcn_input_shape=(WINDOW_SIZE, num_features_tcn),
        features_input_shape=(X_train_features.shape[1],),
        num_classes=num_classes
    )
    model.summary()
    tf.keras.utils.plot_model(model, to_file=OUTPUT_DIR / 'improved_hybrid_model_architecture.png', show_shapes=True)

    # --- Callbacks ---
    checkpoint_path = OUTPUT_DIR / 'improved_hybrid_har_model.keras'
    model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True)

    # --- Train the Model ---
    print("\nTraining Improved Hybrid model...")
    history = model.fit(
        x=[X_train_tcn_scaled, X_train_features_scaled],
        y=y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[model_checkpoint, reduce_lr, early_stopping],
        verbose=1
    )

    # --- Plot and Save Training History ---
    plot_training_history(history, "improved_hybrid_training_history.png")

    # --- Evaluate the Best Model ---
    print("\nEvaluating Improved Hybrid model on test data...")
    # Loading the best saved model ensures we evaluate the peak performance
    best_model = tf.keras.models.load_model(checkpoint_path)
    loss, accuracy = best_model.evaluate(
        x=[X_test_tcn_scaled, X_test_features_scaled],
        y=y_test,
        verbose=0
    )
    print(f"Improved Hybrid Model Test Accuracy: {accuracy:.4f}")

    y_pred_probs = best_model.predict([X_test_tcn_scaled, X_test_features_scaled])
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    print("\nImproved Hybrid Model Classification Report:")
    print(classification_report(y_test, y_pred, target_names=activity_labels))
    plot_confusion_matrix(y_test, y_pred, activity_labels, "Improved Hybrid Model Confusion Matrix", "improved_hybrid_confusion_matrix.png")

# ==============================================================================
# --- Main Execution Block ---
# ==============================================================================
if __name__ == "__main__":
    run_hybrid_training()
    print("\n" + "="*50)
    print("--- SCRIPT COMPLETE ---")
    print(f"All models and plots saved in: {OUTPUT_DIR}")
    print("="*50)