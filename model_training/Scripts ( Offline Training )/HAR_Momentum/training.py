#!/usr/bin/env python3
# training.py
# Enhanced training script: trains TCN for HAR and produces extensive evaluation metrics & plots
# Place this file in ~/tf_project/HAR_Momentum/ and run with your tf_venv activated.

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    auc,
    average_precision_score,
    precision_recall_curve,
    cohen_kappa_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    top_k_accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------- Configuration -----------------
WINDOW_SIZE = 60
STRIDE = 15
KERNEL_SIZE = 7
NUM_FILTERS = 64
NUM_TCN_BLOCKS = 5
DILATION_RATES = [2**i for i in range(NUM_TCN_BLOCKS)]
SPATIAL_DROPOUT_RATE = 0.15
FINAL_DROPOUT_RATE = 0.3
L2_REG = 1e-4

BATCH_SIZE = 64
EPOCHS = 100
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 15
RANDOM_STATE = 42
TOP_K = 2  # for top-k accuracy

# Activity mapping (mirror notebooks)
ACTIVITIES_FROM_FILE1 = ['B', 'D', 'E']
ACTIVITIES_FROM_FILE2 = ['A', 'C']
ALL_ACTIVITIES_TO_KEEP = sorted(ACTIVITIES_FROM_FILE1 + ACTIVITIES_FROM_FILE2)

# Paths (relative to script location)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
RESULTS_BASE_DIR = BASE_DIR / 'results' / 'TCN_Training_Assets'
RESULTS_BASE_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR = RESULTS_BASE_DIR / 'metrics'
METRICS_DIR.mkdir(exist_ok=True)

INPUT_CSV_1 = DATA_DIR / 'resampled_normalized_phone_data.csv'
INPUT_CSV_2 = DATA_DIR / 'combined_collected_data.csv'

FILE_PREFIX = f"tcn_v3_{''.join(ALL_ACTIVITIES_TO_KEEP)}_"
LOCAL_WEIGHTS = RESULTS_BASE_DIR / 'best_weights.weights.h5'   # must end with .weights.h5 for save_weights_only
KERAS_FINAL = RESULTS_BASE_DIR / f'{FILE_PREFIX}har_model.keras'
SCALER_JSON = RESULTS_BASE_DIR / f'{FILE_PREFIX}scaler.json'
LABELS_JSON = RESULTS_BASE_DIR / f'{FILE_PREFIX}labels.json'
TFLITE_PATH = RESULTS_BASE_DIR / f'{FILE_PREFIX}har_model.tflite'

# Paths for metrics and plots
HISTORY_PLOT_PATH = METRICS_DIR / f'{FILE_PREFIX}training_history.png'
CONFUSION_MATRIX_TFLITE_PATH = METRICS_DIR / f'{FILE_PREFIX}confusion_matrix_tflite.png'
CONFUSION_MATRIX_KERAS_PATH = METRICS_DIR / f'{FILE_PREFIX}confusion_matrix_keras.png'
TFLITE_REPORT_PATH = METRICS_DIR / f'{FILE_PREFIX}classification_report_tflite.txt'
KERAS_REPORT_PATH = METRICS_DIR / f'{FILE_PREFIX}classification_report_keras.txt'
METRICS_JSON_PATH = METRICS_DIR / f'{FILE_PREFIX}metrics_summary.json'
METRICS_CSV_PATH = METRICS_DIR / f'{FILE_PREFIX}metrics_table.csv'
PER_CLASS_CSV_KERAS = METRICS_DIR / f'{FILE_PREFIX}per_class_keras.csv'
PER_CLASS_CSV_TFLITE = METRICS_DIR / f'{FILE_PREFIX}per_class_tflite.csv'
ROC_PLOT_KERAS_PATH = METRICS_DIR / f'{FILE_PREFIX}roc_keras.png'
ROC_PLOT_TFLITE_PATH = METRICS_DIR / f'{FILE_PREFIX}roc_tflite.png'
PR_PLOT_KERAS_PATH = METRICS_DIR / f'{FILE_PREFIX}pr_keras.png'
PR_PLOT_TFLITE_PATH = METRICS_DIR / f'{FILE_PREFIX}pr_tflite.png'
F1_BAR_PATH = METRICS_DIR / f'{FILE_PREFIX}f1_bar_comparison.png'

# ----------------- Utilities -----------------
def log(msg=''):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ----------------- Plotting Functions -----------------
def plot_history(history, file_path):
    log(f"Generating training history plot to {file_path}...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(history.history.get('accuracy', []), label='Train Accuracy')
    ax1.plot(history.history.get('val_accuracy', []), label='Validation Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.set_title('Model Accuracy')
    ax1.grid(True)

    ax2.plot(history.history.get('loss', []), label='Train Loss')
    ax2.plot(history.history.get('val_loss', []), label='Validation Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(loc='upper right')
    ax2.set_title('Model Loss')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    log("... plot saved.")

def plot_confusion_matrix(y_true, y_pred, class_names, file_path, title='Confusion Matrix'):
    log(f"Generating confusion matrix plot to {file_path}...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    log("... plot saved.")

def plot_roc(y_true_binarized, y_score, class_names, file_path, title='ROC Curves'):
    log(f"Generating ROC curves to {file_path}...")
    n_classes = y_true_binarized.shape[1]
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        try:
            fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
        except ValueError:
            log(f"Could not compute ROC for class {class_names[i]}")
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    log("... ROC saved.")

def plot_pr(y_true_binarized, y_score, class_names, file_path, title='Precision-Recall Curves'):
    log(f"Generating PR curves to {file_path}...")
    n_classes = y_true_binarized.shape[1]
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        try:
            precision, recall, _ = precision_recall_curve(y_true_binarized[:, i], y_score[:, i])
            ap = average_precision_score(y_true_binarized[:, i], y_score[:, i])
            plt.plot(recall, precision, lw=2, label=f'{class_names[i]} (AP = {ap:.3f})')
        except ValueError:
            log(f"Could not compute PR for class {class_names[i]}")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    log("... PR saved.")

def plot_f1_bar(f1_keras, f1_tflite, class_names, file_path):
    log(f"Generating F1 comparison bar chart to {file_path}...")
    x = np.arange(len(class_names))
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, f1_keras, width, label='Keras F1')
    plt.bar(x + width/2, f1_tflite, width, label='TFLite F1')
    plt.xticks(x, class_names)
    plt.ylabel('F1 Score')
    plt.title('Per-class F1: Keras vs TFLite')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    log("... F1 bar saved.")

# ----------------- GPU setup -----------------
log("TensorFlow version: " + tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    log(f"✅ Using GPU(s): {[g.name for g in gpus]}")
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception as e:
        log(f"Could not set memory growth: {e}")
else:
    log("⚠️ No GPU detected. Using CPU.")

# Reproducibility
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# ----------------- Data loading & checks -----------------
log("Checking input CSV files...")
missing = []
for p in (INPUT_CSV_1, INPUT_CSV_2):
    if not p.exists():
        missing.append(str(p.relative_to(BASE_DIR)))
if missing:
    log("ERROR: Missing expected CSV files in data/:")
    for m in missing:
        log("  - " + m)
    log("Files present in data/:")
    for f in sorted([str(x.name) for x in DATA_DIR.glob('*')]):
        log("  - " + f)
    raise FileNotFoundError("Missing CSVs. Place expected csvs in data/ and re-run.")

log("Loading CSVs...")
df1 = pd.read_csv(INPUT_CSV_1)
df2 = pd.read_csv(INPUT_CSV_2)

# Filter activities
df1_filtered = df1[df1['activity'].isin(ACTIVITIES_FROM_FILE1)].copy()
df2_filtered = df2[df2['activity'].isin(ACTIVITIES_FROM_FILE2)].copy()
combined_df = pd.concat([df1_filtered, df2_filtered], ignore_index=True)
log(f"Combined rows after filtering: {len(combined_df)}")

# ----------------- Windowing -----------------
def create_subject_activity_windows(df, window_size, stride):
    windows, labels, subject_ids = [], [], []
    required_cols = ['x_accel', 'y_accel', 'z_accel', 'x_gyro', 'y_gyro', 'z_gyro']
    for c in required_cols + ['timestamp', 'subject', 'activity']:
        if c not in df.columns:
            raise KeyError(f"Required column '{c}' not found in dataframe.")
    grouped = df.groupby(['subject', 'activity'])
    log(f"Processing {len(grouped)} subject-activity groups for windowing...")
    for name, group_df in grouped:
        data_values = group_df.sort_values('timestamp')[required_cols].values
        if len(data_values) < window_size:
            continue
        for start in range(0, len(data_values) - window_size + 1, stride):
            windows.append(data_values[start : start + window_size])
            labels.append(group_df["activity"].iloc[0])
            subject_ids.append(group_df["subject"].iloc[0])
    if not windows:
        raise ValueError("No windows created. Check window_size/stride or dataset lengths.")
    return np.array(windows), np.array(labels), np.array(subject_ids)

log("Creating windows...")
X, y_raw, subjects = create_subject_activity_windows(combined_df, WINDOW_SIZE, STRIDE)
log(f"Created windows: {X.shape}")

# ----------------- Label encoding -----------------
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
num_classes = len(label_encoder.classes_)
log(f"Encoded classes ({num_classes}): {list(label_encoder.classes_)}")

# ----------------- Train/test split -----------------
log("Splitting into train/test...")
X_train, X_test, y_train, y_test, subjects_train, subjects_test = train_test_split(
    X, y, subjects, test_size=0.25, random_state=RANDOM_STATE, stratify=y)

# ----------------- Scaling -----------------
log("Fitting StandardScaler on training data (preventing leakage)...")
num_samples_train, timesteps, num_features = X_train.shape
scaler = StandardScaler()
X_train_reshaped = X_train.reshape(-1, num_features)
X_train_scaled_reshaped = scaler.fit_transform(X_train_reshaped)
X_train_scaled = X_train_scaled_reshaped.reshape(num_samples_train, timesteps, num_features)

X_test_reshaped = X_test.reshape(-1, num_features)
X_test_scaled_reshaped = scaler.transform(X_test_reshaped)
X_test_scaled = X_test_scaled_reshaped.reshape(X_test.shape[0], timesteps, num_features)

X_train_scaled = np.ascontiguousarray(X_train_scaled.astype(np.float32))
X_test_scaled = np.ascontiguousarray(X_test_scaled.astype(np.float32))
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
log("Feature scaling complete. Dtypes: X_train=%s y_train=%s" % (X_train_scaled.dtype, y_train.dtype))

# ----------------- Build TCN model -----------------
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Add, Conv1D, SpatialDropout1D, GlobalAveragePooling1D, Activation
from tensorflow.keras.regularizers import l2

def residual_block(x, dilation_rate, nb_filters, kernel_size, dropout_rate=0.0, block_id=0):
    prev_x = x
    conv1 = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='same',
                   kernel_regularizer=l2(L2_REG))(x)
    conv1 = BatchNormalization()(conv1); conv1 = Activation('relu')(conv1); conv1 = SpatialDropout1D(dropout_rate)(conv1)
    conv2 = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='same',
                   kernel_regularizer=l2(L2_REG))(conv1)
    conv2 = BatchNormalization()(conv2); conv2 = Activation('relu')(conv2); conv2 = SpatialDropout1D(dropout_rate)(conv2)
    if prev_x.shape[-1] != conv2.shape[-1]:
        prev_x = Conv1D(nb_filters, 1, padding='same')(prev_x)
    return Add()([prev_x, conv2])

def build_tcn_model(input_shape, num_classes):
    input_layer = Input(shape=input_shape, name='input_layer')
    x = input_layer
    for i, rate in enumerate(DILATION_RATES):
        x = residual_block(x, rate, NUM_FILTERS, KERNEL_SIZE, SPATIAL_DROPOUT_RATE, block_id=i+1)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(FINAL_DROPOUT_RATE)(x)
    output_layer = Dense(num_classes, activation='softmax', name='output_dense')(x)
    return Model(inputs=input_layer, outputs=output_layer)

log("Building model...")
model = build_tcn_model((WINDOW_SIZE, num_features), num_classes)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary(print_fn=lambda s: log(s))

# ----------------- Prepare train/val datasets (tf.data) -----------------
log("Creating train/validation split...")
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train_scaled, y_train, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE, stratify=y_train
)
log(f"Train shape: {X_train_final.shape}, Val shape: {X_val.shape}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = tf.data.Dataset.from_tensor_slices((X_train_final, y_train_final))
train_ds = train_ds.shuffle(2048, seed=RANDOM_STATE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE).prefetch(AUTOTUNE)

# ----------------- Callbacks -----------------
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None): self.epoch_times = []
    def on_epoch_begin(self, epoch, logs=None): self._start = time.time()
    def on_epoch_end(self, epoch, logs=None):
        dt = time.time() - self._start
        self.epoch_times.append(dt)
        log(f"Epoch {epoch} time: {dt:.1f}s")

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1)

checkpoint_cb = ModelCheckpoint(
    filepath=str(LOCAL_WEIGHTS),
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=True,
    mode='max',
    verbose=1
)

cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_final), y=y_train_final)
class_weights_dict = dict(enumerate(cw))
log(f"Class weights: {class_weights_dict}")

time_cb = TimeHistory()

# ----------------- Train -----------------
log("Starting training...")
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[checkpoint_cb, reduce_lr, early_stopping, time_cb],
    class_weight=class_weights_dict,
    verbose=1
)

# ----------------- Generate and Save Training Plots -----------------
plot_history(history, HISTORY_PLOT_PATH)

# ----------------- Load best weights and save final Keras model -----------------
log("Training finished; loading best weights...")
model.load_weights(str(LOCAL_WEIGHTS))

log(f"Saving full Keras model to {KERAS_FINAL} ...")
model.save(str(KERAS_FINAL))

# ----------------- Save scaler & labels (JSON) -----------------
log("Saving scaler and label mapping...")
scaler_dict = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}
with open(SCALER_JSON, 'w') as f:
    json.dump(scaler_dict, f, indent=4)

labels_dict = {str(i): label for i, label in enumerate(label_encoder.classes_)}
with open(LABELS_JSON, 'w') as f:
    json.dump(labels_dict, f, indent=4)

log("Saved scaler and labels JSON.")

# ----------------- Save definitive test set arrays -----------------
test_set_dir = RESULTS_BASE_DIR / 'definitive_test_set'
test_set_dir.mkdir(exist_ok=True)
np.save(test_set_dir / 'X_test_scaled.npy', X_test_scaled)
np.save(test_set_dir / 'y_test.npy', y_test)
log(f"Saved test set to {test_set_dir}")

# ----------------- TFLite conversion (full integer quantization) -----------------
log("Converting to quantized TFLite model (full integer quantization)...")
def representative_dataset_gen():
    num_samples = min(300, X_train_final.shape[0])
    idxs = np.random.choice(X_train_final.shape[0], num_samples, replace=False)
    for i in idxs:
        sample = X_train_final[i:i+1].astype(np.float32)
        yield [sample]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

try:
    tflite_model = converter.convert()
    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)
    log(f"Saved quantized TFLite to: {TFLITE_PATH}")
except Exception as e:
    log("TFLite conversion failed: " + str(e))
    raise

# ----------------- TFLite verification -----------------
log("Verifying TFLite model on test set...")
interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

y_pred_tflite = []
y_prob_tflite = []
in_scale, in_zero_point = input_details.get('quantization', (None, None))
out_scale, out_zero_point = output_details.get('quantization', (None, None))

for i in range(X_test.shape[0]):
    raw_window = X_test[i]  # unscaled window
    test_window_scaled = scaler.transform(raw_window).astype(np.float32)

    if in_scale is None or in_scale == 0:
        interpreter.set_tensor(input_details['index'], test_window_scaled[np.newaxis, ...].astype(np.float32))
    else:
        q = (test_window_scaled / in_scale + in_zero_point).round().astype(input_details['dtype'])
        interpreter.set_tensor(input_details['index'], q[np.newaxis, ...])

    interpreter.invoke()
    out_q = interpreter.get_tensor(output_details['index'])[0]

    if out_scale is None or out_scale == 0:
        out_f = out_q.astype(np.float32)
    else:
        out_f = (out_q.astype(np.float32) - out_zero_point) * out_scale

    try:
        exp = np.exp(out_f - np.max(out_f))
        probs = exp / np.sum(exp)
    except Exception:
        probs = out_f / (np.sum(out_f) + 1e-9)

    y_prob_tflite.append(probs)
    y_pred_tflite.append(int(np.argmax(probs)))

y_pred_tflite = np.array(y_pred_tflite)
y_prob_tflite = np.vstack(y_prob_tflite)
tflite_acc = accuracy_score(y_test, y_pred_tflite)
log(f"TFLite test accuracy: {tflite_acc:.4f}")

# ----------------- Save TFLite Metrics and Plots -----------------
log("Saving TFLite classification report and metrics...")
report_tflite_str = classification_report(y_test, y_pred_tflite, target_names=label_encoder.classes_)
with open(TFLITE_REPORT_PATH, 'w') as f:
    f.write("TFLite Model Classification Report\n")
    f.write("====================================\n")
    f.write(report_tflite_str)
log(f"... report saved to {TFLITE_REPORT_PATH}")

plot_confusion_matrix(y_test, y_pred_tflite, label_encoder.classes_, CONFUSION_MATRIX_TFLITE_PATH, title='Confusion Matrix (TFLite)')

precision_tflite, recall_tflite, f1_tflite, support_tflite = precision_recall_fscore_support(y_test, y_pred_tflite, labels=range(num_classes), zero_division=0)
per_class_df_tflite = pd.DataFrame({
    'class': label_encoder.classes_,
    'precision': precision_tflite,
    'recall': recall_tflite,
    'f1': f1_tflite,
    'support': support_tflite
})
per_class_df_tflite.to_csv(PER_CLASS_CSV_TFLITE, index=False)
log(f"Per-class TFLite metrics saved to {PER_CLASS_CSV_TFLITE}")

# ----------------- Final report for Keras model -----------------
log("Evaluating original Keras model on test set...")
keras_loss, keras_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
y_prob_keras = model.predict(X_test_scaled)
y_pred_keras = np.argmax(y_prob_keras, axis=1)
log(f"Keras test accuracy: {keras_acc:.4f}")

log("Saving Keras classification report and metrics...")
report_keras_str = classification_report(y_test, y_pred_keras, target_names=label_encoder.classes_)
with open(KERAS_REPORT_PATH, 'w') as f:
    f.write("Keras Model Classification Report\n")
    f.write("=================================\n")
    f.write(report_keras_str)
log(f"... report saved to {KERAS_REPORT_PATH}")

plot_confusion_matrix(y_test, y_pred_keras, label_encoder.classes_, CONFUSION_MATRIX_KERAS_PATH, title='Confusion Matrix (Keras)')

precision_keras, recall_keras, f1_keras, support_keras = precision_recall_fscore_support(y_test, y_pred_keras, labels=range(num_classes), zero_division=0)
per_class_df_keras = pd.DataFrame({
    'class': label_encoder.classes_,
    'precision': precision_keras,
    'recall': recall_keras,
    'f1': f1_keras,
    'support': support_keras
})
per_class_df_keras.to_csv(PER_CLASS_CSV_KERAS, index=False)
log(f"Per-class Keras metrics saved to {PER_CLASS_CSV_KERAS}")

# ----------------- Compute advanced metrics and plots -----------------
log("Computing advanced metrics (AUC, PR-AUC, Kappa, MCC, Balanced Acc, Top-K)...")
metrics_summary = {}

metrics_summary['keras_accuracy'] = float(keras_acc)
metrics_summary['tflite_accuracy'] = float(tflite_acc)

try:
    metrics_summary['kappa_keras'] = float(cohen_kappa_score(y_test, y_pred_keras))
except Exception:
    metrics_summary['kappa_keras'] = None
try:
    metrics_summary['kappa_tflite'] = float(cohen_kappa_score(y_test, y_pred_tflite))
except Exception:
    metrics_summary['kappa_tflite'] = None

try:
    metrics_summary['mcc_keras'] = float(matthews_corrcoef(y_test, y_pred_keras))
except Exception:
    metrics_summary['mcc_keras'] = None
try:
    metrics_summary['mcc_tflite'] = float(matthews_corrcoef(y_test, y_pred_tflite))
except Exception:
    metrics_summary['mcc_tflite'] = None

try:
    metrics_summary['balanced_accuracy_keras'] = float(balanced_accuracy_score(y_test, y_pred_keras))
except Exception:
    metrics_summary['balanced_accuracy_keras'] = None
try:
    metrics_summary['balanced_accuracy_tflite'] = float(balanced_accuracy_score(y_test, y_pred_tflite))
except Exception:
    metrics_summary['balanced_accuracy_tflite'] = None

try:
    metrics_summary[f'top{TOP_K}_accuracy_keras'] = float(top_k_accuracy_score(y_test, y_prob_keras, k=TOP_K, labels=range(num_classes)))
except Exception:
    metrics_summary[f'top{TOP_K}_accuracy_keras'] = None
try:
    metrics_summary[f'top{TOP_K}_accuracy_tflite'] = float(top_k_accuracy_score(y_test, y_prob_tflite, k=TOP_K, labels=range(num_classes)))
except Exception:
    metrics_summary[f'top{TOP_K}_accuracy_tflite'] = None

try:
    y_test_binarized = label_binarize(y_test, classes=range(num_classes))
    if y_prob_keras.shape[1] == num_classes:
        aucs_keras = {}
        ap_keras = {}
        for i in range(num_classes):
            try:
                auc_i = roc_auc_score(y_test_binarized[:, i], y_prob_keras[:, i])
                ap_i = average_precision_score(y_test_binarized[:, i], y_prob_keras[:, i])
            except Exception:
                auc_i = None
                ap_i = None
            aucs_keras[label_encoder.classes_[i]] = auc_i
            ap_keras[label_encoder.classes_[i]] = ap_i
        metrics_summary['auc_keras_per_class'] = aucs_keras
        metrics_summary['ap_keras_per_class'] = ap_keras

    if y_prob_tflite.shape[1] == num_classes:
        aucs_tflite = {}
        ap_tflite = {}
        for i in range(num_classes):
            try:
                auc_i = roc_auc_score(y_test_binarized[:, i], y_prob_tflite[:, i])
                ap_i = average_precision_score(y_test_binarized[:, i], y_prob_tflite[:, i])
            except Exception:
                auc_i = None
                ap_i = None
            aucs_tflite[label_encoder.classes_[i]] = auc_i
            ap_tflite[label_encoder.classes_[i]] = ap_i
        metrics_summary['auc_tflite_per_class'] = aucs_tflite
        metrics_summary['ap_tflite_per_class'] = ap_tflite

    try:
        metrics_summary['auc_keras_macro'] = float(roc_auc_score(y_test_binarized, y_prob_keras, average='macro', multi_class='ovr'))
    except Exception:
        metrics_summary['auc_keras_macro'] = None
    try:
        metrics_summary['auc_tflite_macro'] = float(roc_auc_score(y_test_binarized, y_prob_tflite, average='macro', multi_class='ovr'))
    except Exception:
        metrics_summary['auc_tflite_macro'] = None

    try:
        plot_roc(y_test_binarized, y_prob_keras, label_encoder.classes_, ROC_PLOT_KERAS_PATH, title='ROC Curves (Keras)')
    except Exception as e:
        log(f"Failed to generate Keras ROC: {e}")
    try:
        plot_roc(y_test_binarized, y_prob_tflite, label_encoder.classes_, ROC_PLOT_TFLITE_PATH, title='ROC Curves (TFLite)')
    except Exception as e:
        log(f"Failed to generate TFLite ROC: {e}")
    try:
        plot_pr(y_test_binarized, y_prob_keras, label_encoder.classes_, PR_PLOT_KERAS_PATH, title='PR Curves (Keras)')
    except Exception as e:
        log(f"Failed to generate Keras PR: {e}")
    try:
        plot_pr(y_test_binarized, y_prob_tflite, label_encoder.classes_, PR_PLOT_TFLITE_PATH, title='PR Curves (TFLite)')
    except Exception as e:
        log(f"Failed to generate TFLite PR: {e}")

except Exception as e:
    log(f"ROC/PR/AUC computation failed: {e}")

# ----------------- F1 Comparison Plot -----------------
try:
    plot_f1_bar(f1_keras, f1_tflite, label_encoder.classes_, F1_BAR_PATH)
except Exception as e:
    log(f"Failed to plot F1 comparison: {e}")

# ----------------- Save metrics summary JSON & CSV -----------------
metrics_summary['per_class_keras'] = per_class_df_keras.set_index('class').to_dict(orient='index')
metrics_summary['per_class_tflite'] = per_class_df_tflite.set_index('class').to_dict(orient='index')

with open(METRICS_JSON_PATH, 'w') as f:
    json.dump(metrics_summary, f, indent=4)
log(f"Saved metrics summary JSON to {METRICS_JSON_PATH}")

combined_table = per_class_df_keras.copy()
combined_table = combined_table.rename(columns={
    'precision': 'precision_keras', 'recall': 'recall_keras', 'f1': 'f1_keras', 'support': 'support_keras'
})
combined_table['precision_tflite'] = per_class_df_tflite['precision']
combined_table['recall_tflite'] = per_class_df_tflite['recall']
combined_table['f1_tflite'] = per_class_df_tflite['f1']
combined_table['support_tflite'] = per_class_df_tflite['support']
combined_table.to_csv(METRICS_CSV_PATH, index=False)
log(f"Saved combined metrics CSV to {METRICS_CSV_PATH}")

# ----------------- Accuracy drop and verdict -----------------
drop = keras_acc - tflite_acc
log(f"Accuracy drop after quantization: {drop:.4f} ({drop*100:.2f}%)")
if drop < 0.02:
    log("VERDICT: SUCCESS — small accuracy drop.")
else:
    log("VERDICT: WARNING — consider reviewing quantization or model architecture.")

# ----------------- Print created files -----------------
log("Final assets in results directory:")
all_assets = list(RESULTS_BASE_DIR.glob('**/*'))
for p in sorted(all_assets):
    if p.is_file():
        try:
            size = p.stat().st_size
            log(f" - {p.relative_to(RESULTS_BASE_DIR)}  ({size/1024:.1f} KB)")
        except Exception:
            log(f" - {p.relative_to(RESULTS_BASE_DIR)}")

log("Done.")
