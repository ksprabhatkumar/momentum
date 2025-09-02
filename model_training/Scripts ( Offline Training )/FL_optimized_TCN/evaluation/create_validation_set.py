# evaluation/create_validation_set.py
# Creates a hold-out validation set from the original CSV data.

import numpy as np
import pandas as pd
import json
from pathlib import Path
import random

# --- Configuration ---
print("--- [1/3] Setting up configuration ---")
RANDOM_SEED = 123  # Use a different seed from training for a different random sample
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# --- Path Setup ---
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / ".." / "data"
VALIDATION_DATA_DIR = SCRIPT_DIR / ".." / "validation_data"
VALIDATION_DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- Data Files (same as training script) ---
INPUT_CSV_1 = DATA_DIR / 'resampled_normalized_phone_data.csv'
INPUT_CSV_2 = DATA_DIR / 'combined_collected_data.csv'
ACTIVITIES_FROM_FILE1 = ['B', 'D', 'E']
ACTIVITIES_FROM_FILE2 = ['A', 'C']

# --- Windowing Parameters (must match training script) ---
WINDOW_SIZE = 60
STRIDE = 15
NUM_SAMPLES_TO_CREATE = 500 # How many random windows to save for validation

# --- 2. Data Loading and Windowing ---
print("--- [2/3] Loading data and creating all possible windows ---")

def create_windows(df, window_size, stride):
    """Replicates the windowing logic from the training script."""
    windows, labels = [], []
    required_cols = ['x_accel', 'y_accel', 'z_accel', 'x_gyro', 'y_gyro', 'z_gyro']
    for _, group_df in df.groupby(['subject', 'activity']):
        data_values = group_df.sort_values('timestamp')[required_cols].values
        if len(data_values) < window_size: continue
        for start in range(0, len(data_values) - window_size + 1, stride):
            windows.append(data_values[start : start + window_size].tolist()) # Convert to list for JSON
            labels.append(group_df["activity"].iloc[0])
    return windows, labels

df1 = pd.read_csv(INPUT_CSV_1)[lambda x: x['activity'].isin(ACTIVITIES_FROM_FILE1)]
df2 = pd.read_csv(INPUT_CSV_2)[lambda x: x['activity'].isin(ACTIVITIES_FROM_FILE2)]
combined_df = pd.concat([df1, df2], ignore_index=True)

all_windows, all_labels = create_windows(combined_df, WINDOW_SIZE, STRIDE)
print(f"Generated a pool of {len(all_windows)} total windows.")

# --- 3. Sampling and Saving the Validation Set ---
print(f"--- [3/3] Sampling {NUM_SAMPLES_TO_CREATE} windows and saving to JSON ---")

# Combine into a list of dictionaries
combined_data = list(zip(all_windows, all_labels))
random.shuffle(combined_data)

# Take a random sample
validation_sample = combined_data[:NUM_SAMPLES_TO_CREATE]

# Convert to the desired JSON format { "window": [...], "label": "A" }
validation_json_data = [
    {"window": window, "label": label} for window, label in validation_sample
]

# Save to file
output_path = VALIDATION_DATA_DIR / "validation_windows.json"
with open(output_path, 'w') as f:
    json.dump(validation_json_data, f, indent=2)

print(f"\n✅ Success! Saved {len(validation_json_data)} validation windows to:\n{output_path}")