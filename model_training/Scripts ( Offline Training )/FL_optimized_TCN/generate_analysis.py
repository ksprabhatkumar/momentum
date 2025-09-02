import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# ==============================================================================
# 1. SETUP: Define paths and configurations
# ==============================================================================
print("--- [1/4] Setting up analysis configuration ---")

# --- Paths ---
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / ".." / "HAR_Momentum" / "data"
ANALYSIS_RESULTS_DIR = SCRIPT_DIR / "analysis_results"
ANALYSIS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
print(f"Results will be saved to: {ANALYSIS_RESULTS_DIR}")

# --- File and Activity Mappings ---
FILE_1_PATH = DATA_DIR / 'resampled_normalized_phone_data.csv'
FILE_1_ACTIVITIES = ['B', 'D', 'E']
FILE_1_NAME = "Resampled Phone Data"

FILE_2_PATH = DATA_DIR / 'combined_collected_data.csv'
FILE_2_ACTIVITIES = ['A', 'C']
FILE_2_NAME = "Combined Collected Data"

SAMPLING_RATE_HZ = 20.0

# ==============================================================================
# 2. HELPER FUNCTIONS for Analysis and Visualization
# ==============================================================================

def load_and_filter_data(file_path: Path, activities_to_keep: list, file_name: str) -> pd.DataFrame | None:
    """Loads a CSV, filters it for specific activities, and reports the process."""
    print("\n" + "="*80)
    print(f"Processing: {file_name} ({file_path.name})")
    print("="*80)

    if not file_path.exists():
        print(f"❌ ERROR: File not found at {file_path}")
        return None

    try:
        df = pd.read_csv(file_path)
        print(f"Loaded successfully. Original shape: {df.shape}")
        
        filtered_df = df[df['activity'].isin(activities_to_keep)].copy()
        print(f"Filtered for activities {activities_to_keep}. New shape: {filtered_df.shape}")
        
        if filtered_df.empty:
            print("⚠️ WARNING: No data remained after filtering.")
            return None
            
        return filtered_df
    except Exception as e:
        print(f"❌ ERROR: Could not read or process the file. {e}")
        return None


def generate_analysis_report(df: pd.DataFrame, df_name: str):
    """Prints a detailed text report about the dataframe's contents."""
    print(f"\n--- Detailed Analysis for: {df_name} ---")

    # 1. Class Distribution
    print("\n[Activity Distribution]")
    activity_counts = df['activity'].value_counts()
    activity_percentages = df['activity'].value_counts(normalize=True) * 100
    for activity, count in activity_counts.items():
        print(f"  - Activity '{activity}': {count} readings ({activity_percentages[activity]:.2f}%)")

    # 2. Subject Analysis
    print("\n[Subject Distribution]")
    unique_subjects = df['subject'].unique()
    print(f"  - Total unique subjects: {len(unique_subjects)}")
    print(f"  - Subject IDs: {sorted(unique_subjects)}")

    # 3. Subject-Activity Breakdown
    print("\n[Subject-Activity Breakdown (Readings per Subject/Activity)]")
    subject_activity_counts = df.groupby(['subject', 'activity']).size().unstack(fill_value=0)
    print(subject_activity_counts.to_string())

    # 4. Estimated Recording Duration
    print(f"\n[Estimated Recording Duration in Minutes (assuming {SAMPLING_RATE_HZ} Hz)]") # CHANGED HEADER
    subject_activity_duration_sec = subject_activity_counts / SAMPLING_RATE_HZ
    subject_activity_duration_min = subject_activity_duration_sec / 60
    # --- THIS IS THE CORRECTED LINE ---
    print(subject_activity_duration_min.round(2).to_string())


def create_visualizations(df: pd.DataFrame, df_name: str, output_dir: Path):
    """Generates and saves plots for the dataframe."""
    print(f"\n--- Generating visualizations for: {df_name} ---")

    safe_df_name = df_name.lower().replace(' ', '_')

    # Plot 1: Activity Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='activity', data=df, order=sorted(df['activity'].unique()), palette='viridis')
    plt.title(f'Activity Distribution in {df_name}')
    plt.xlabel('Activity Class')
    plt.ylabel('Number of Sensor Readings')
    plt.tight_layout()
    plot_path = output_dir / f"{safe_df_name}_activity_distribution.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"  ✓ Saved activity distribution plot to {plot_path.name}")

    # Plot 2: Data Contribution per Subject
    plt.figure(figsize=(12, 8))
    sns.countplot(y='subject', data=df, order=df['subject'].value_counts().index, palette='plasma')
    plt.title(f'Data Contribution per Subject in {df_name}')
    plt.xlabel('Number of Sensor Readings')
    plt.ylabel('Subject ID')
    plt.tight_layout()
    plot_path = output_dir / f"{safe_df_name}_subject_contribution.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"  ✓ Saved subject contribution plot to {plot_path.name}")

    # Plot 3: Subject vs. Activity Heatmap
    crosstab = pd.crosstab(df['subject'], df['activity'])
    plt.figure(figsize=(10, 8))
    sns.heatmap(crosstab, annot=True, fmt='d', cmap='YlGnBu', linewidths=.5)
    plt.title(f'Subject vs. Activity Matrix in {df_name} (Readings Count)')
    plt.tight_layout()
    plot_path = output_dir / f"{safe_df_name}_subject_activity_heatmap.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"  ✓ Saved subject-activity heatmap to {plot_path.name}")


# ==============================================================================
# 3. MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    print("--- [2/4] Starting analysis for the first dataset ---")
    df1_filtered = load_and_filter_data(FILE_1_PATH, FILE_1_ACTIVITIES, FILE_1_NAME)
    if df1_filtered is not None:
        generate_analysis_report(df1_filtered, FILE_1_NAME)
        create_visualizations(df1_filtered, FILE_1_NAME, ANALYSIS_RESULTS_DIR)

    print("\n--- [3/4] Starting analysis for the second dataset ---")
    df2_filtered = load_and_filter_data(FILE_2_PATH, FILE_2_ACTIVITIES, FILE_2_NAME)
    if df2_filtered is not None:
        generate_analysis_report(df2_filtered, FILE_2_NAME)
        create_visualizations(df2_filtered, FILE_2_NAME, ANALYSIS_RESULTS_DIR)
        
    print("\n--- [4/4] Analysis complete. Check the console output and the 'analysis_results' directory. ---")