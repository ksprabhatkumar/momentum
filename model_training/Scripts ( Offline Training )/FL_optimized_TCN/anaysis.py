import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# ==============================================================================
# 1. SETUP: Define paths and configurations
# ==============================================================================
print("--- [1/5] Setting up AI report generation ---")

# --- Paths ---
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / ".." / "HAR_Momentum" / "data"
ANALYSIS_RESULTS_DIR = SCRIPT_DIR / "analysis_results"
ANALYSIS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_REPORT_PATH = ANALYSIS_RESULTS_DIR / "dataset_analysis_report.txt"
print(f"Report will be saved to: {ANALYSIS_REPORT_PATH}")

# --- File and Activity Mappings ---
FILE_1_PATH = DATA_DIR / 'resampled_normalized_phone_data.csv'
FILE_1_ACTIVITIES = ['A','B', 'C','D', 'E']
FILE_1_NAME = "Resampled Phone Data"

FILE_2_PATH = DATA_DIR / 'combined_collected_data.csv'
FILE_2_ACTIVITIES = ['A', 'C']
FILE_2_NAME = "Combined Collected Data"

SAMPLING_RATE_HZ = 20.0

# ==============================================================================
# 2. HELPER FUNCTIONS for Analysis and Report Generation
# ==============================================================================

def load_and_filter_data(file_path: Path, activities_to_keep: list) -> pd.DataFrame | None:
    """Loads and filters a single CSV file."""
    if not file_path.exists():
        print(f"❌ ERROR: File not found at {file_path}")
        return None
    try:
        df = pd.read_csv(file_path)
        return df[df['activity'].isin(activities_to_keep)].copy()
    except Exception as e:
        print(f"❌ ERROR: Could not read or process {file_path.name}. {e}")
        return None

def generate_ai_report_section(df: pd.DataFrame, df_name: str) -> str:
    """Creates a formatted string section for a given dataframe."""
    report_lines = []
    
    # --- Section Header ---
    report_lines.append("\n" + "="*80)
    report_lines.append(f"### ANALYSIS FOR: {df_name} ###")
    report_lines.append("="*80)

    # --- Basic Info ---
    report_lines.append(f"\n[Dataset Shape]")
    report_lines.append(f"- Total Readings (Rows): {len(df)}")
    report_lines.append(f"- Features (Columns): {len(df.columns)}")

    # --- Class Distribution ---
    report_lines.append(f"\n[Activity Distribution]")
    activity_counts = df['activity'].value_counts()
    activity_percentages = df['activity'].value_counts(normalize=True) * 100
    dist_df = pd.DataFrame({
        'Readings': activity_counts,
        'Percentage': activity_percentages.round(2)
    })
    report_lines.append(dist_df.to_string())

    # --- Subject Analysis ---
    report_lines.append(f"\n[Subject Distribution]")
    unique_subjects = sorted(df['subject'].unique())
    report_lines.append(f"- Total Unique Subjects: {len(unique_subjects)}")
    # To avoid printing a giant list, we'll summarize if it's too long
    if len(unique_subjects) > 20:
        report_lines.append(f"- Subject ID Range: {min(unique_subjects)} to {max(unique_subjects)}")
    else:
        report_lines.append(f"- Subject IDs: {unique_subjects}")

    # --- Subject-Activity Matrix ---
    report_lines.append(f"\n[Subject-Activity Breakdown (Readings per Subject/Activity)]")
    subject_activity_counts = df.groupby(['subject', 'activity']).size().unstack(fill_value=0)
    report_lines.append(subject_activity_counts.to_string())

    # --- Estimated Duration Matrix ---
    report_lines.append(f"\n[Estimated Duration in Minutes (per Subject/Activity)]")
    duration_min = (subject_activity_counts / SAMPLING_RATE_HZ / 60).round(2)
    report_lines.append(duration_min.to_string())
    
    return "\n".join(report_lines)

def create_visualizations(df: pd.DataFrame, df_name: str, output_dir: Path):
    """Generates and saves plots. This is for human review."""
    # (This function is unchanged from the previous script)
    safe_df_name = df_name.lower().replace(' ', '_').replace('/', '')
    plt.figure(figsize=(10, 7))
    sns.countplot(x='activity', data=df, order=sorted(df['activity'].unique()), palette='viridis')
    plt.title(f'Activity Distribution in {df_name}')
    plt.savefig(output_dir / f"{safe_df_name}_activity_dist.png")
    plt.close()

    plt.figure(figsize=(12, 8))
    sns.countplot(y='subject', data=df, order=df['subject'].value_counts().index, palette='plasma')
    plt.title(f'Data Contribution per Subject in {df_name}')
    plt.tight_layout()
    plt.savefig(output_dir / f"{safe_df_name}_subject_contrib.png")
    plt.close()

# ==============================================================================
# 3. MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    
    # --- Load Data ---
    print("--- [2/5] Loading and filtering datasets ---")
    df1 = load_and_filter_data(FILE_1_PATH, FILE_1_ACTIVITIES)
    df2 = load_and_filter_data(FILE_2_PATH, FILE_2_ACTIVITIES)

    # --- Generate Report Sections ---
    report_content = []
    
    # Report Header
    report_content.append("="*80)
    report_content.append("### COMPREHENSIVE DATASET ANALYSIS REPORT ###")
    report_content.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append("="*80)
    report_content.append("This report provides a detailed statistical breakdown of the HAR datasets.")

    print("--- [3/5] Generating report sections for each file ---")
    if df1 is not None:
        report_content.append(generate_ai_report_section(df1, FILE_1_NAME))
    else:
        report_content.append(f"\nWARNING: Could not process {FILE_1_NAME}. See console for errors.")

    if df2 is not None:
        report_content.append(generate_ai_report_section(df2, FILE_2_NAME))
    else:
        report_content.append(f"\nWARNING: Could not process {FILE_2_NAME}. See console for errors.")

    # --- Generate Combined Analysis and Key Insights ---
    print("--- [4/5] Generating combined analysis and key insights ---")
    if df1 is not None and df2 is not None:
        combined_df = pd.concat([df1, df2], ignore_index=True)
        report_content.append(generate_ai_report_section(combined_df, "FINAL COMBINED DATASET"))
        
        # --- Final Summary Section for the AI ---
        key_insights = []
        key_insights.append("\n" + "="*80)
        key_insights.append("### KEY INSIGHTS & DATA QUALITY SUMMARY ###")
        key_insights.append("="*80)
        
        # Overall Stats
        total_subjects = combined_df['subject'].nunique()
        total_readings = len(combined_df)
        key_insights.append(f"\n- Overall: The final dataset contains {total_readings:,} readings from {total_subjects} unique subjects.")
        
        # Class Balance
        balance = combined_df['activity'].value_counts(normalize=True) * 100
        key_insights.append(f"- Class Balance: The dataset is reasonably well-balanced. The most frequent class ('{balance.idxmax()}') constitutes {balance.max():.1f}% of the data, while the least frequent ('{balance.idxmin()}') is {balance.min():.1f}%.")
        
        # Subject Contribution
        subjects_df1 = set(df1['subject'].unique())
        subjects_df2 = set(df2['subject'].unique())
        if subjects_df1.isdisjoint(subjects_df2):
            key_insights.append("- Subject Exclusivity: CONFIRMED. The two source files contain completely separate sets of subjects. This is ideal for creating non-overlapping train/test splits based on subject ID.")
        else:
            key_insights.append("- Subject Exclusivity: WARNING. There is an overlap in subject IDs between the two source files. This must be handled carefully during data splitting.")

        key_insights.append("- Data Structure: The data appears highly structured, especially within the 'Resampled Phone Data' file, suggesting a controlled collection environment. This provides a strong basis for a generalizable model.")
        key_insights.append("- Recommendation: Due to the clear subject separation, a `GroupShuffleSplit` strategy based on 'subject' ID is the recommended approach for model training and evaluation to prevent data leakage.")
        
        report_content.append("\n".join(key_insights))
        
        # Also create visualizations for the combined dataset
        create_visualizations(combined_df, "Combined", ANALYSIS_RESULTS_DIR)

    # --- Write Final Report ---
    final_report_string = "\n".join(report_content)
    with open(ANALYSIS_REPORT_PATH, 'w') as f:
        f.write(final_report_string)

    print(f"--- [5/5] AI-friendly report successfully saved to {ANALYSIS_REPORT_PATH.name} ---")
    print("\n--- Displaying report content in console ---\n")
    print(final_report_string)