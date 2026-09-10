import pandas as pd
from pathlib import Path

def prepare_statistical_datasets(clinical_csv_path, metrics_csv_path, qc_csv_path, output_dir):
    """
    Merge clinical, imaging metrics, and QC data into cross-sectional and longitudinal tables.

    Args:
        clinical_csv_path: Clinical CSV path.
        metrics_csv_path: Imaging metrics CSV path (long format).
        qc_csv_path: QC CSV path.
        output_dir: Output directory.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("[INFO] Reading raw data...")
    clinical_df = pd.read_csv(clinical_csv_path, dtype={'subject_id': 'string'})
    metrics_df = pd.read_csv(metrics_csv_path, dtype={'Subject_ID': 'string'})
    qc_df = pd.read_csv(qc_csv_path, dtype={'subject_id': 'string'})
    
    # ---------------------------------------------------------
    # 1. Normalize subject IDs
    # ---------------------------------------------------------
    clinical_df['subject_id'] = clinical_df['subject_id'].astype(str).str.strip()
    
    metrics_df.rename(columns={'Subject_ID': 'subject_id'}, inplace=True)
    metrics_df['subject_id'] = metrics_df['subject_id'].astype(str).str.strip()
    
    qc_df.rename(
        columns={
            'Subject_ID': 'subject_id',
            'run': 'Run',
            'mean_FD': 'Mean_FD',
        },
        inplace=True,
    )
    qc_df['subject_id'] = qc_df['subject_id'].astype(str).str.strip()

    required_qc_columns = {'subject_id', 'Run', 'Mean_FD', 'included'}
    missing_qc_columns = required_qc_columns - set(qc_df.columns)
    if missing_qc_columns:
        raise ValueError(
            "QC file is missing required columns: "
            + ", ".join(sorted(missing_qc_columns))
        )

    included_status = qc_df['included'].astype(str).str.strip().str.lower()
    if not included_status.isin({'yes', 'no'}).all():
        raise ValueError("QC column 'included' must contain only Yes or No.")
    qc_df = qc_df[included_status.eq('yes')].copy()

    if qc_df[['subject_id', 'Run']].duplicated().any():
        raise ValueError("QC file contains duplicate subject_id/Run rows.")
    
    # ---------------------------------------------------------
    # 2. Pivot metrics from long to wide format
    # ---------------------------------------------------------
    print("[INFO] Processing and pivoting metrics data...")
    # Fill missing feature names (e.g., transition_frequency without Feature)
    metrics_df['Feature'] = metrics_df['Feature'].fillna('')
    
    # Combine Metric and Feature into a column name (e.g., counts_CAP-1)
    def make_metric_col_name(row):
        m = str(row['Metric']).strip()
        f = str(row['Feature']).strip()
        if f:
            return f"{m}_{f}"
        return m
    
    metrics_df['Metric_Feature'] = metrics_df.apply(make_metric_col_name, axis=1)
    
    # Pivot: each subject_id + Run is one row
    metrics_wide = metrics_df.pivot_table(
        index=['subject_id', 'Run'],
        columns='Metric_Feature',
        values='Value'
    ).reset_index()
    
    # Preserve Group mapping if present
    group_map = metrics_df.drop_duplicates(subset=['subject_id', 'Group'])[['subject_id', 'Group']]
    metrics_wide = pd.merge(metrics_wide, group_map, on='subject_id', how='left')

    # ---------------------------------------------------------
    # 3. Merge metrics, QC, and clinical tables
    # ---------------------------------------------------------
    print("[INFO] Merging metrics, QC, and clinical data...")
    # Merge metrics and QC
    merged_df = pd.merge(
        metrics_wide,
        qc_df[['subject_id', 'Run', 'Mean_FD']],
        on=['subject_id', 'Run'],
        how='inner',
        validate='one_to_one',
    )
    
    # Merge clinical data
    merged_all = pd.merge(merged_df, clinical_df, on='subject_id', how='inner')
    
    # Resolve duplicate column names after merge
    if 'Group_x' in merged_all.columns:
        merged_all.rename(columns={'Group_x': 'Group'}, inplace=True)
        merged_all.drop(columns=['Group_y'], inplace=True, errors='ignore')
        
    merged_all.to_csv(out_dir / "01_merged_all_runs.csv", index=False)
    print(f" -> Merged base table generated (total rows: {len(merged_all)})")

    # ---------------------------------------------------------
    # 4. Cross-sectional table: HC (run-0) vs PD (run-1 or run-0)
    # ---------------------------------------------------------
    print("[INFO] Generating cross-sectional table (HC vs PD-OFF)...")
    # Keep HC run-0 and PD run-0 or run-1 (baseline or OFF)
    cross_df = merged_all[
        ((merged_all['Group'] == 'HC') & (merged_all['Run'] == 'run-0')) |
        ((merged_all['Group'] == 'PD') & (merged_all['Run'].isin(['run-0', 'run-1'])))
    ].copy()
    
    # If a PD subject has both run-0 and run-1, keep run-1
    cross_df['_run_prio'] = cross_df['Run'].map({'run-1': 1, 'run-0': 2})
    cross_df = cross_df.sort_values(['subject_id', '_run_prio']).drop_duplicates('subject_id')
    cross_df.drop(columns=['_run_prio'], inplace=True)
    
    cross_df.to_csv(out_dir / "02_cross_sectional_base.csv", index=False)
    hc_n = (cross_df['Group'] == 'HC').sum()
    pd_n = (cross_df['Group'] == 'PD').sum()
    print(f" -> Cross-sectional table generated (HC: {hc_n}, PD: {pd_n})")

    # ---------------------------------------------------------
    # 5. Longitudinal paired table: PD run-1 (OFF) vs run-2 (ON)
    # ---------------------------------------------------------
    print("[INFO] Generating longitudinal paired wide table (PD OFF vs ON)...")
    pd_df = merged_all[merged_all['Group'] == 'PD'].copy()
    
    df_off = pd_df[pd_df['Run'] == 'run-1'].copy()
    df_on = pd_df[pd_df['Run'] == 'run-2'].copy()
    
    # Add suffixes to distinguish OFF and ON
    df_off = df_off.add_suffix('_OFF').rename(columns={'subject_id_OFF': 'subject_id'})
    df_on = df_on.add_suffix('_ON').rename(columns={'subject_id_ON': 'subject_id'})
    
    # Merge on subject_id
    long_wide = pd.merge(df_off, df_on, on='subject_id', how='inner')
    
    long_wide.to_csv(out_dir / "03_longitudinal_paired_wide.csv", index=False)
    print(f" -> Longitudinal paired table generated (subjects with both pre/post: {len(long_wide)})")

    # Return output paths
    return {
        "merged_all_csv": str(out_dir / "01_merged_all_runs.csv"),
        "cross_sectional_csv": str(out_dir / "02_cross_sectional_base.csv"),
        "longitudinal_wide_csv": str(out_dir / "03_longitudinal_paired_wide.csv")
    }

# ==========================================
# Example usage
# ==========================================
if __name__ == "__main__":
    # Configure local paths
    CLINICAL_CSV = "./data/clinical.csv"
    METRICS_CSV = "./data/metrics_long.csv"
    QC_CSV = "./outputs/clinical_statistics/Subject_FD.csv"
    OUTPUT_DIR = "./data/processed_datasets"
    
    # Run only if paths exist
    if Path(CLINICAL_CSV).exists() and Path(METRICS_CSV).exists() and Path(QC_CSV).exists():
        result = prepare_statistical_datasets(
            clinical_csv_path=CLINICAL_CSV,
            metrics_csv_path=METRICS_CSV,
            qc_csv_path=QC_CSV,
            output_dir=OUTPUT_DIR
        )
    print("\n[SUCCESS] All files prepared!")
    print(f"Merged all-runs table: {result['merged_all_csv']}")
    print(f"Cross-sectional table (for ANCOVA): {result['cross_sectional_csv']}")
    print(f"Longitudinal table (for LMM): {result['longitudinal_wide_csv']}")
    print("[WARNING] File paths may not match your local filesystem. Update the example paths as needed.")
