import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon, t
import re
import os

# ----------------- File path configuration -----------------
clinical_path = 'data/merged_clinical_metrics.csv'
qc_path = 'outputs/PDNeuroCAPs/01_extraction/qc_report.csv'
output_path = 'outputs/clinical_statistics'

# ----------------- Helper functions -----------------
def normalize_id(s):
    """Normalize Subject_ID for robust matching."""
    if pd.isna(s): return ""
    s = str(s).strip().lower()
    
    match = re.match(r'([a-z\-]*)0*(\d+)', s)
    if match:
        prefix = match.group(1).replace('-', '')
        number = match.group(2)
        return f"{prefix}{number}"
    return s

def format_pval(p):
    """Format p-values."""
    if pd.isna(p) or p == "": return "-"
    if p < 0.001: return "<0.001"
    return f"{p:.3f}"

def calc_p_ind(df1, df2, col):
    """Independent-samples test (Mann-Whitney U)."""
    if col not in df1.columns or col not in df2.columns: return np.nan
    s1 = pd.to_numeric(df1[col], errors='coerce').dropna()
    s2 = pd.to_numeric(df2[col], errors='coerce').dropna()
    if len(s1) == 0 or len(s2) == 0: return np.nan
    
    if len(set(s1)) == 1 and len(set(s2)) == 1 and s1.iloc[0] == s2.iloc[0]:
        return 1.0 
    try:
        _, p = mannwhitneyu(s1, s2, alternative='two-sided')
        return p
    except:
        return np.nan

def calc_p_paired(df, col_off, col_on):
    """Paired-sample Wilcoxon signed-rank test (OFF vs ON)."""
    if col_off not in df.columns or col_on not in df.columns: return np.nan
    tmp = df[[col_off, col_on]].apply(pd.to_numeric, errors='coerce').dropna()
    if len(tmp) < 2: return np.nan
    if (tmp[col_off] == tmp[col_on]).all(): return 1.0
    try:
        _, p = wilcoxon(tmp[col_off], tmp[col_on], alternative='two-sided')
        return p
    except:
        return np.nan

def get_stats_dict(group_name, df, col):
    """Extract group statistics: mean ± SD and 95% CI."""
    if col not in df.columns:
        return {"Group": f"{group_name}(N=0)", "Mean_FD": "-", "95% CI_low": "-", "95% CI_high": "-", "p": "-"}
        
    s = pd.to_numeric(df[col], errors='coerce').dropna()
    n = len(s)
    
    if n == 0:
        return {"Group": f"{group_name}(N=0)", "Mean_FD": "-", "95% CI_low": "-", "95% CI_high": "-", "p": "-"}
    
    mean_val = s.mean()
    sd_val = s.std()
    
    # Compute 95% CI
    if n > 1:
        se = sd_val / np.sqrt(n)
        ci_low, ci_high = t.interval(0.95, df=n-1, loc=mean_val, scale=se)
        ci_low_str = f"{ci_low:.4f}"
        ci_high_str = f"{ci_high:.4f}"
    else:
        ci_low_str = "-"
        ci_high_str = "-"
        
    return {
        "Group": f"{group_name}(N={n})",
        "Mean_FD": f"{mean_val:.4f} ± {sd_val:.4f}",
        "95% CI_low": ci_low_str,
        "95% CI_high": ci_high_str,
        "p": "-"
    }

def get_p_row(comp_name, p_val):
    """Build a p-value row."""
    return {
        "Group": comp_name,
        "Mean_FD": "-",
        "95% CI_low": "-",
        "95% CI_high": "-",
        "p": format_pval(p_val)
    }

# ----------------- Main logic -----------------
def main_qc_analysis():
    if not os.path.exists(clinical_path) or not os.path.exists(qc_path):
        print("Error: Data files not found. Please check the file paths!")
        return

    df_clin = pd.read_csv(clinical_path)
    df_qc = pd.read_csv(qc_path)

    # QC preprocessing and wide-table pivot (keep all runs)
    df_qc['norm_id'] = df_qc['Subject_ID'].apply(normalize_id)
    
    qc_pivot = df_qc.pivot(
        index='norm_id', 
        columns='Run', 
        values='Mean_FD'
    )
    # Rename columns: run-0 -> Mean_FD_run-0, run-1 -> Mean_FD_run-1, etc.
    qc_pivot.columns = [f"Mean_FD_{col}" for col in qc_pivot.columns]
    qc_pivot.reset_index(inplace=True)

    # Create a baseline FD column for PD vs HC
    # Prefer run-1 (PD OFF baseline); fallback to run-0 (HC/others single baseline)
    if 'Mean_FD_run-1' in qc_pivot.columns and 'Mean_FD_run-0' in qc_pivot.columns:
        qc_pivot['Mean_FD_baseline'] = qc_pivot['Mean_FD_run-1'].fillna(qc_pivot['Mean_FD_run-0'])
    elif 'Mean_FD_run-1' in qc_pivot.columns:
        qc_pivot['Mean_FD_baseline'] = qc_pivot['Mean_FD_run-1']
    elif 'Mean_FD_run-0' in qc_pivot.columns:
        qc_pivot['Mean_FD_baseline'] = qc_pivot['Mean_FD_run-0']
    else:
        qc_pivot['Mean_FD_baseline'] = np.nan

    # Clinical preprocessing and merge
    df_clin['norm_id'] = df_clin['subject_id'].apply(normalize_id)
    df_merged = pd.merge(df_clin, qc_pivot, on='norm_id', how='inner')
    
    # --- Grouping ---
    # HC group (baseline)
    df_hc = df_merged[df_merged['Group'] == 'HC'].copy()
    
    # PD overall (baseline)
    df_pd_all = df_merged[df_merged['Group'] == 'PD'].copy()

    # PD paired group (requires both run-1 and run-2)
    # Needed for PD-OFF vs PD-ON and responder analysis
    if 'Mean_FD_run-1' in df_merged.columns and 'Mean_FD_run-2' in df_merged.columns:
        df_pd_paired = df_pd_all.dropna(subset=['Mean_FD_run-1', 'Mean_FD_run-2']).copy()
    else:
        df_pd_paired = pd.DataFrame(columns=df_pd_all.columns)
        
    df_pd_paired['UPDRS-III(OFF)_num'] = pd.to_numeric(df_pd_paired['UPDRS-III(OFF)'], errors='coerce')
    df_pd_paired['UPDRS-III(ON)_num'] = pd.to_numeric(df_pd_paired['UPDRS-III(ON)'], errors='coerce')
    
    df_valid_pd = df_pd_paired.dropna(subset=['UPDRS-III(OFF)_num', 'UPDRS-III(ON)_num']).copy()
    
    # Responder vs non-responder split
    updrs_off = df_valid_pd['UPDRS-III(OFF)_num']
    updrs_on = df_valid_pd['UPDRS-III(ON)_num']
    reduction = np.where(updrs_off == 0, 0, (updrs_off - updrs_on) / updrs_off)
    
    resp_mask = reduction >= 0.30
    df_resp = df_valid_pd[resp_mask].copy()
    df_non = df_valid_pd[~resp_mask].copy()

    # Build the long-format statistics table
    table_data = []

    # [Block 1] PD vs HC (baseline data)
    table_data.append(get_stats_dict("PD", df_pd_all, 'Mean_FD_baseline'))
    table_data.append(get_stats_dict("HC", df_hc, 'Mean_FD_baseline'))
    table_data.append(get_p_row("PD vs. HC", calc_p_ind(df_pd_all, df_hc, 'Mean_FD_baseline')))

    # [Block 2] PD-OFF vs PD-ON (paired run-1 and run-2)
    table_data.append(get_stats_dict("PD-OFF", df_valid_pd, 'Mean_FD_run-1'))
    table_data.append(get_stats_dict("PD-ON", df_valid_pd, 'Mean_FD_run-2'))
    table_data.append(get_p_row("PD-OFF vs. PD-ON", calc_p_paired(df_valid_pd, 'Mean_FD_run-1', 'Mean_FD_run-2')))

    # [Block 3] Responder-OFF vs Non-responder-OFF
    table_data.append(get_stats_dict("Responder-OFF", df_resp, 'Mean_FD_run-1'))
    table_data.append(get_stats_dict("Non-responder-OFF", df_non, 'Mean_FD_run-1'))
    table_data.append(get_p_row("Responder-OFF vs. Non-responder-OFF", calc_p_ind(df_resp, df_non, 'Mean_FD_run-1')))

    # [Block 4] Responder-ON vs Non-responder-ON
    table_data.append(get_stats_dict("Responder-ON", df_resp, 'Mean_FD_run-2'))
    table_data.append(get_stats_dict("Non-responder-ON", df_non, 'Mean_FD_run-2'))
    table_data.append(get_p_row("Responder-ON vs. Non-responder-ON", calc_p_ind(df_resp, df_non, 'Mean_FD_run-2')))

    # Export and print
    res_df = pd.DataFrame(table_data)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    out_file = os.path.join(output_path, "Table3_QC_MeanFD_Long_Format.csv")
    res_df.to_csv(out_file, index=False, encoding="utf-8-sig")
    
    print(f"\nQC FD longitudinal metrics computed successfully! Exported: {out_file}")
    print("-" * 75)
    print(res_df.to_string(index=False))
    print("-" * 75)

if __name__ == "__main__":
    main_qc_analysis()