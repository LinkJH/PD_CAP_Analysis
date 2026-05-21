import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon, fisher_exact
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

clinical_path = 'data/merged_clinical_metrics.csv'
qc_path = 'outputs/PDNeuroCAPs/01_extraction/qc_report.csv'
output_path = 'outputs/clinical_statistics'

def normalize_id(s):
    """
    Normalize Subject_ID for robust matching while preserving group prefixes.
    """
    if pd.isna(s): return ""
    s = str(s).strip().lower()
    
    match = re.match(r'([a-z\-]*)0*(\d+)', s)
    if match:
        prefix = match.group(1).replace('-', '')
        number = match.group(2)
        return f"{prefix}{number}"
    
    return s

def get_mean_sd(df, col):
    """Compute mean ± SD for continuous variables."""
    s = pd.to_numeric(df[col], errors='coerce').dropna()
    if len(s) == 0: return "-"
    return f"{s.mean():.2f} ± {s.std():.2f}"

def get_gender_str(df, col):
    """Compute gender ratio (Male/Female), 1=Male, 2=Female."""
    s = pd.to_numeric(df[col], errors='coerce').dropna()
    m = (s == 1).sum()
    f = (s == 2).sum()
    if m == 0 and f == 0: return "-"
    return f"{m}/{f}"

def format_pval(p):
    """Format p-values."""
    if pd.isna(p) or p == "": return "-"
    if p < 0.001: return "<0.001"
    return f"{p:.3f}"

def calc_p_ind(df1, df2, col, is_gender=False):
    """Independent-samples test."""
    s1 = pd.to_numeric(df1[col], errors='coerce').dropna()
    s2 = pd.to_numeric(df2[col], errors='coerce').dropna()
    if len(s1) == 0 or len(s2) == 0: return np.nan
    
    if is_gender:
        m1, f1 = (s1 == 1).sum(), (s1 == 2).sum()
        m2, f2 = (s2 == 1).sum(), (s2 == 2).sum()
        obs = [[m1, f1], [m2, f2]]
        try:
            _, p = fisher_exact(obs)
            return p
        except:
            return np.nan
    else:
        if len(set(s1)) == 1 and len(set(s2)) == 1 and s1.iloc[0] == s2.iloc[0]:
            return 1.0 
        try:
            _, p = mannwhitneyu(s1, s2, alternative='two-sided')
            return p
        except:
            return np.nan

def calc_p_paired(df, col_off, col_on):
    """Paired-sample Wilcoxon signed-rank test (OFF vs ON)."""
    tmp = df[[col_off, col_on]].apply(pd.to_numeric, errors='coerce').dropna()
    if len(tmp) < 2: return np.nan
    if (tmp[col_off] == tmp[col_on]).all(): return 1.0
    try:
        _, p = wilcoxon(tmp[col_off], tmp[col_on], alternative='two-sided')
        return p
    except:
        return np.nan

def main():
    if not os.path.exists(clinical_path) or not os.path.exists(qc_path):
        print("Error: Data files not found. Please check the file paths!")
        return

    df_clin = pd.read_csv(clinical_path)
    df_qc = pd.read_csv(qc_path)

    # Filter subjects based on QC
    qc_ids = df_qc['Subject_ID'].astype(str).str.strip().unique()
    qc_norm = [normalize_id(x) for x in qc_ids if normalize_id(x)]
    
    df_clin['norm_id'] = df_clin['subject_id'].apply(normalize_id)
    df_valid = df_clin[df_clin['norm_id'].isin(qc_norm)].copy()
    print(f"Read {len(df_clin)} clinical records, {len(qc_ids)} QC records, matched {len(df_valid)} records.")

    # LEDD handling: flag zeros and treat them as missing
    if 'LEDD(mg)' in df_valid.columns:
        ledd_numeric = pd.to_numeric(df_valid['LEDD(mg)'], errors='coerce')
        df_valid['LEDD_is_zero'] = (ledd_numeric == 0)  # flag whether LEDD is zero
        df_valid['LEDD(mg)'] = ledd_numeric
        df_valid.loc[df_valid['LEDD_is_zero'], 'LEDD(mg)'] = np.nan

    # Split groups
    df_pd = df_valid[df_valid['Group'] == 'PD'].copy()
    df_hc = df_valid[df_valid['Group'] == 'HC'].copy()

    # Select PD patients with ON data
    df_pd['UPDRS-III(ON)_num'] = pd.to_numeric(df_pd['UPDRS-III(ON)'], errors='coerce')
    df_offon = df_pd[df_pd['UPDRS-III(ON)_num'].notna()].copy()

    # Responder split
    updrs_off = pd.to_numeric(df_offon['UPDRS-III(OFF)'], errors='coerce')
    updrs_on = pd.to_numeric(df_offon['UPDRS-III(ON)'], errors='coerce')
    reduction = np.where(updrs_off == 0, 0, (updrs_off - updrs_on) / updrs_off)
    resp_mask = reduction >= 0.30
    df_resp = df_offon[resp_mask].copy()
    df_non = df_offon[~resp_mask].copy()

    # Plot LEDD raincloud: responders vs non-responders
    # Collect and clean valid data
    plot_df_resp = df_resp[['subject_id', 'LEDD(mg)']].copy()
    plot_df_resp['Group'] = 'Responder'
    plot_df_non = df_non[['subject_id', 'LEDD(mg)']].copy()
    plot_df_non['Group'] = 'Non-responder'
    plot_df = pd.concat([plot_df_resp, plot_df_non]).dropna(subset=['LEDD(mg)'])

    # Set plot style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 6))

    # 1. Violin layer (cloud) with higher transparency and no inner lines
    sns.violinplot(
        x="Group", y="LEDD(mg)", data=plot_df, 
        inner=None, color="lightgray", alpha=0.4, zorder=1
    )
    # 2. Strip layer (rain) with jittered points to show distribution
    sns.stripplot(
        x="Group", y="LEDD(mg)", data=plot_df,
        alpha=0.6, jitter=0.2, size=5, palette="Set2", hue="Group", legend=False, zorder=2
    )
    # 3. Box layer (umbrella) with narrow boxes for IQR
    sns.boxplot(
        x="Group", y="LEDD(mg)", data=plot_df,
        width=0.15, color="white", fliersize=0, boxprops={'zorder': 3}, zorder=3
    )

    plt.title('LEDD(mg) Distribution: Responder vs Non-responder', fontsize=14)
    plt.ylabel('LEDD (mg)', fontsize=12)
    plt.xlabel('Group', fontsize=12)
    plt.tight_layout()
    
    # Save figure
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    plot_filepath = os.path.join(output_path, "LEDD_Raincloud_Plot.png")
    plt.savefig(plot_filepath, dpi=300)
    print(f"LEDD raincloud plot generated and saved to: {plot_filepath}\n")


    # Count subjects
    N_pd, N_hc, N_offon = len(df_pd), len(df_hc), len(df_offon)
    N_resp, N_non = len(df_resp), len(df_non)

    # Define table rows and display strategy
    rows_config_t1 = [
        ("Age(years)", "Age", "shared"),
        ("Gender(M/F)", "Gender", "gender"),
        ("Years of Education(years)", "Years of Education", "shared"),
        ("Course of Disease(years)", "Course of Disease", "pd_baseline"),
        ("LEDD(mg)", "LEDD(mg)", "pd_baseline"),
        ("H-Y", ("H-Y(OFF)", "H-Y(ON)"), "paired_metric"),
        ("UPDRS-III", ("UPDRS-III(OFF)", "UPDRS-III(ON)"), "paired_metric"),
    ]

    rows_config_t2 = [
        ("Age(years)", "Age", "shared"),
        ("Gender(M/F)", "Gender", "gender"),
        ("Years of Education(years)", "Years of Education", "shared"),
        ("Course of Disease(years)", "Course of Disease", "pd_baseline"),
        ("LEDD(mg)", "LEDD(mg)", "pd_baseline"),
        ("H-Y(OFF)", "H-Y(OFF)", "off_metric"),
        ("UPDRS-III(OFF)", "UPDRS-III(OFF)", "off_metric"),
        ("H-Y(ON)", "H-Y(ON)", "on_metric"),
        ("UPDRS-III(ON)", "UPDRS-III(ON)", "on_metric"),
    ]

    # Generate Table 1 (main)
    t1_data = []
    for disp, col, rtype in rows_config_t1:
        row = {"Features": disp}
        is_g = (rtype == "gender")
        format_func = get_gender_str if is_g else get_mean_sd

        if rtype in ["shared", "gender"]:
            row["PD"] = format_func(df_pd, col)
            row["HC"] = format_func(df_hc, col)
            row["p PD vs. HC"] = format_pval(calc_p_ind(df_pd, df_hc, col, is_gender=is_g))
            row["PD-OFF"] = format_func(df_offon, col)
            row["PD-ON"] = "-"
            row["p OFF vs. ON"] = "-"
        elif rtype == "pd_baseline":
            row["PD"] = format_func(df_pd, col)
            row["HC"] = "-"
            row["p PD vs. HC"] = "-"
            row["PD-OFF"] = format_func(df_offon, col)
            row["PD-ON"] = "-"
            row["p OFF vs. ON"] = "-"
        elif rtype == "paired_metric":
            col_off, col_on = col
            row["PD"] = format_func(df_pd, col_off)
            row["HC"] = "-"
            row["p PD vs. HC"] = "-"
            row["PD-OFF"] = format_func(df_offon, col_off)
            row["PD-ON"] = format_func(df_offon, col_on)
            row["p OFF vs. ON"] = format_pval(calc_p_paired(df_offon, col_off, col_on))

        t1_data.append(row)

    # Generate Table 2 (supplement)
    t2_data = []
    for disp, col, rtype in rows_config_t2:
        row = {"Features": disp}
        is_g = (rtype == "gender")
        format_func = get_gender_str if is_g else get_mean_sd

        if rtype in ["shared", "gender", "pd_baseline", "off_metric"]:
            row["Responder-OFF"] = format_func(df_resp, col)
            row["Non-responder-OFF"] = format_func(df_non, col)
            row["p Responder vs. Non-responder(OFF)"] = format_pval(calc_p_ind(df_resp, df_non, col, is_gender=is_g))
            row["Responder-ON"] = "-"
            row["Non-responder-ON"] = "-"
            row["p Responder vs. Non-responder(ON)"] = "-"
        elif rtype == "on_metric":
            row["Responder-OFF"] = "-"
            row["Non-responder-OFF"] = "-"
            row["p Responder vs. Non-responder(OFF)"] = "-"
            row["Responder-ON"] = format_func(df_resp, col)
            row["Non-responder-ON"] = format_func(df_non, col)
            row["p Responder vs. Non-responder(ON)"] = format_pval(calc_p_ind(df_resp, df_non, col, is_gender=is_g))

        t2_data.append(row)

    # Rename headers and save
    t1_df = pd.DataFrame(t1_data)
    col_mapping_1 = {
        "Features": "Features",
        "PD": f"PD(N={N_pd})",
        "HC": f"HC(N={N_hc})",
        "p PD vs. HC": "p PD vs. HC",
        "PD-OFF": f"PD-OFF(N={N_offon})",
        "PD-ON": f"PD-ON(N={N_offon})",
        "p OFF vs. ON": "p OFF vs. ON"
    }
    t1_df.rename(columns=col_mapping_1, inplace=True)
    t1_df.to_csv(os.path.join(output_path, "Table1_Main_Clinical_Metrics.csv"), index=False, encoding="utf-8-sig")

    t2_df = pd.DataFrame(t2_data)
    col_mapping_2 = {
        "Features": "Features",
        "Responder-OFF": f"Responder-OFF(N={N_resp})",
        "Non-responder-OFF": f"Non-responder-OFF(N={N_non})",
        "p Responder vs. Non-responder(OFF)": "p Responder vs. Non-responder(OFF)",
        "Responder-ON": f"Responder-ON(N={N_resp})",
        "Non-responder-ON": f"Non-responder-ON(N={N_non})",
        "p Responder vs. Non-responder(ON)": "p Responder vs. Non-responder(ON)"
    }
    t2_df.rename(columns=col_mapping_2, inplace=True)
    t2_df.to_csv(os.path.join(output_path, "Table2_Responder_Metrics.csv"), index=False, encoding="utf-8-sig")

    print("Computation complete. Table1 and Table2 exported successfully.")

if __name__ == "__main__":
    main()