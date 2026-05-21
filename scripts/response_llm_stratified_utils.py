import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.multitest import fdrcorrection

# ==========================================
# Helper and cleaning functions
# ==========================================
def _safe_tag(name):
    return str(name).replace(" ", "_").replace("/", "_").replace("-", "_").replace(".", "_to_").replace(":", "_")

def _get_metric_prefix(feature_name):
    clean_name = str(feature_name).replace('Delta_', '')
    if 'transition_probability' in clean_name:
        return 'transition_probability'
    if 'transition_frequency' in clean_name:
        return 'transition_frequency'
    return clean_name.split('_')[0]

def _safe_pct_change(off_values, on_values):
    off_arr = pd.to_numeric(off_values, errors="coerce")
    on_arr = pd.to_numeric(on_values, errors="coerce")
    delta = off_arr - on_arr
    denom = off_arr.abs()
    pct = np.where(denom > 1e-12, delta / denom * 100.0, np.nan)
    return delta, pct

def _add_metricwise_fdr(df, p_col, out_col, group_cols=["Metric_Group"]):
    """
    Apply BH-FDR per group (metric family, gender, etc.).
    """
    if p_col not in df.columns:
        return
    if 'Metric_Group' not in df.columns:
        df['Metric_Group'] = df['feature_base'].apply(_get_metric_prefix)
        
    df[out_col] = np.nan
    # Keep only grouping columns that exist in the data
    actual_groups = [c for c in group_cols if c in df.columns]
    
    for name, idx in df.groupby(actual_groups).groups.items():
        pvals = pd.to_numeric(df.loc[idx, p_col], errors="coerce").to_numpy(dtype=float)
        valid_mask = ~np.isnan(pvals)
        if valid_mask.sum() > 0:
            _, corr_p = fdrcorrection(pvals[valid_mask], alpha=0.05, method='indep')
            df.loc[idx[valid_mask], out_col] = corr_p

# ==========================================
# Core LMM helpers
# ==========================================
def _prepare_feature_long_for_three_way(df_wide, metric_base_col):
    off_col = f"{metric_base_col}_OFF"
    on_col = f"{metric_base_col}_ON"
    
    need_cols = [
        'subject_id', 'Response_Group', 'PctChange_UPDRS',
        off_col, on_col, 'Gender_OFF', 'Age_OFF', 'Years of Education_OFF',
        'Mean_FD_OFF', 'Mean_FD_ON'
    ]
    
    missing = [c for c in need_cols if c not in df_wide.columns]
    if missing:
        return None
        
    base = df_wide[need_cols].copy()
    num_cols = [off_col, on_col, 'Gender_OFF', 'Age_OFF', 'Years of Education_OFF', 'Mean_FD_OFF', 'Mean_FD_ON']
    for c in num_cols:
        base[c] = pd.to_numeric(base[c], errors='coerce')
        
    base = base.dropna().copy()
    if base.empty or len(base) < 4:
        return None

    off_df = pd.DataFrame({
        "subject_id": base["subject_id"].astype(str),
        "Metric": base[off_col],
        "Condition_encoded": 1, 
        "Response_Group": base["Response_Group"],
        "Gender_i": base["Gender_OFF"],
        "Age_i": base["Age_OFF"],
        "Education_i": base["Years of Education_OFF"],
        "Mean_FD": base["Mean_FD_OFF"]
    })
    
    on_df = off_df.copy()
    on_df["Metric"] = base[on_col]
    on_df["Condition_encoded"] = 0 
    on_df["Mean_FD"] = base["Mean_FD_ON"]

    long_df = pd.concat([off_df, on_df], ignore_index=True)
    long_df["Responder_encoded"] = (long_df["Response_Group"] == "Responder").astype(int)
    
    long_df["Gender_centered"] = long_df["Gender_i"] - long_df["Gender_i"].mean()
    long_df["Age_centered"] = long_df["Age_i"] - long_df["Age_i"].mean()
    long_df["Years_of_Education_centered"] = long_df["Education_i"] - long_df["Education_i"].mean()
    long_df["Mean_FD_centered"] = long_df["Mean_FD"] - long_df["Mean_FD"].mean()

    long_df["Cond_x_Group"] = long_df["Condition_encoded"] * long_df["Responder_encoded"]
    long_df["Cond_x_Sex"] = long_df["Condition_encoded"] * long_df["Gender_centered"]
    long_df["Group_x_Sex"] = long_df["Responder_encoded"] * long_df["Gender_centered"]
    long_df["Cond_x_Group_x_Sex"] = long_df["Condition_encoded"] * long_df["Responder_encoded"] * long_df["Gender_centered"]

    if long_df["Responder_encoded"].nunique() < 2:
        return None

    return long_df


def _fit_three_way_lmm(long_df):
    """Step 1: Fit LMM with the three-way interaction."""
    y = long_df["Metric"]
    groups = long_df["subject_id"]

    fixed_cols = [
        "Condition_encoded", "Responder_encoded", "Gender_centered",
        "Cond_x_Group", "Cond_x_Sex", "Group_x_Sex", "Cond_x_Group_x_Sex",
        "Age_centered", "Years_of_Education_centered", "Mean_FD_centered"
    ]

    usable_cols = []
    for c in fixed_cols:
        if c in ["Condition_encoded", "Responder_encoded", "Gender_centered", 
                 "Cond_x_Group", "Cond_x_Sex", "Group_x_Sex", "Cond_x_Group_x_Sex"]:
            usable_cols.append(c)
        elif long_df[c].nunique(dropna=True) > 1:
            usable_cols.append(c)

    X = sm.add_constant(long_df[usable_cols], has_constant="add")
    if np.linalg.matrix_rank(X.values) < X.shape[1]:
        return None

    fit_attempts = [("lbfgs", False), ("lbfgs", True), ("powell", True)]
    fitted = None
    
    for method, reml in fit_attempts:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = MixedLM(endog=y, exog=X, groups=groups)
                fitted = model.fit(reml=reml, method=method, maxiter=500, disp=False)
            if fitted.converged and np.isfinite(fitted.params.get('Condition_encoded', np.nan)):
                break
        except Exception:
            fitted = None
            continue

    if fitted is None:
        return None

    beta_4 = fitted.fe_params.get("Cond_x_Group", np.nan)
    p_4 = fitted.pvalues.get("Cond_x_Group", np.nan)
    beta_7 = fitted.fe_params.get("Cond_x_Group_x_Sex", np.nan)
    p_7 = fitted.pvalues.get("Cond_x_Group_x_Sex", np.nan)

    fe_names = list(fitted.fe_params.index)
    cov_fe = fitted.cov_params().loc[fe_names, fe_names]

    def _contrast(beta_names_and_weights):
        lvec = np.zeros(len(fe_names), dtype=float)
        for pname, weight in beta_names_and_weights:
            if pname in fe_names:
                lvec[fe_names.index(pname)] = weight
        bvec = fitted.fe_params.values.astype(float)
        est = float(np.dot(lvec, bvec))
        var = float(np.dot(lvec, np.dot(cov_fe.values, lvec)))
        if not np.isfinite(var) or var <= 0:
            return np.nan, np.nan
        z_val = est / np.sqrt(var)
        p_val = float(2 * (1 - norm.cdf(abs(z_val))))
        return est, p_val

    beta_off_on_nonres, p_off_on_nonres = _contrast([("Condition_encoded", 1.0)])
    beta_off_on_res, p_off_on_res = _contrast([("Condition_encoded", 1.0), ("Cond_x_Group", 1.0)])
    beta_group_on, p_group_on = _contrast([("Responder_encoded", 1.0)])
    beta_group_off, p_group_off = _contrast([("Responder_encoded", 1.0), ("Cond_x_Group", 1.0)])

    return {
        "beta_2way_Cond_x_Group": beta_4,
        "p_2way_Cond_x_Group": p_4,
        "beta_3way_Cond_x_Group_x_Sex": beta_7,
        "p_3way_Cond_x_Group_x_Sex": p_7,
        "beta_off_on_nonres": beta_off_on_nonres,
        "p_off_on_nonres": p_off_on_nonres,
        "beta_off_on_res": beta_off_on_res,
        "p_off_on_res": p_off_on_res,
        "beta_group_on": beta_group_on,
        "p_group_on": p_group_on,
        "beta_group_off": beta_group_off,
        "p_group_off": p_group_off,
        "converged_3way": bool(getattr(fitted, "converged", True))
    }


def _fit_stratified_lmm(sub_df):
    """Step 2: Fit stratified two-way interaction LMM and subgroup contrasts."""
    y = sub_df["Metric"]
    groups = sub_df["subject_id"]

    fixed_cols = [
        "Condition_encoded", "Responder_encoded", "Cond_x_Group",
        "Age_centered", "Years_of_Education_centered", "Mean_FD_centered"
    ]

    usable_cols = []
    for c in fixed_cols:
        if c in ["Condition_encoded", "Responder_encoded", "Cond_x_Group"]:
            usable_cols.append(c)
        elif sub_df[c].nunique(dropna=True) > 1:
            usable_cols.append(c)

    X = sm.add_constant(sub_df[usable_cols], has_constant="add")
    if np.linalg.matrix_rank(X.values) < X.shape[1]:
        return None

    fit_attempts = [("lbfgs", False), ("lbfgs", True), ("powell", True)]
    fitted = None
    
    for method, reml in fit_attempts:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = MixedLM(endog=y, exog=X, groups=groups)
                fitted = model.fit(reml=reml, method=method, maxiter=500, disp=False)
            if fitted.converged:
                break
        except Exception:
            fitted = None
            continue

    if fitted is None:
        return None

    beta_inter = fitted.fe_params.get("Cond_x_Group", np.nan)
    p_inter = fitted.pvalues.get("Cond_x_Group", np.nan)

    # Extract subgroup-specific contrasts
    fe_names = list(fitted.fe_params.index)
    cov_fe = fitted.cov_params().loc[fe_names, fe_names]

    def _contrast(beta_names_and_weights):
        lvec = np.zeros(len(fe_names), dtype=float)
        for pname, weight in beta_names_and_weights:
            if pname in fe_names:
                lvec[fe_names.index(pname)] = weight
        bvec = fitted.fe_params.values.astype(float)
        est = float(np.dot(lvec, bvec))
        var = float(np.dot(lvec, np.dot(cov_fe.values, lvec)))
        if not np.isfinite(var) or var <= 0:
            return np.nan, np.nan
        z_val = est / np.sqrt(var)
        p_val = float(2 * (1 - norm.cdf(abs(z_val))))
        return est, p_val

    beta_off_on_nonres, p_off_on_nonres = _contrast([("Condition_encoded", 1.0)])
    beta_off_on_res, p_off_on_res = _contrast([("Condition_encoded", 1.0), ("Cond_x_Group", 1.0)])
    beta_group_on, p_group_on = _contrast([("Responder_encoded", 1.0)])
    beta_group_off, p_group_off = _contrast([("Responder_encoded", 1.0), ("Cond_x_Group", 1.0)])

    return {
        "beta_interaction": beta_inter,
        "p_interaction": p_inter,
        "beta_off_on_nonres": beta_off_on_nonres,
        "p_off_on_nonres": p_off_on_nonres,
        "beta_off_on_res": beta_off_on_res,
        "p_off_on_res": p_off_on_res,
        "beta_group_on": beta_group_on,
        "p_group_on": p_group_on,
        "beta_group_off": beta_group_off,
        "p_group_off": p_group_off,
        "converged": bool(getattr(fitted, "converged", True))
    }

# ==========================================
# Visualization
# ==========================================
def _plot_stratified_interaction(long_df, feature_tag, out_dir, p_3way, genders_info):
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_df = long_df.copy()
    plot_df["Condition"] = np.where(plot_df["Condition_encoded"] == 1, "OFF", "ON")
    palette = {"Responder": "#d95f02", "Non-responder": "#1b9e77"}

    g_vals = sorted(plot_df["Gender_i"].unique())
    if len(g_vals) != 2:
        return 

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    
    sns.pointplot(
        data=plot_df, x="Condition", y="Metric", hue="Response_Group",
        dodge=0.15, markers=["o", "s"], linestyles=["-", "-"],
        errorbar=("ci", 95), palette=palette, ax=axes[0]
    )
    axes[0].set_title(f"Overall Cohort\n(3-way p={p_3way:.3g})")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    for i, g_val in enumerate(g_vals):
        ax = axes[i+1]
        sub_df = plot_df[plot_df["Gender_i"] == g_val]
        g_name = f"Gender {g_val}"
        
        sns.pointplot(
            data=sub_df, x="Condition", y="Metric", hue="Response_Group",
            dodge=0.15, markers=["o", "s"], linestyles=["-", "-"],
            errorbar=("ci", 95), palette=palette, ax=ax
        )
        
        p_sub = genders_info.get(g_val, {}).get("p_val", np.nan)
        ax.set_title(f"Sub-cohort: {g_name}\n(Interaction p={p_sub:.3g})")
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(f"Stratified Interaction Analysis: {feature_tag}", fontsize=14, y=1.05)
    fig.tight_layout()
    fig.savefig(out_dir / f"{_safe_tag(feature_tag)}_stratified_pointplot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# ==========================================
# Main pipeline
# ==========================================
def run_response_stratified_pipeline(data_dir, output_dir, response_threshold_pct=30.0):
    d_dir = Path(data_dir)
    out_dir = Path(output_dir)
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    long_csv = d_dir / "03_longitudinal_paired_wide.csv"
    if not long_csv.exists():
        print(f"[ERROR] Input file not found: {long_csv}")
        return

    df_long = pd.read_csv(long_csv)
    
    off_col = next((c for c in df_long.columns if 'UPDRS-III(OFF)' in c and c.endswith('_OFF')), None)
    on_col = next((c for c in df_long.columns if 'UPDRS-III(ON)' in c and c.endswith('_OFF')), None)
    if on_col is None:
        on_col = next((c for c in df_long.columns if 'UPDRS-III(OFF)' in c and c.endswith('_ON')), None)
        
    delta_u, pct_u = _safe_pct_change(df_long[off_col], df_long[on_col])
    df_long['Delta_UPDRS'] = delta_u
    df_long['PctChange_UPDRS'] = pct_u
    
    df_long['Response_Group'] = np.where(pct_u >= response_threshold_pct, 'Responder', 'Non-responder')

    ignore_patterns = ['subject_id', 'ID', 'Group', 'Run', 'Gender', 'Age', 'Education', 
                       'Mean_FD', 'Std_FD', 'Course', 'LEDD', 'H-Y', 'UPDRS', 'Responder', 'Condition']
                       
    long_off_cols = [c for c in df_long.columns if c.endswith('_OFF')]
    base_features = [
        c.replace('_OFF', '') for c in long_off_cols 
        if not any(ip in c for ip in ignore_patterns) and pd.api.types.is_numeric_dtype(df_long[c])
    ]

    print(f"[INFO] Starting three-way interaction and gender stratified analysis...")
    print(f"[DEBUG] Total candidate numeric features: {len(base_features)}")
    
    if len(base_features) == 0:
        return

    # Initialize result containers
    results_step1 = []
    results_step2 = []

    for i, feat in enumerate(base_features):
        verbose = True if i < 3 else False
        if verbose: print(f"\n--- Testing feature: {feat} ---")
        
        long_feat_df = _prepare_feature_long_for_three_way(df_long, feat)
        if long_feat_df is None: continue

        # Step 1: overall three-way model and main contrasts
        fit_3way = _fit_three_way_lmm(long_feat_df)
        if fit_3way is None: continue

        res_row_step1 = {
            "Feature": feat,
            "feature_base": feat,
            "N_total": len(long_feat_df),
            "beta_2way_Overall": fit_3way["beta_2way_Cond_x_Group"],
            "p_2way_Overall": fit_3way["p_2way_Cond_x_Group"],
            "beta_3way_Interaction": fit_3way["beta_3way_Cond_x_Group_x_Sex"],
            "p_3way_Interaction": fit_3way["p_3way_Cond_x_Group_x_Sex"],
            "beta_off_on_nonres": fit_3way["beta_off_on_nonres"],
            "p_off_on_nonres": fit_3way["p_off_on_nonres"],
            "beta_off_on_res": fit_3way["beta_off_on_res"],
            "p_off_on_res": fit_3way["p_off_on_res"],
            "beta_group_on": fit_3way["beta_group_on"],
            "p_group_on": fit_3way["p_group_on"],
            "beta_group_off": fit_3way["beta_group_off"],
            "p_group_off": fit_3way["p_group_off"]
        }
        results_step1.append(res_row_step1)

        # Step 2: gender-stratified subgroup analysis
        genders = long_feat_df["Gender_i"].unique()
        genders_info = {}
        
        for g_val in genders:
            sub_df = long_feat_df[long_feat_df["Gender_i"] == g_val]
            if sub_df["Responder_encoded"].nunique() < 2 or len(sub_df) < 10:
                continue
                
            fit_strat = _fit_stratified_lmm(sub_df)
            if fit_strat:
                res_row_step2 = {
                    "Feature": feat,
                    "feature_base": feat,
                    "Gender_Cohort": f"Gender_{g_val}",
                    "N_subcohort": len(sub_df),
                    "beta_interaction": fit_strat["beta_interaction"],
                    "p_interaction": fit_strat["p_interaction"],
                    "beta_off_on_nonres": fit_strat["beta_off_on_nonres"],
                    "p_off_on_nonres": fit_strat["p_off_on_nonres"],
                    "beta_off_on_res": fit_strat["beta_off_on_res"],
                    "p_off_on_res": fit_strat["p_off_on_res"],
                    "beta_group_on": fit_strat["beta_group_on"],
                    "p_group_on": fit_strat["p_group_on"],
                    "beta_group_off": fit_strat["beta_group_off"],
                    "p_group_off": fit_strat["p_group_off"]
                }
                results_step2.append(res_row_step2)
                genders_info[g_val] = {"p_val": fit_strat["p_interaction"]}

        if fit_3way["p_3way_Cond_x_Group_x_Sex"] > 0.05 and fit_3way["p_2way_Cond_x_Group"] < 0.05:
            _plot_stratified_interaction(long_feat_df, feat, plots_dir, fit_3way["p_3way_Cond_x_Group_x_Sex"], genders_info)

    if not results_step1: 
        print("\n[FATAL] All features were skipped, cannot generate result file.")
        return
    
    # -----------------------------
    # Post-processing: Step 1 (overall cohort)
    # -----------------------------
    df_step1 = pd.DataFrame(results_step1)
    
    _add_metricwise_fdr(df_step1, "p_2way_Overall", "p_2way_Overall_FDR", ["Metric_Group"])
    _add_metricwise_fdr(df_step1, "p_3way_Interaction", "p_3way_Interaction_FDR", ["Metric_Group"])
    _add_metricwise_fdr(df_step1, "p_off_on_nonres", "p_off_on_nonres_FDR", ["Metric_Group"])
    _add_metricwise_fdr(df_step1, "p_off_on_res", "p_off_on_res_FDR", ["Metric_Group"])
    _add_metricwise_fdr(df_step1, "p_group_on", "p_group_on_FDR", ["Metric_Group"])
    _add_metricwise_fdr(df_step1, "p_group_off", "p_group_off_FDR", ["Metric_Group"])

    out_csv_step1 = out_dir / "01_overall_3way_lmm_summary.csv"
    df_step1.to_csv(out_csv_step1, index=False)
    print(f"[SUCCESS] Step 1 overall model saved -> {out_csv_step1.name}")

    # -----------------------------
    # Post-processing: Step 2 (gender stratified)
    # -----------------------------
    if results_step2:
        df_step2 = pd.DataFrame(results_step2)
        
        # FDR correction is grouped by metric type and gender to keep strata independent
        _add_metricwise_fdr(df_step2, "p_interaction", "p_interaction_FDR", ["Metric_Group", "Gender_Cohort"])
        _add_metricwise_fdr(df_step2, "p_off_on_nonres", "p_off_on_nonres_FDR", ["Metric_Group", "Gender_Cohort"])
        _add_metricwise_fdr(df_step2, "p_off_on_res", "p_off_on_res_FDR", ["Metric_Group", "Gender_Cohort"])
        _add_metricwise_fdr(df_step2, "p_group_on", "p_group_on_FDR", ["Metric_Group", "Gender_Cohort"])
        _add_metricwise_fdr(df_step2, "p_group_off", "p_group_off_FDR", ["Metric_Group", "Gender_Cohort"])

        # Group rows by feature for easier review
        df_step2 = df_step2.sort_values(by=["Feature", "Gender_Cohort"])
        out_csv_step2 = out_dir / "02_stratified_subcohort_lmm_summary.csv"
        df_step2.to_csv(out_csv_step2, index=False)
        print(f"[SUCCESS] Step 2 gender stratified model saved -> {out_csv_step2.name}")

if __name__ == "__main__":
    run_response_stratified_pipeline(
        data_dir="./data/processed_datasets",
        output_dir="./data/stratified_output",
        response_threshold_pct=30.0
    )