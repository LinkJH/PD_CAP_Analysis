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
    """Sanitize strings for filenames."""
    return str(name).replace(" ", "_").replace("/", "_").replace("-", "_").replace(".", "_to_").replace(":", "_")

def _get_metric_prefix(feature_name):
    """Extract metric family for independent FDR correction."""
    clean_name = str(feature_name).replace('Delta_', '')
    if 'transition_probability' in clean_name:
        return 'transition_probability'
    if 'transition_frequency' in clean_name:
        return 'transition_frequency'
    return clean_name.split('_')[0]

def _safe_pct_change(off_values, on_values):
    """Safely compute delta and percent change."""
    off_arr = pd.to_numeric(off_values, errors="coerce")
    on_arr = pd.to_numeric(on_values, errors="coerce")
    delta = off_arr - on_arr
    denom = off_arr.abs()
    pct = np.where(denom > 1e-12, delta / denom * 100.0, np.nan)
    return delta, pct

def _add_metricwise_fdr(df, p_col, out_col):
    """Apply BH-FDR per metric family."""
    df[out_col] = np.nan
    df['Metric_Group'] = df['feature_base'].apply(_get_metric_prefix)
    
    for metric_name, idx in df.groupby("Metric_Group").groups.items():
        pvals = pd.to_numeric(df.loc[idx, p_col], errors="coerce").to_numpy(dtype=float)
        valid_mask = ~np.isnan(pvals)
        if valid_mask.sum() > 0:
            _, corr_p = fdrcorrection(pvals[valid_mask], alpha=0.05, method='indep')
            df.loc[idx[valid_mask], out_col] = corr_p

# ==========================================
# Core LMM helpers
# ==========================================
def _prepare_feature_long_for_interaction(df_wide, metric_base_col):
    """Melt a wide feature into long format for LMM interaction and center covariates."""
    off_col = f"{metric_base_col}_OFF"
    on_col = f"{metric_base_col}_ON"
    
    if off_col not in df_wide.columns or on_col not in df_wide.columns:
        return None

    # Select required columns (target, response group, covariates)
    need_cols = [
        'subject_id', 'Response_Group', 'PctChange_UPDRS',
        off_col, on_col,
        'Gender_OFF', 'Age_OFF', 'Years of Education_OFF',
        'Mean_FD_OFF', 'Mean_FD_ON'
    ]
    
    missing = [c for c in need_cols if c not in df_wide.columns]
    if missing:
        return None
        
    base = df_wide[need_cols].copy()
    
    # Coerce to numeric
    num_cols = [off_col, on_col, 'Gender_OFF', 'Age_OFF', 'Years of Education_OFF', 'Mean_FD_OFF', 'Mean_FD_ON']
    for c in num_cols:
        base[c] = pd.to_numeric(base[c], errors='coerce')
        
    # Drop pairs with any missing values
    base = base.dropna().copy()
    if base.empty or len(base) < 4:
        return None

    # Build OFF-state rows
    off_df = pd.DataFrame({
        "subject_id": base["subject_id"].astype(str),
        "Metric": base[off_col],
        "Condition_encoded": 1,  # OFF = 1
        "Response_Group": base["Response_Group"],
        "Gender_i": base["Gender_OFF"],
        "Age_i": base["Age_OFF"],
        "Education_i": base["Years of Education_OFF"],
        "Mean_FD": base["Mean_FD_OFF"]
    })
    
    # Build ON-state rows
    on_df = off_df.copy()
    on_df["Metric"] = base[on_col]
    on_df["Condition_encoded"] = 0  # ON = 0
    on_df["Mean_FD"] = base["Mean_FD_ON"]

    # Concatenate long format
    long_df = pd.concat([off_df, on_df], ignore_index=True)
    
    # Encode responder (1=Responder, 0=Non-responder)
    long_df["Responder_encoded"] = (long_df["Response_Group"] == "Responder").astype(int)
    
    # Compute interaction term
    long_df["Interaction"] = long_df["Condition_encoded"] * long_df["Responder_encoded"]

    # Center covariates within available samples
    long_df["Gender_centered"] = long_df["Gender_i"] - long_df["Gender_i"].mean()
    long_df["Age_centered"] = long_df["Age_i"] - long_df["Age_i"].mean()
    long_df["Years_of_Education_centered"] = long_df["Education_i"] - long_df["Education_i"].mean()
    long_df["Mean_FD_centered"] = long_df["Mean_FD"] - long_df["Mean_FD"].mean()  # global centering

    # Require both response groups for interaction analysis
    if long_df["Responder_encoded"].nunique() < 2:
        return None

    return long_df


def _fit_interaction_lmm(long_df):
    """Fit LMM with interaction and extract contrasts."""
    y = long_df["Metric"]
    groups = long_df["subject_id"]

    # Define candidate fixed effects
    fixed_cols = [
        "Condition_encoded", "Responder_encoded", "Interaction",
        "Gender_centered", "Age_centered", "Years_of_Education_centered", "Mean_FD_centered"
    ]

    # Filter zero-variance covariates to avoid singular matrices
    usable_cols = []
    for c in fixed_cols:
        if c in ["Condition_encoded", "Responder_encoded", "Interaction"]:
            usable_cols.append(c)
            continue
        if long_df[c].nunique(dropna=True) > 1:
            usable_cols.append(c)

    X = sm.add_constant(long_df[usable_cols], has_constant="add")
    if np.linalg.matrix_rank(X.values) < X.shape[1]:
        return None

    # Retry with multiple optimizers
    fit_attempts = [("lbfgs", False), ("lbfgs", True), ("powell", True), ("cg", False)]
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

    # Compute custom contrasts
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
            return est, np.nan
        z_val = est / np.sqrt(var)
        p_val = float(2 * (1 - norm.cdf(abs(z_val))))
        return est, p_val

    # With OFF=1, ON=0:
    # Non-responder OFF-ON effect = beta_condition
    # Responder OFF-ON effect = beta_condition + beta_interaction
    # ON: responder vs non-responder = beta_responder
    # OFF: responder vs non-responder = beta_responder + beta_interaction
    beta_off_on_nonres, p_off_on_nonres = _contrast([("Condition_encoded", 1.0)])
    beta_off_on_res, p_off_on_res = _contrast([("Condition_encoded", 1.0), ("Interaction", 1.0)])
    beta_group_on, p_group_on = _contrast([("Responder_encoded", 1.0)])
    beta_group_off, p_group_off = _contrast([("Responder_encoded", 1.0), ("Interaction", 1.0)])
    beta_inter, p_inter = _contrast([("Interaction", 1.0)])

    out = {
        "beta_condition": beta_off_on_nonres,
        "p_condition": p_off_on_nonres,
        "beta_responder_vs_non_on": beta_group_on,
        "p_responder_vs_non_on": p_group_on,
        "beta_responder_vs_non_off": beta_group_off,
        "p_responder_vs_non_off": p_group_off,
        "beta_off_on_responder": beta_off_on_res,
        "p_off_on_responder": p_off_on_res,
        "beta_interaction": beta_inter,
        "p_interaction": p_inter,
        "converged": bool(getattr(fitted, "converged", True)),
        "lmm_spec": "+".join(usable_cols),
    }
    return out

# ==========================================
# Visualization
# ==========================================
def _plot_interaction_effects(long_df, feature_tag, out_dir, p_interaction_fdr):
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_df = long_df.copy()
    plot_df["Condition"] = np.where(plot_df["Condition_encoded"] == 1, "OFF", "ON")
    palette = {"Responder": "#d95f02", "Non-responder": "#1b9e77"}

    # Pointplot of interaction effects
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    sns.pointplot(
        data=plot_df, x="Condition", y="Metric", hue="Response_Group",
        dodge=0.15, markers=["o", "s"], linestyles=["-", "-"],
        errorbar=("ci", 95), palette=palette, ax=ax
    )
    ax.set_title(f"{feature_tag}\nInteraction $p_{{FDR}}$={p_interaction_fdr:.4g}")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / f"{_safe_tag(feature_tag)}_interaction_pointplot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Boxplot + stripplot
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    sns.boxplot(
        data=plot_df, x="Condition", y="Metric", hue="Response_Group",
        palette=palette, dodge=True, ax=ax, showfliers=False
    )
    sns.stripplot(
        data=plot_df, x="Condition", y="Metric", hue="Response_Group",
        palette=palette, dodge=True, alpha=0.55, size=4, ax=ax
    )
    # Fix legend warning by reusing de-duplicated handles
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[:2], labels[:2], title="Response Group", loc="best")
        
    ax.set_title(f"{feature_tag} (Condition vs Response Group)")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / f"{_safe_tag(feature_tag)}_interaction_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ==========================================
# Main pipeline
# ==========================================
def run_response_lmm_interaction_pipeline(data_dir, output_dir, response_threshold_pct=30.0):
    d_dir = Path(data_dir)
    out_dir = Path(output_dir)
    plots_dir = out_dir / "plots"
    sig_plots_dir = plots_dir / "significant_interactions"

    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    sig_plots_dir.mkdir(parents=True, exist_ok=True)
    
    long_csv = d_dir / "03_longitudinal_paired_wide.csv"
    if not long_csv.exists():
        print(f"[ERROR] Input file not found: {long_csv}")
        return

    df_long = pd.read_csv(long_csv)
    print(f"[INFO] Successfully read longitudinal wide table, total pairs: {len(df_long)}")

    # Define clinical response groups
    off_col = next((c for c in df_long.columns if 'UPDRS-III(OFF)' in c and c.endswith('_OFF')), None)
    on_col = next((c for c in df_long.columns if 'UPDRS-III(ON)' in c and c.endswith('_OFF')), None)
    if on_col is None:
        on_col = next((c for c in df_long.columns if 'UPDRS-III(OFF)' in c and c.endswith('_ON')), None)
        
    if not off_col or not on_col:
        print("[ERROR] Cannot find UPDRS-III OFF and ON columns in the wide table.")
        return
        
    delta_u, pct_u = _safe_pct_change(df_long[off_col], df_long[on_col])
    df_long['Delta_UPDRS'] = delta_u
    df_long['PctChange_UPDRS'] = pct_u
    df_long['Response_Group'] = np.where(df_long['PctChange_UPDRS'] >= response_threshold_pct, 'Responder', 'Non-responder')
    
    print(f"[INFO] Clinical response defined (threshold: {response_threshold_pct}%): Responder={sum(df_long['Response_Group']=='Responder')}, Non-responder={sum(df_long['Response_Group']=='Non-responder')}")

    # Identify all numeric features with _OFF suffix
    ignore_patterns = ['subject_id', 'ID', 'Group', 'Run', 'Gender', 'Age', 'Education', 
                       'Mean_FD', 'Std_FD', 'Course', 'LEDD', 'H-Y', 'UPDRS', 'Responder', 'Condition']
                       
    long_off_cols = [c for c in df_long.columns if c.endswith('_OFF')]
    base_features = [
        c.replace('_OFF', '') for c in long_off_cols 
        if not any(ip in c for ip in ignore_patterns) and pd.api.types.is_numeric_dtype(df_long[c])
    ]

    print(f"[INFO] Running interaction effect analysis on {len(base_features)} imaging features...")
    results = []
    long_df_cache = {}

    # Fit interaction LMM for each feature
    for feat in base_features:
        long_feat_df = _prepare_feature_long_for_interaction(df_long, feat)
        if long_feat_df is None:
            continue

        fit_out = _fit_interaction_lmm(long_feat_df)
        if fit_out is None:
            continue

        feature_tag = feat
        results.append({
            "Feature": feature_tag,
            "feature_base": feat,
            "N_subjects": int(long_feat_df["subject_id"].nunique()),
            "N_rows": int(len(long_feat_df)),
            "N_responder_rows": int((long_feat_df["Response_Group"] == "Responder").sum()),
            "beta_off_on": fit_out["beta_condition"],
            "p_off_on": fit_out["p_condition"],
            "beta_responder_vs_non_on": fit_out["beta_responder_vs_non_on"],
            "p_responder_vs_non_on": fit_out["p_responder_vs_non_on"],
            "beta_responder_vs_non_off": fit_out["beta_responder_vs_non_off"],
            "p_responder_vs_non_off": fit_out["p_responder_vs_non_off"],
            "beta_off_on_responder": fit_out["beta_off_on_responder"],
            "p_off_on_responder": fit_out["p_off_on_responder"],
            "beta_interaction": fit_out["beta_interaction"],
            "p_interaction": fit_out["p_interaction"],
            "converged": fit_out["converged"],
            "lmm_spec": fit_out["lmm_spec"],
        })
        long_df_cache[feature_tag] = long_feat_df

    if not results:
        print("[WARN] All interaction models failed to converge or have missing data.")
        return

    # Summarize and apply FDR correction
    summary_df = pd.DataFrame(results)
    
    _add_metricwise_fdr(summary_df, "p_interaction", "p_interaction_FDR")
    _add_metricwise_fdr(summary_df, "p_off_on", "p_off_on_FDR")
    _add_metricwise_fdr(summary_df, "p_responder_vs_non_on", "p_responder_vs_non_on_FDR")
    _add_metricwise_fdr(summary_df, "p_responder_vs_non_off", "p_responder_vs_non_off_FDR")
    _add_metricwise_fdr(summary_df, "p_off_on_responder", "p_off_on_responder_FDR")

    summary_df["interaction_significant_FDR"] = np.where(summary_df["p_interaction_FDR"] < 0.05, "Yes", "No")
    
    summary_df = summary_df.sort_values(['interaction_significant_FDR', 'p_interaction'], ascending=[False, True])
    summary_df.to_csv(out_dir / "interaction_lmm_summary.csv", index=False)

    sig_df = summary_df[summary_df["interaction_significant_FDR"] == "Yes"].copy()
    sig_df.to_csv(out_dir / "interaction_lmm_significant.csv", index=False)

    # Plot features with significant interaction effects
    for _, row in sig_df.iterrows():
        tag = row["Feature"]
        if tag in long_df_cache:
            _plot_interaction_effects(
                long_df=long_df_cache[tag],
                feature_tag=tag,
                out_dir=sig_plots_dir,
                p_interaction_fdr=float(row["p_interaction_FDR"])
            )

    print(f"\n[SUCCESS] Interaction effect (LMM) analysis complete!")
    print(f"Number of successfully fitted models: {len(summary_df)}")
    print(f"Significant interaction features (p_FDR < 0.05): {len(sig_df)}")
    print(f"Output summary directory: {out_dir}")


if __name__ == "__main__":
    run_response_lmm_interaction_pipeline(
        data_dir="./data/processed_datasets",
        output_root="./data/interaction_output",
        response_threshold_pct=30.0
    )