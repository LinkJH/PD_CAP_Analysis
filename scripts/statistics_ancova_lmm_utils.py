import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.multitest import fdrcorrection
from pathlib import Path
import warnings

def _get_metric_prefix(feature_name):
    """Extract metric family name for independent FDR correction."""
    return str(feature_name).split('_')[0]

def adjust_fdr_by_group(df, p_col='p_value', group_col='Metric_Group', fdr_col='p_FDR'):
    """Apply BH-FDR within each metric family."""
    df[fdr_col] = np.nan
    for name, group_idx in df.groupby(group_col).groups.items():
        pvals = df.loc[group_idx, p_col].values
        valid_mask = ~np.isnan(pvals)
        if valid_mask.sum() > 0:
            _, pvals_fdr = fdrcorrection(pvals[valid_mask], alpha=0.05, method='indep')
            df.loc[group_idx[valid_mask], fdr_col] = pvals_fdr
    return df

def run_cross_sectional_ancova(df_cross, target_col):
    """
    ANCOVA formula: Y = beta0 + beta1 Group + beta2 Sex + beta3 Age +
    beta4 Education + beta5 MeanFD + eps
    """
    needed_cols = [target_col, 'Group', 'Gender', 'Age', 'Years of Education', 'Mean_FD']
    valid_df = df_cross.dropna(subset=needed_cols).copy()
    
    # Coerce covariates and target to numeric
    for col in [target_col, 'Gender', 'Age', 'Years of Education', 'Mean_FD']:
        valid_df[col] = pd.to_numeric(valid_df[col], errors='coerce')
        
    valid_df = valid_df.dropna(subset=[target_col, 'Gender', 'Age', 'Years of Education', 'Mean_FD']).copy()
    
    if len(valid_df) < 5:
        return None
        
    # Encode and center within valid samples
    valid_df['Group_encoded'] = valid_df['Group'].map({'HC': 0, 'PD': 1})
    valid_df['Gender_centered'] = valid_df['Gender'] - valid_df['Gender'].mean()
    valid_df['Age_centered'] = valid_df['Age'] - valid_df['Age'].mean()
    valid_df['Education_centered'] = valid_df['Years of Education'] - valid_df['Years of Education'].mean()
    valid_df['Mean_FD_centered'] = valid_df['Mean_FD'] - valid_df['Mean_FD'].mean()

    X_cols = ['Group_encoded', 'Gender_centered', 'Age_centered', 'Education_centered', 'Mean_FD_centered']
    
    # Cast to float for statsmodels
    X = valid_df[X_cols].astype(float)
    X = sm.add_constant(X)
    y = valid_df[target_col].astype(float)

    model = sm.OLS(y, X).fit()
    
    try:
        f_test = model.f_test("Group_encoded = 0")
        f_val = float(np.squeeze(f_test.fvalue))
        p_f = float(np.squeeze(f_test.pvalue))
    except Exception:
        f_val, p_f = np.nan, np.nan

    return {
        'Feature': target_col,
        'beta_Group': model.params.get('Group_encoded', np.nan),
        't_value': model.tvalues.get('Group_encoded', np.nan),
        'p_value': model.pvalues.get('Group_encoded', np.nan),
        'F_value': f_val,
        'p_F_ancova': p_f,
        'N': len(valid_df)
    }

def run_longitudinal_lmm(df_wide, target_col):
    """
    LME formula: Y_ij = beta0 + beta1 Condition_ij + beta2 Sex_i + beta3 Age_i +
    beta4 Education_i + beta5 MeanFD_ij + u_i + eps_ij
    """
    col_off = f"{target_col}_OFF"
    col_on = f"{target_col}_ON"
    
    needed_base = ['subject_id', 'Gender_OFF', 'Age_OFF', 'Years of Education_OFF']
    needed_time = [col_off, col_on, 'Mean_FD_OFF', 'Mean_FD_ON']
    
    missing = [c for c in needed_base + needed_time if c not in df_wide.columns]
    if missing:
        return None

    # Keep complete cases
    valid_wide = df_wide.dropna(subset=needed_base + needed_time).copy()
    
    # Coerce analysis columns to numeric
    numeric_cols = [col_off, col_on, 'Gender_OFF', 'Age_OFF', 'Years of Education_OFF', 'Mean_FD_OFF', 'Mean_FD_ON']
    for col in numeric_cols:
        valid_wide[col] = pd.to_numeric(valid_wide[col], errors='coerce')
        
    valid_wide = valid_wide.dropna(subset=numeric_cols).copy()

    if len(valid_wide) < 3:
        return None
        
    # Convert to long format (time-varying covariates)
    off_df = pd.DataFrame({
        'subject_id': valid_wide['subject_id'],
        'Metric': valid_wide[col_off],
        'Condition_encoded': 1,  # OFF=1
        'Mean_FD': valid_wide['Mean_FD_OFF'],
        'Gender_i': valid_wide['Gender_OFF'],
        'Age_i': valid_wide['Age_OFF'],
        'Education_i': valid_wide['Years of Education_OFF']
    })
    
    on_df = off_df.copy()
    on_df['Metric'] = valid_wide[col_on]
    on_df['Condition_encoded'] = 0  # ON=0
    on_df['Mean_FD'] = valid_wide['Mean_FD_ON']
    
    long_df = pd.concat([off_df, on_df], ignore_index=True)
    
    # Center within samples
    long_df['Gender_centered'] = long_df['Gender_i'] - long_df['Gender_i'].mean()
    long_df['Age_centered'] = long_df['Age_i'] - long_df['Age_i'].mean()
    long_df['Education_centered'] = long_df['Education_i'] - long_df['Education_i'].mean()
    long_df['MeanFD_ij_centered'] = long_df['Mean_FD'] - long_df['Mean_FD'].mean() 

    y = long_df['Metric'].astype(float)
    groups = long_df['subject_id']
    X_cols = ['Condition_encoded', 'Gender_centered', 'Age_centered', 'Education_centered', 'MeanFD_ij_centered']
    
    # Filter zero-variance covariates to avoid singular matrices
    usable_cols = [c for c in X_cols if long_df[c].nunique() > 1 or c == 'Condition_encoded']
    X = long_df[usable_cols].astype(float)
    X = sm.add_constant(X)
    
    res_dict = {
        'Feature': target_col, 'beta_Condition': np.nan, 't_value': np.nan, 
        'p_value': np.nan, 'N_pairs': len(valid_wide), 'converged': False
    }
    residuals_data = None

    # Retry with multiple optimizers to reduce non-convergence
    fit_attempts = [
        ("lbfgs", False),  # LBFGS ML
        ("lbfgs", True),   # LBFGS REML
        ("powell", True),  # Powell (robust but slower)
        ("cg", False)      # Conjugate gradient
    ]
    
    for method, reml in fit_attempts:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = MixedLM(endog=y, exog=X, groups=groups)
                fitted = model.fit(reml=reml, method=method, maxiter=500, disp=False)
                
            # Accept fit only if converged and finite
            if fitted.converged and np.isfinite(fitted.params.get('Condition_encoded', np.nan)):
                res_dict.update({
                    'beta_Condition': fitted.params['Condition_encoded'],
                    't_value': fitted.tvalues['Condition_encoded'],
                    'p_value': fitted.pvalues['Condition_encoded'],
                    'converged': True
                })
                
                # Capture residuals
                residuals_data = pd.DataFrame({
                    'Feature': target_col,
                    'subject_id': long_df['subject_id'],
                    'Condition_encoded': long_df['Condition_encoded'],
                    'Fitted': fitted.fittedvalues,
                    'Residuals': fitted.resid
                })
                break  # Stop once a valid fit is found
                
        except Exception:
            continue  # Try next optimizer

    return res_dict, residuals_data

def merge_analysis_results(df_ancova, df_lmm, output_path=None):
    """
    Merge ANCOVA and LMM results into a single table.

    Args:
        df_ancova: ANCOVA (cross-sectional) results.
        df_lmm: LMM (longitudinal) results.
        output_path: Optional output path (e.g., 'merged_results.csv').

    Returns:
        DataFrame with merged results.
    """
    
    # ---------------- 1. ANCOVA table (cross-sectional) ----------------
    df_ancova_clean = pd.DataFrame()
    
    # Extract metric prefix
    df_ancova_clean['Metric'] = df_ancova['Feature'].apply(
        lambda x: x.rsplit('_', 1)[0] if any(c.isdigit() for c in x) else x
    )
    # Extract feature suffix (e.g., CAP-1, 1.1)
    df_ancova_clean['Feature'] = df_ancova['Feature'].apply(lambda x: x.split('_')[-1])
    
    df_ancova_clean['Analysis_Type'] = 'Cross-sectional'
    df_ancova_clean['N'] = df_ancova['N']
    df_ancova_clean['beta'] = df_ancova['beta_Group']
    df_ancova_clean['t_value'] = df_ancova['t_value']
    df_ancova_clean['p_value'] = df_ancova['p_value']
    df_ancova_clean['p_FDR'] = df_ancova['p_FDR']
    
    # ---------------- 2. LMM table (longitudinal) ----------------
    df_lmm_clean = pd.DataFrame()
    
    df_lmm_clean['Metric'] = df_lmm['Feature'].apply(
        lambda x: x.rsplit('_', 1)[0] if any(c.isdigit() for c in x) else x
    )
    df_lmm_clean['Feature'] = df_lmm['Feature'].apply(lambda x: x.split('_')[-1])
    
    df_lmm_clean['Analysis_Type'] = 'Longitudinal'
    # LMM reports N_pairs; total rows are pairs times 2
    df_lmm_clean['N'] = df_lmm['N_pairs'] * 2 
    df_lmm_clean['beta'] = df_lmm['beta_Condition']
    df_lmm_clean['t_value'] = df_lmm['t_value']
    df_lmm_clean['p_value'] = df_lmm['p_value']
    df_lmm_clean['p_FDR'] = df_lmm['p_FDR']
    
    # ---------------- 3. Merge and save ----------------
    merged_df = pd.concat([df_ancova_clean, df_lmm_clean], ignore_index=True)
    significant_df = merged_df[merged_df['p_FDR'] <= 0.05]
    
    if output_path:
        merged_df.to_csv(output_path, index=False)
        significant_df.to_csv(str(output_path).replace('.csv', '_significant.csv'), index=False)

        print(f"[INFO] Results merged and saved to: {output_path}")
        
    return merged_df

def run_statistical_pipeline(cross_csv, long_wide_csv, output_dir):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df_cross = pd.read_csv(cross_csv)
    df_long = pd.read_csv(long_wide_csv)
    
    # Include 'ID' and other potential header noise
    ignore_patterns = ['subject_id', 'ID', 'Group', 'Run', 'Gender', 'Age', 'Education', 'Mean_FD', 'Std_FD', 
                       'Course', 'LEDD', 'H-Y', 'UPDRS', 'Responder', 'Condition']
    
    # Run cross-sectional ANCOVA
    print("[INFO] Running ANCOVA (Cross-sectional)...")
    # Enforce numeric dtype to exclude text-like ID columns
    cross_metrics = [
        c for c in df_cross.columns 
        if not any(ip in c for ip in ignore_patterns) and pd.api.types.is_numeric_dtype(df_cross[c])
    ]
    
    ancova_results = []
    for metric in cross_metrics:
        res = run_cross_sectional_ancova(df_cross, metric)
        if res: ancova_results.append(res)
        
    df_ancova = pd.DataFrame(ancova_results)
    if not df_ancova.empty:
        df_ancova['Metric_Group'] = df_ancova['Feature'].apply(_get_metric_prefix)
        df_ancova = adjust_fdr_by_group(df_ancova)
        df_ancova.to_csv(out_dir / "stats_ANCOVA_results.csv", index=False)

    # Run longitudinal LMM
    print("[INFO] Running LMM (Longitudinal)...")
    long_cols = [c for c in df_long.columns if c.endswith('_OFF')]
    long_metrics = [
        c.replace('_OFF', '') for c in long_cols 
        if not any(ip in c for ip in ignore_patterns) and pd.api.types.is_numeric_dtype(df_long[c])
    ]
    
    lmm_results = []
    all_residuals = []
    for metric in long_metrics:
        res = run_longitudinal_lmm(df_long, metric)
        if res:
            stats_dict, resid_df = res
            lmm_results.append(stats_dict)
            if resid_df is not None:
                all_residuals.append(resid_df)
                
    df_lmm = pd.DataFrame(lmm_results)
    if not df_lmm.empty:
        df_lmm['Metric_Group'] = df_lmm['Feature'].apply(_get_metric_prefix)
        df_lmm = adjust_fdr_by_group(df_lmm)
        df_lmm.to_csv(out_dir / "stats_LMM_results.csv", index=False)
        
    if all_residuals:
        df_resid = pd.concat(all_residuals, ignore_index=True)
        df_resid.to_csv(out_dir / "stats_LMM_residuals.csv", index=False)

    # Merge result tables
    merge_analysis_results(df_ancova, df_lmm, output_path=out_dir / "all_cross_longitudinal_merged.csv")

    print(f"[SUCCESS] Statistical analysis complete! Results saved to: {out_dir}")