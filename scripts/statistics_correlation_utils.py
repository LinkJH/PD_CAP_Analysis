import pandas as pd
import numpy as np
import pingouin as pg
import scipy.stats as stats
from statsmodels.stats.multitest import fdrcorrection
from pathlib import Path
import warnings

# ==========================================
# Helper functions
# ==========================================
def _get_metric_prefix(feature_name):
    """Extract metric family; keep TP/TF from being split incorrectly."""
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

def _run_partial_corr_on_group(df_group, group_name, analysis_type, x_cols, y_col, covar_cols):
    results = []
    for x_col in x_cols:
        needed_cols = [x_col, y_col] + covar_cols
        missing = [c for c in needed_cols if c not in df_group.columns]
        if missing:
            continue
            
        valid_df = df_group[needed_cols].copy()
        for c in needed_cols:
            valid_df[c] = pd.to_numeric(valid_df[c], errors='coerce')
        valid_df = valid_df.dropna().copy()
        
        # Partial correlation needs enough degrees of freedom
        if len(valid_df) < len(covar_cols) + 3:
            continue
            
        # Center covariates and drop zero-variance columns
        valid_covars = []
        for c in covar_cols:
            if valid_df[c].nunique() > 1:
                c_centered = f"{c}_centered"
                valid_df[c_centered] = valid_df[c] - valid_df[c].mean()
                valid_covars.append(c_centered)
            
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Use partial correlation when covariates remain
                if valid_covars:
                    res = pg.partial_corr(data=valid_df, x=x_col, y=y_col, covar=valid_covars, method='spearman')
                    
                    # Handle different pingouin p-value column names
                    if 'p-val' in res.columns:
                        p_col = 'p-val'
                    elif 'p_val' in res.columns:
                        p_col = 'p_val'
                    elif 'pval' in res.columns:
                        p_col = 'pval'
                    else:
                        raise KeyError(f"p-value column not found. Pingouin returned columns: {list(res.columns)}")
                        
                    r_val = float(res['r'].iloc[0])
                    p_val = float(res[p_col].iloc[0])
                    
                # If all covariates are constant, fall back to Spearman correlation
                else:
                    r_val, p_val = stats.spearmanr(valid_df[x_col], valid_df[y_col])
                    
            results.append({
                'Group': group_name,
                'Analysis_Type': analysis_type,
                'Imaging_Feature': x_col,
                'Clinical_Target': y_col,
                'N': len(valid_df),
                'Spearman_r': r_val,
                'p_value': p_val
            })
        except Exception as e:
            # Surface errors to identify problematic features
            print(f"  [DEBUG skip] Feature: {x_col} | Error: {e}")
            continue
            
    return results

def apply_fdr_correction(df_results):
    if df_results.empty:
        return df_results
        
    df_results['Metric_Group'] = df_results['Imaging_Feature'].apply(_get_metric_prefix)
    df_results['p_FDR'] = np.nan
    df_results['Significant_FDR'] = 'No'
    
    # Apply FDR separately by group, analysis type, and metric family
    groupby_cols = ['Group', 'Analysis_Type', 'Metric_Group']
    for name, idx in df_results.groupby(groupby_cols).groups.items():
        pvals = df_results.loc[idx, 'p_value'].values
        valid_mask = ~np.isnan(pvals)
        if valid_mask.sum() > 0:
            _, pvals_fdr = fdrcorrection(pvals[valid_mask], alpha=0.05, method='indep')
            df_results.loc[idx[valid_mask], 'p_FDR'] = pvals_fdr
            
    df_results['Significant_FDR'] = np.where(df_results['p_FDR'] < 0.05, 'Yes', 'No')
    return df_results

# ==========================================
# Main analysis pipeline
# ==========================================
def run_correlation_pipeline(data_dir, output_dir, response_threshold_pct=30.0):
    d_dir = Path(data_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("[INFO] Starting correlation analysis (Spearman Partial Correlation)...")
    all_results = []
    
    ignore_patterns = ['subject_id', 'ID', 'Group', 'Run', 'Gender', 'Age', 'Education',
                       'Mean_FD', 'Course', 'LEDD', 'H-Y', 'UPDRS', 'Responder', 'Condition']
    
    # ---------------------------------------------------------
    # Cross-sectional analysis
    # ---------------------------------------------------------
    cross_csv = d_dir / "02_cross_sectional_base.csv"
    if cross_csv.exists():
        df_cross = pd.read_csv(cross_csv)
        df_pd_cross = df_cross[df_cross['Group'] == 'PD'].copy()
        
        all_cross_features = [
            c for c in df_pd_cross.columns 
            if not any(ip in c for ip in ignore_patterns) and pd.api.types.is_numeric_dtype(df_pd_cross[c])
        ]
        
        y_cross = next((c for c in df_pd_cross.columns if 'UPDRS-III(OFF)' in c), None)
        cross_covars = ['Gender', 'Age', 'Years of Education', 'Mean_FD']
        
        if all_cross_features and y_cross:
            res_cross = _run_partial_corr_on_group(
                df_group=df_pd_cross,
                group_name="All_PD",
                analysis_type="Cross-sectional",
                x_cols=all_cross_features,
                y_col=y_cross,
                covar_cols=cross_covars
            )
            all_results.extend(res_cross)
            print(f" -> Cross-sectional analysis complete (tested {len(res_cross)} features)")
            
    # ---------------------------------------------------------
    # Longitudinal analysis
    # ---------------------------------------------------------
    long_csv = d_dir / "03_longitudinal_paired_wide.csv"
    if long_csv.exists():
        df_long = pd.read_csv(long_csv)
        
        long_off_cols = [c for c in df_long.columns if c.endswith('_OFF')]
        base_features = [
            c.replace('_OFF', '') for c in long_off_cols 
            if not any(ip in c for ip in ignore_patterns) and pd.api.types.is_numeric_dtype(df_long[c])
        ]
        
        off_col = next((c for c in df_long.columns if 'UPDRS-III(OFF)' in c and c.endswith('_OFF')), None)
        on_col = next((c for c in df_long.columns if 'UPDRS-III(ON)' in c and c.endswith('_OFF')), None)
        if on_col is None:
            on_col = next((c for c in df_long.columns if 'UPDRS-III(OFF)' in c and c.endswith('_ON')), None)
            
        if off_col and on_col and base_features:
            delta_u, pct_u = _safe_pct_change(df_long[off_col], df_long[on_col])
            
            # Insert new columns in one pass to avoid PerformanceWarning
            new_cols = {
                'Delta_UPDRS': delta_u,
                'PctChange_UPDRS': pct_u,
                'Response_Group': np.where(pct_u >= response_threshold_pct, 'Responder', 'Non-responder')
            }
            
            if 'Mean_FD_OFF' in df_long.columns and 'Mean_FD_ON' in df_long.columns:
                new_cols['Delta_Mean_FD'] = pd.to_numeric(df_long['Mean_FD_OFF'], errors='coerce') - pd.to_numeric(df_long['Mean_FD_ON'], errors='coerce')
            
            delta_features = []
            for feat in base_features:
                f_off, f_on = f"{feat}_OFF", f"{feat}_ON"
                if f_off in df_long.columns and f_on in df_long.columns:
                    d_val, _ = _safe_pct_change(df_long[f_off], df_long[f_on])
                    new_cols[f"Delta_{feat}"] = d_val
                    delta_features.append(f"Delta_{feat}")
                    
            # Concatenate once to keep memory contiguous
            df_long = pd.concat([df_long, pd.DataFrame(new_cols, index=df_long.index)], axis=1)
            
            long_covars = ['Gender_OFF', 'Age_OFF', 'Years of Education_OFF', 'Delta_Mean_FD']
            
            groups_to_test = [
                ("All_PD", df_long),
                ("Responder", df_long[df_long['Response_Group'] == 'Responder']),
                ("Non-responder", df_long[df_long['Response_Group'] == 'Non-responder'])
            ]
            
            for g_name, g_df in groups_to_test:
                res_d = _run_partial_corr_on_group(
                    df_group=g_df, 
                    group_name=g_name, 
                    analysis_type="Longitudinal", 
                    x_cols=delta_features, 
                    y_col='Delta_UPDRS', 
                    covar_cols=long_covars
                )
                all_results.extend(res_d)
                print(f" -> Longitudinal analysis group [{g_name}] complete (tested {len(res_d)} delta features)")
                
    # ---------------------------------------------------------
    # Summary and FDR correction
    # ---------------------------------------------------------
    if not all_results:
        print("[WARN] No correlation results generated. Check console for [DEBUG] error messages.")
        return
        
    df_summary = pd.DataFrame(all_results)
    df_summary = apply_fdr_correction(df_summary)
    
    df_summary = df_summary.sort_values(['Group', 'Analysis_Type', 'Metric_Group', 'p_value'])
    
    summary_path = out_dir / "correlation_results_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    
    sig_df = df_summary[df_summary['Significant_FDR'] == 'Yes'].copy()
    sig_path = out_dir / "correlation_results_significant.csv"
    sig_df.to_csv(sig_path, index=False)
    
    print("\n[SUCCESS] Correlation analysis complete!")
    print(f"Total tests performed: {len(df_summary)}")
    print(f"Significant associations (p_FDR < 0.05): {len(sig_df)}")
    print(f"Output summary file: {summary_path}")

if __name__ == "__main__":
    run_correlation_pipeline(
        data_dir="./data/processed_datasets",
        output_dir="./data/correlation_output",
        response_threshold_pct=30.0
    )