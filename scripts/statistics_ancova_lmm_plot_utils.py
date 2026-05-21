import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ptitprince as pt
import statsmodels.api as sm
from pathlib import Path

def _safe_tag(name):
    """Sanitize strings for filenames."""
    return str(name).replace(" ", "_").replace("/", "_").replace(".", "_to_")

def _add_significance_bracket(ax, x1, x2, y_max, p_val, height_offset=0.08):
    """
    Add statistical significance brackets and stars.

    Args:
        ax: matplotlib axes.
        x1: X position of the first category (e.g., 0).
        x2: X position of the second category (e.g., 1).
        y_max: Max y-value used to place the bracket.
        p_val: FDR-adjusted p-value.
    """
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    y_base = y_max + y_range * height_offset
    tick_len = y_range * 0.02
    
    # Draw the bracket
    ax.plot([x1, x1, x2, x2], [y_base, y_base + tick_len, y_base + tick_len, y_base], lw=1.5, color='black')
    
    # Decide star label
    if p_val < 0.001:
        text = "***"
    elif p_val < 0.01:
        text = "**"
    elif p_val < 0.05:
        text = "*"
    else:
        text = "ns"
        
    # Add star label
    ax.text((x1 + x2) / 2, y_base + tick_len, text, ha='center', va='bottom', color='black', fontsize=16, fontweight='bold')
    
    # Extend y-limit to avoid clipping
    new_y_max = y_base + tick_len * 4
    if ax.get_ylim()[1] < new_y_max:
        ax.set_ylim(ax.get_ylim()[0], new_y_max)

def plot_ancova_significant_features(cross_csv, ancova_res_csv, output_dir):
    df_cross = pd.read_csv(cross_csv)
    df_res = pd.read_csv(ancova_res_csv)
    
    out_dir = Path(output_dir) / "ancova_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter significant features
    sig_features = df_res[df_res['p_FDR'] < 0.05]
    if sig_features.empty:
        print("[INFO] ANCOVA: no significant features, skipping plot.")
        return
        
    for _, row in sig_features.iterrows():
        feat = row['Feature']
        p_fdr = row['p_FDR']
        
        # Drop NaNs to avoid plot errors
        plot_df = df_cross.dropna(subset=['Group', feat]).copy()
        
        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
        order = ['HC', 'PD']
        
        # Raincloud plot (explicit hue to avoid seaborn warning)
        pt.RainCloud(
            x='Group', y=feat, hue='Group', data=plot_df, palette='Set2', 
            bw=0.2, width_viol=0.4, width_box=0.15, ax=ax, orient='v', 
            alpha=0.65, dodge=False, order=order
        )
        
        # Remove redundant legend created by hue
        if ax.legend_:
            ax.legend_.remove()
            
        # Add significance bracket
        y_max = plot_df[feat].max()
        _add_significance_bracket(ax, x1=0, x2=1, y_max=y_max, p_val=p_fdr)
        
        ax.set_title(f"Cross-sectional: {feat}\nANCOVA $p_{{FDR}}$ = {p_fdr:.4f}", pad=15)
        ax.set_xlabel("Group")
        ax.set_ylabel(feat)
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        
        fig.tight_layout()
        fig.savefig(out_dir / f"cross_{_safe_tag(feat)}_raincloud.png")
        plt.close(fig)

def plot_lmm_significant_features(long_wide_csv, lmm_res_csv, output_dir):
    df_long = pd.read_csv(long_wide_csv)
    df_res = pd.read_csv(lmm_res_csv)
    
    out_dir = Path(output_dir) / "lmm_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    sig_features = df_res[df_res['p_FDR'] < 0.05]
    if sig_features.empty:
        print("[INFO] LMM: no significant features, skipping plot.")
        return
        
    for _, row in sig_features.iterrows():
        feat = row['Feature']
        p_fdr = row['p_FDR']
        
        col_off = f"{feat}_OFF"
        col_on = f"{feat}_ON"
        
        if col_off not in df_long.columns or col_on not in df_long.columns:
            continue
            
        # Drop NaNs to keep paired data
        valid_long = df_long.dropna(subset=['subject_id', col_off, col_on]).copy()
        
        # Compute delta (OFF - ON) for a single-group raincloud plot
        valid_long['Delta'] = valid_long[col_off] - valid_long[col_on]
        valid_long['Condition'] = 'OFF - ON'
        
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
        
        # Delta raincloud plot
        pt.RainCloud(
            x='Condition', y='Delta', hue='Condition', data=valid_long, 
            palette=['#d95f02'], bw=0.2, width_viol=0.4, width_box=0.15, 
            ax=ax, orient='v', alpha=0.65, dodge=False
        )
        
        # Add y=0 reference line
        ax.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
        
        # Remove legend
        if ax.legend_:
            ax.legend_.remove()
        
        # No bracket needed; show p-value in the title
        ax.set_title(f"Longitudinal Difference: {feat}\nLMM $p_{{FDR}}$ = {p_fdr:.4f}", pad=15)
        ax.set_xlabel("Medication Effect")
        ax.set_ylabel(f"$\Delta$ {feat} (OFF - ON)")
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        
        fig.tight_layout()
        fig.savefig(out_dir / f"long_{_safe_tag(feat)}_delta_raincloud.png")
        plt.close(fig)

def plot_lmm_diagnostics(residuals_csv, output_dir):
    """Generate diagnostics to assess LMM fit."""
    if not Path(residuals_csv).exists():
        return
        
    df_resid = pd.read_csv(residuals_csv)
    out_dir = Path(output_dir) / "lmm_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for feat, group in df_resid.groupby('Feature'):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=200)
        
        # Residuals vs fitted
        axes[0].scatter(group['Fitted'], group['Residuals'], alpha=0.6)
        axes[0].axhline(0, color='red', linestyle='--')
        axes[0].set_xlabel("Fitted Values")
        axes[0].set_ylabel("Residuals")
        axes[0].set_title(f"{feat} - Residuals")
        
        # QQ plot
        sm.qqplot(group['Residuals'].dropna(), line='s', ax=axes[1])
        axes[1].set_title(f"{feat} - QQ Plot")
        
        fig.tight_layout()
        fig.savefig(out_dir / f"diag_{_safe_tag(feat)}.png")
        plt.close(fig)

def generate_all_visualizations(data_dir, stats_dir, output_dir):
    d_dir = Path(data_dir)
    s_dir = Path(stats_dir)
    
    print("[INFO] Generating ANCOVA cross-sectional raincloud plots...")
    plot_ancova_significant_features(
        cross_csv=d_dir / "02_cross_sectional_base.csv",
        ancova_res_csv=s_dir / "stats_ANCOVA_results.csv",
        output_dir=output_dir
    )
    
    print("[INFO] Generating LMM longitudinal delta raincloud plots...")
    plot_lmm_significant_features(
        long_wide_csv=d_dir / "03_longitudinal_paired_wide.csv",
        lmm_res_csv=s_dir / "stats_LMM_results.csv",
        output_dir=output_dir
    )
    
    print("[INFO] Generating LMM residual diagnostic plots...")
    plot_lmm_diagnostics(
        residuals_csv=s_dir / "stats_LMM_residuals.csv",
        output_dir=output_dir
    )
    print(f"[SUCCESS] All visualizations complete! Figures saved to: {output_dir}")

if __name__ == "__main__":
    generate_all_visualizations(
        data_dir="./data/processed_datasets",
        stats_dir="./data/statistics_output",
        output_dir="./data/visualizations_output"
    )