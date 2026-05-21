import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import adjusted_rand_score

def run_bootstrap_clustering_stability(
    cap_analysis, 
    n_boot=1000, 
    random_state=42,
    output_dir=None,
    group_name=None,
    n_init_boot=10,
    use_cached_data=True  # Prefer cached files when available
):
    """
    Evaluate CAP clustering stability with bootstrap resampling.
    Supports cached inputs and outputs 95% CI and trimmed boxplots.
    """
    if cap_analysis.kmeans is None or cap_analysis.concatenated_timeseries is None:
        raise ValueError("cap_analysis has no kmeans or concatenated_timeseries attribute. Do not delete cached data after calling get_caps.")

    if group_name is None:
        if "All Subjects" in cap_analysis.kmeans:
            group_name = "All Subjects"
        else:
            group_name = list(cap_analysis.kmeans.keys())[0]

    safe_group_name = group_name.replace(" ", "_").replace("/", "_")
    
    # Prepare cache paths
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        out_csv_raw_corr = os.path.join(output_dir, f"bootstrapping_raw_correlations_{safe_group_name}.csv")
        out_csv_raw_ari = os.path.join(output_dir, f"bootstrapping_raw_ari_{safe_group_name}.csv")
        cache_exists = os.path.exists(out_csv_raw_corr) and os.path.exists(out_csv_raw_ari)
    else:
        cache_exists = False
        out_csv_raw_corr = None
        out_csv_raw_ari = None

    print(f"[INFO] Group '{group_name}': Preparing Bootstrap clustering stability assessment...")
    
    km_ref = cap_analysis.kmeans[group_name]
    n_clusters = km_ref.n_clusters
    
    # ==========================================
    # Branch 1: read cached data (skip long computation)
    # ==========================================
    if use_cached_data and cache_exists:
        print(f"[INFO] use_cached_data=True and local files detected.")
        print(f"[INFO] ---> Skipping Bootstrap computation, reading from cache and generating plots...")
        
        # Read correlations
        corr_df = pd.read_csv(out_csv_raw_corr)
        boot_correlations = corr_df.values
        
        # Read ARI
        ari_df_raw = pd.read_csv(out_csv_raw_ari)
        boot_ari_scores = ari_df_raw.values.flatten()
        
        # Align with cached iteration count when it differs
        if boot_correlations.shape[0] != n_boot:
            print(f"[WARN] Cached iteration count ({boot_correlations.shape[0]}) differs from n_boot ({n_boot}). Using cached data as reference.")
            n_boot = boot_correlations.shape[0]

    # ==========================================
    # Branch 2: full bootstrap computation
    # ==========================================
    else:
        if use_cached_data and not cache_exists:
            print(f"[INFO] use_cached_data=True but no complete local files found.")
            print(f"[INFO] ---> Forcing full Bootstrap resampling and clustering computation...")
        
        reference_centers = km_ref.cluster_centers_
        X_all = cap_analysis.concatenated_timeseries[group_name]
        n_samples, n_rois = X_all.shape
        
        # Reference labels as ground truth
        ref_labels = km_ref.predict(X_all)
        
        print(f"[INFO] Total data frames: {n_samples}, ROIs: {n_rois}")
        print(f"[INFO] Clusters (CAPs): {n_clusters}, Bootstrap iterations: {n_boot}")
        
        np.random.seed(random_state)
        boot_correlations = np.zeros((n_boot, n_clusters))
        boot_ari_scores = np.zeros(n_boot) 
        
        max_iter = getattr(km_ref, "max_iter", 300)
        algorithm = getattr(km_ref, "algorithm", "lloyd")
        
        for i in tqdm(range(n_boot), desc="Bootstrap Resampling"):
            # 1. Resample frames with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X_all[indices, :]
            
            # 2. Refit KMeans
            km_boot = KMeans(
                n_clusters=n_clusters, 
                init='k-means++',
                n_init=n_init_boot, 
                max_iter=max_iter, 
                algorithm=algorithm, 
                random_state=random_state + i
            )
            km_boot.fit(X_boot)
            boot_centers = km_boot.cluster_centers_
            
            # 3. ARI for bootstrap model on full data
            boot_labels_all = km_boot.predict(X_all)
            boot_ari_scores[i] = adjusted_rand_score(ref_labels, boot_labels_all)
            
            # 4. Match centroids using correlation distance
            dist_matrix = cdist(reference_centers, boot_centers, metric='correlation')
            row_ind, col_ind = linear_sum_assignment(dist_matrix)
            
            # 5. Record correlation r = 1 - cdist
            for ref_idx, boot_idx in zip(row_ind, col_ind):
                corr_val = 1.0 - dist_matrix[ref_idx, boot_idx]
                boot_correlations[i, ref_idx] = corr_val

        # Persist new results when output_dir is provided
        if output_dir:
            # Save correlations
            columns = [f"CAP_{j+1}" for j in range(n_clusters)]
            corr_df = pd.DataFrame(boot_correlations, columns=columns)
            corr_df.to_csv(out_csv_raw_corr, index=False)
            
            # Save ARI (header must match reader)
            np.savetxt(out_csv_raw_ari, boot_ari_scores, delimiter=",", header="ARI_Score", comments="")
            print(f"[INFO] New raw correlation and ARI data cached locally.")

    # ==========================================
    # Stats and plotting (shared for cached or computed data)
    # ==========================================
    
    # Summarize spatial correlations (95% CI)
    ci_low = np.percentile(boot_correlations, 2.5, axis=0)
    ci_high = np.percentile(boot_correlations, 97.5, axis=0)
    
    stability_df = pd.DataFrame({
        "CAP_Index": np.arange(1, n_clusters + 1),
        "Mean_Correlation": np.mean(boot_correlations, axis=0),
        "Median_Correlation": np.median(boot_correlations, axis=0),
        "Std_Correlation": np.std(boot_correlations, axis=0),
        "95%_CI_Low": ci_low,
        "95%_CI_High": ci_high
    })
    
    # Summarize ARI (95% CI)
    ari_ci_low = np.percentile(boot_ari_scores, 2.5)
    ari_ci_high = np.percentile(boot_ari_scores, 97.5)
    
    ari_df = pd.DataFrame([{
        "Metric": "Adjusted Rand Index (ARI)",
        "Mean": np.mean(boot_ari_scores),
        "Median": np.median(boot_ari_scores),
        "Std": np.std(boot_ari_scores),
        "95%_CI_Low": ari_ci_low,
        "95%_CI_High": ari_ci_high
    }])
    
    print("\n=== Bootstrap Clustering Spatial Stability (Pearson Centroid Correlation, 95% CI) ===")
    print(stability_df.to_string(index=False))
    
    print("\n=== Bootstrap Assignment Consistency (Adjusted Rand Index, 95% CI) ===")
    print(ari_df.to_string(index=False))
    
    if output_dir:
        # Save summary tables
        out_csv_stats = os.path.join(output_dir, f"bootstrapping_clustering_stability_{safe_group_name}.csv")
        stability_df.to_csv(out_csv_stats, index=False)
        
        out_csv_ari = os.path.join(output_dir, f"bootstrapping_ari_summary_{safe_group_name}.csv")
        ari_df.to_csv(out_csv_ari, index=False)
        
        # Plot using a DataFrame (rebuild columns if needed)
        columns = [f"CAP_{j+1}" for j in range(n_clusters)]
        corr_df_for_plot = pd.DataFrame(boot_correlations, columns=columns)
        _plot_stability_boxplot(corr_df_for_plot, n_clusters, output_dir, safe_group_name)
        
        print(f"[INFO] Statistical reports and visualization boxplots updated: {output_dir}")

    return stability_df, boot_correlations, boot_ari_scores

def _plot_stability_boxplot(corr_df, n_clusters, output_dir, safe_group_name):
    """
    Helper: draw scatter boxplots for CAP stability using a custom gradient.
    Only plot values within the 95% CI to avoid extreme compression.
    """
    colors = [
        "#5364c0",  # blue
        "#77b7f0",  # light blue
        "#edf1fd",  # off-white
        "#c34fa2",  # pink
        "#77295d"   # dark magenta
    ]
    custom_cmap = LinearSegmentedColormap.from_list('custom_RdBu', colors)
    palette = [custom_cmap(i) for i in np.linspace(0, 1, n_clusters)]
    
    filtered_data = []
    for cap_col in corr_df.columns:
        col_data = corr_df[cap_col]
        low_bound = np.percentile(col_data, 2.5)
        high_bound = np.percentile(col_data, 97.5)
        
        # Keep values within CI bounds
        mask = (col_data >= low_bound) & (col_data <= high_bound)
        filtered_cap = col_data[mask].to_frame(name="Pearson_Correlation")
        filtered_cap["CAP_Network"] = cap_col
        filtered_data.append(filtered_cap)
        
    corr_melted_filtered = pd.concat(filtered_data, ignore_index=True)
    
    plt.figure(figsize=(10, 6))
    
    sns.boxplot(
        x="CAP_Network", 
        y="Pearson_Correlation", 
        hue="CAP_Network",
        data=corr_melted_filtered, 
        palette=palette, 
        showfliers=False,
        legend=False,
        width=0.5
    )
    
    sns.stripplot(
        x="CAP_Network", 
        y="Pearson_Correlation", 
        data=corr_melted_filtered, 
        color="black", 
        alpha=0.3, 
        jitter=True, 
        size=3
    )
    
    # Adjust y-range based on filtered data
    y_min = corr_melted_filtered["Pearson_Correlation"].min()
    y_max = corr_melted_filtered["Pearson_Correlation"].max()
    y_margin = (y_max - y_min) * 0.1
    # Keep headroom without exceeding 1.0
    plt.ylim(y_min - y_margin, min(1.02, y_max + y_margin))
    
    plt.title("Bootstrap Clustering Stability across CAPs (95% CI Range)", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel("Spatial Pearson Correlation", fontsize=12)
    plt.xlabel("Co-Activation Patterns", fontsize=12)
    
    sns.despine()
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"bootstrapping_stability_boxplot_{safe_group_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()