import pickle

import pandas as pd
from neurocaps.analysis.cap._internals import cluster


def reorder_caps_with_state_check(cap_analysis, cap_cache, new_cap_order, force=False):
    """
    Reorder CAPs once and persist reorder metadata to avoid duplicate reorder on reruns.
    """
    has_reorder_mapping = hasattr(cap_analysis, "_reorder_mapping")

    if has_reorder_mapping and not force:
        print("=" * 60)
        print("[INFO] CAPs already reordered, skipping reorder operation")
        print("=" * 60)
        existing_mapping = cap_analysis._reorder_mapping
        print(f"Applied reorder mapping: {existing_mapping}")
        print("\nTo reorder again, re-run this cell with force=True")
        return False

    print("=" * 60)
    print(f"Starting CAP reorder ({new_cap_order})")
    print("=" * 60)

    new_indices = [x - 1 for x in new_cap_order]

    print("\nNew CAP order mapping:")
    print("  Original position -> New position (original CAP at new position)")
    for new_pos, original_cap_num in enumerate(new_cap_order, 1):
        old_index = original_cap_num - 1
        print(f"  CAP-{original_cap_num} (original index {old_index}) moved to position {new_pos}")

    print("\nRearranging KMeans cluster centers...")
    for group_name in cap_analysis._kmeans:
        old_centers = cap_analysis._kmeans[group_name].cluster_centers_.copy()
        n_clusters = old_centers.shape[0]

        if len(new_indices) != n_clusters:
            print(f"[WARN] Group {group_name}: expected {n_clusters} CAPs, but {len(new_indices)} provided")
            continue

        cap_analysis._kmeans[group_name].cluster_centers_ = old_centers[new_indices]
        print(f"  [OK] cluster_centers_ for group '{group_name}' rearranged")

    print("\nRebuilding CAP dictionary...")
    cap_analysis._caps = cluster.create_caps_dict(cap_analysis._kmeans)
    print("  [OK] CAP dictionary rebuilt")

    print("\nVerifying new CAP order:")
    for group_name, cap_dict in cap_analysis._caps.items():
        cap_names = list(cap_dict.keys())
        print(f"  Group '{group_name}': {cap_names}")

    cap_analysis._reorder_mapping = {
        "new_order": new_cap_order,
        "timestamp": pd.Timestamp.now().isoformat(),
    }

    print(f"\nSaving updated cache to: {cap_cache}")
    with open(cap_cache, "wb") as f:
        pickle.dump(cap_analysis, f)
    print("  [OK] Cache saved successfully (with reorder flag)")

    print("\n" + "=" * 60)
    print("CAP reorder complete")
    print("=" * 60)

    return True
