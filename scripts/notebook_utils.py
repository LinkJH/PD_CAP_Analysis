import json
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_name(text: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in str(text))


def get_roi_labels(parcel_approach: dict, n_rois: int) -> list[str]:
    custom = parcel_approach.get("Custom", {}) if isinstance(parcel_approach, dict) else {}
    nodes = custom.get("nodes")

    if isinstance(nodes, list) and len(nodes) == n_rois:
        return [str(x) for x in nodes]

    if isinstance(nodes, dict):
        try:
            sorted_items = sorted(nodes.items(), key=lambda kv: int(str(kv[0])))
        except ValueError:
            sorted_items = sorted(nodes.items(), key=lambda kv: str(kv[0]))

        labels = [str(v) for _, v in sorted_items]
        if len(labels) == n_rois:
            return labels

    return [f"ROI_{i + 1:03d}" for i in range(n_rois)]


def save_subject_timeseries_csv(
    subject_timeseries: dict,
    output_dir: Path,
    roi_labels: list[str] | None = None,
    save_wide_csv: bool = True,
    save_long_csv: bool = False,
    save_npz: bool = True,
    write_index_json: bool = True,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    index_records = []
    for subj_id in sorted(subject_timeseries):
        for run_id in sorted(subject_timeseries[subj_id]):
            arr = np.asarray(subject_timeseries[subj_id][run_id])
            if arr.ndim != 2:
                raise ValueError(f"Timeseries for subject {subj_id}, run {run_id} is not 2D.")

            n_frames, n_rois = arr.shape
            curr_labels = roi_labels if (roi_labels and len(roi_labels) == n_rois) else [
                f"ROI_{i + 1:03d}" for i in range(n_rois)
            ]

            safe_subj = _safe_name(subj_id)
            safe_run = _safe_name(run_id)
            stem = f"sub-{safe_subj}_{safe_run}"

            if save_wide_csv:
                df_wide = pd.DataFrame(arr, columns=curr_labels)
                df_wide.insert(0, "Frame", np.arange(n_frames, dtype=int))
                wide_path = output_dir / f"{stem}_timeseries_wide.csv"
                df_wide.to_csv(wide_path, index=False)
            else:
                wide_path = None

            if save_long_csv:
                df_long = pd.DataFrame(arr, columns=curr_labels)
                df_long["Frame"] = np.arange(n_frames, dtype=int)
                df_long = df_long.melt(
                    id_vars=["Frame"],
                    var_name="ROI",
                    value_name="Signal",
                )
                df_long.insert(0, "Run", run_id)
                df_long.insert(0, "Subject_ID", subj_id)
                long_path = output_dir / f"{stem}_timeseries_long.csv"
                df_long.to_csv(long_path, index=False)
            else:
                long_path = None

            if save_npz:
                npz_path = output_dir / f"{stem}_timeseries.npz"
                np.savez_compressed(npz_path, timeseries=arr, roi_labels=np.asarray(curr_labels))
            else:
                npz_path = None

            summary_rows.append(
                {
                    "Subject_ID": subj_id,
                    "Run": run_id,
                    "N_Frames": n_frames,
                    "N_ROIs": n_rois,
                }
            )

            index_records.append(
                {
                    "Subject_ID": subj_id,
                    "Run": run_id,
                    "N_Frames": n_frames,
                    "N_ROIs": n_rois,
                    "wide_csv": str(wide_path) if wide_path else None,
                    "long_csv": str(long_path) if long_path else None,
                    "npz": str(npz_path) if npz_path else None,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "timeseries_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    if write_index_json:
        index_path = output_dir / "timeseries_file_index.json"
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_records, f, indent=2, ensure_ascii=False)

    return summary_df


def save_cluster_scores(cluster_scores: dict | None, output_dir: Path) -> pd.DataFrame | None:
    if not cluster_scores or "Scores" not in cluster_scores:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    method = cluster_scores.get("Cluster_Selection_Method", "unknown")

    with open(output_dir / "cluster_scores.json", "w", encoding="utf-8") as f:
        json.dump(cluster_scores, f, indent=2, ensure_ascii=False)

    rows = []
    for group_name, score_dict in cluster_scores["Scores"].items():
        for k, score in sorted(score_dict.items(), key=lambda kv: int(kv[0])):
            rows.append(
                {
                    "Group": group_name,
                    "Cluster_Selection_Method": method,
                    "k": int(k),
                    "Score": float(score),
                }
            )

    df = pd.DataFrame(rows).sort_values(["Group", "k"]).reset_index(drop=True)
    df.to_csv(output_dir / f"cluster_scores_{method}.csv", index=False)
    return df


def save_predicted_labels(predicted_subject_timeseries: dict, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for subj_id in sorted(predicted_subject_timeseries):
        for run_id in sorted(predicted_subject_timeseries[subj_id]):
            labels = np.asarray(predicted_subject_timeseries[subj_id][run_id]).astype(int)
            frame_df = pd.DataFrame(
                {
                    "Subject_ID": subj_id,
                    "Run": run_id,
                    "Frame": np.arange(labels.size, dtype=int),
                    "CAP_Label": labels,
                }
            )
            safe_subj = _safe_name(subj_id)
            safe_run = _safe_name(run_id)
            frame_df.to_csv(output_dir / f"sub-{safe_subj}_{safe_run}_cap_labels.csv", index=False)
            rows.append(frame_df)

    all_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["Subject_ID", "Run", "Frame", "CAP_Label"]
    )
    all_df.to_csv(output_dir / "all_subjects_cap_labels.csv", index=False)
    return all_df


def save_cap_templates(
    caps: dict,
    parcel_approach: dict,
    output_dir: Path,
    save_long_csv: bool = True,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    group_dfs = {}

    for group_name, cap_dict in caps.items():
        first_cap = next(iter(cap_dict.values()))
        n_rois = np.asarray(first_cap).ravel().size
        roi_labels = get_roi_labels(parcel_approach, n_rois)

        rows = []
        for cap_name, vector in cap_dict.items():
            vec = np.asarray(vector).ravel()
            if vec.size != n_rois:
                raise ValueError(
                    f"CAP vector size mismatch in group {group_name}, cap {cap_name}: "
                    f"expected {n_rois}, got {vec.size}."
                )
            row = {"Group": group_name, "CAP": cap_name}
            row.update({roi_labels[i]: float(vec[i]) for i in range(n_rois)})
            rows.append(row)

        df = pd.DataFrame(rows)
        safe_group = _safe_name(group_name)
        df.to_csv(output_dir / f"cap_templates_{safe_group}.csv", index=False)

        if save_long_csv:
            long_df = df.melt(
                id_vars=["Group", "CAP"],
                var_name="ROI",
                value_name="TemplateValue",
            )
            long_df.to_csv(output_dir / f"cap_templates_{safe_group}_long.csv", index=False)

        group_dfs[group_name] = df

    return group_dfs
