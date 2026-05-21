from pathlib import Path
import math
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import permutation_test_score

POSITIVE_LABEL = 1  # Responder
NEGATIVE_LABEL = 0  # Non-responder


def _parse_feature_spec(feature_spec):
    spec = str(feature_spec).strip()
    if "::" in spec:
        metric_name, feature_name = spec.split("::", 1)
    else:
        metric_name, feature_name = spec, ""

    metric_name = metric_name.strip()
    feature_name = feature_name.strip()

    if not metric_name:
        raise ValueError(f"Invalid feature spec: {feature_spec}")

    return metric_name, feature_name


def _sanitize_name_for_path(name):
    keep = []
    for ch in str(name):
        if ch.isalnum() or ch in {"_", "-"}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def _prepare_modeling_table(data_csv, feature_specs, off_run="run-1", response_threshold_pct=30.0):
    """
    Reworked helper: read from the merged wide table and prepare modeling data.
    Uses response_threshold_pct to compute responder labels and keeps core columns only.
    """
    df = pd.read_csv(data_csv)

    # Filter OFF run and PD group
    df["subject_id"] = df["subject_id"].astype(str)
    if "Run" in df.columns:
        df = df[df["Run"].astype(str) == str(off_run)].copy()
        
    if "Group" in df.columns:
        df = df[df["Group"] == "PD"].copy()

    # Prepare labels and improvement metrics
    if "UPDRS-III(OFF)" in df.columns and "UPDRS-III(ON)" in df.columns:
        df["UPDRS-III(OFF)"] = pd.to_numeric(df["UPDRS-III(OFF)"], errors="coerce")
        df["UPDRS-III(ON)"] = pd.to_numeric(df["UPDRS-III(ON)"], errors="coerce")
        
        # Drop rows with missing scores
        df = df.dropna(subset=["UPDRS-III(OFF)", "UPDRS-III(ON)"]).copy()
        
        # Compute delta and percent change
        df["Delta_UPDRS_III_OFF"] = df["UPDRS-III(OFF)"] - df["UPDRS-III(ON)"]
        df["PctChange_UPDRS_III_OFF"] = (df["Delta_UPDRS_III_OFF"] / df["UPDRS-III(OFF)"]) * 100
        
        # Normalize threshold to percent
        threshold = response_threshold_pct * 100 if response_threshold_pct < 1 else response_threshold_pct
        
        # Generate label
        df["label"] = (df["PctChange_UPDRS_III_OFF"] >= threshold).astype(int)
        df["Response_Group"] = df["label"].map({1: "Responder", 0: "Non-responder"})
    else:
        return None, {"error": "Required columns for UPDRS-III(OFF) and UPDRS-III(ON) not found in data."}

    # Ensure key clinical columns are numeric
    for c in ["Age", "Gender", "Years of Education", "H-Y(OFF)"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if not feature_specs:
        return None, {"error": "feature_specs is empty."}

    # Resolve imaging features
    resolved_features = []
    for idx, spec in enumerate(feature_specs, start=1):
        metric_name, feature_name = _parse_feature_spec(spec)
        
        # Build expected wide-table column name
        if feature_name:
            source_col = f"{metric_name}_{feature_name}"
        else:
            source_col = f"{metric_name}"

        if source_col not in df.columns:
            return None, {
                "error": f"feature column '{source_col}' not found in data. "
            }

        model_col = f"feature_{idx}"
        df[model_col] = pd.to_numeric(df[source_col], errors="coerce")
        
        resolved_features.append(
            {
                "idx": idx,
                "spec": str(spec),
                "metric": metric_name,
                "feature": feature_name,
                "model_col": model_col,
            }
        )

    df = df.drop_duplicates(subset=["subject_id"], keep="first")

    # Keep core columns only
    target_cols = [
        "subject_id",
        "Delta_UPDRS_III_OFF",
        "PctChange_UPDRS_III_OFF",
        "Response_Group",
        "UPDRS-III(OFF)",
        "Age",
        "Gender",
        "Years of Education",
        "H-Y(OFF)"
    ]
    
    # Add feature_1, feature_2, ...
    feature_cols = [r["model_col"] for r in resolved_features]
    target_cols.extend(feature_cols)
    
    # Add label last
    target_cols.append("label")
    
    # Drop columns that are not present in the data
    final_cols = [c for c in target_cols if c in df.columns]
    df = df[final_cols]

    info = {
        "n_subjects_total": int(df["subject_id"].nunique()),
        "n_responder": int((df["label"] == POSITIVE_LABEL).sum()),
        "n_non_responder": int((df["label"] == NEGATIVE_LABEL).sum()),
        "off_run": str(off_run),
        "feature_specs": " | ".join([str(x) for x in feature_specs]),
        "resolved_feature_cols": " | ".join([f"{r['model_col']}<-{r['metric']}::{r['feature']}" for r in resolved_features]),
    }
    
    return df, info


def _evaluate_svm_cv(data_df, feature_cols, model_name, n_splits=5, random_state=6753):
    keep_cols = ["subject_id", "Response_Group", "label", *feature_cols]
    used_df = data_df[keep_cols].copy()
    used_df = used_df.dropna(subset=feature_cols + ["label"])

    if used_df.empty:
        return pd.DataFrame(), pd.DataFrame(), used_df

    for c in feature_cols:
        used_df[c] = pd.to_numeric(used_df[c], errors="coerce")
    used_df = used_df.dropna(subset=feature_cols)

    class_counts = used_df["label"].value_counts()
    if len(class_counts) < 2:
        raise ValueError(f"{model_name}: only one class available after filtering.")
    if int(class_counts.min()) < int(n_splits):
        raise ValueError(
            f"{model_name}: smallest class count ({int(class_counts.min())}) is less than n_splits ({n_splits})."
        )

    X = used_df[feature_cols].to_numpy(dtype=float)
    y = used_df["label"].to_numpy(dtype=int)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="linear",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=random_state,
                ),
            ),
        ]
    )

    fold_rows = []
    pred_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        x_train, x_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf.fit(x_train, y_train)

        y_pred = clf.predict(x_test)
        y_prob = clf.predict_proba(x_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        try:
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            auc = np.nan

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[NEGATIVE_LABEL, POSITIVE_LABEL]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

        fold_rows.append(
            {
                "Model": model_name,
                "Repeat": 0,
                "Fold": fold_idx,
                "N_train": int(len(train_idx)),
                "N_test": int(len(test_idx)),
                "Accuracy": float(acc),
                "AUC": float(auc) if np.isfinite(auc) else np.nan,
                "Sensitivity": float(sensitivity) if np.isfinite(sensitivity) else np.nan,
                "Specificity": float(specificity) if np.isfinite(specificity) else np.nan,
                "F1": float(f1),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
            }
        )

        fold_subjects = used_df.iloc[test_idx]["subject_id"].to_numpy()
        for sid, yy, yp, pp in zip(fold_subjects, y_test, y_pred, y_prob):
            pred_rows.append(
                {
                    "Model": model_name,
                    "Repeat": 0,
                    "Fold": fold_idx,
                    "subject_id": str(sid),
                    "y_true": int(yy),
                    "y_pred": int(yp),
                    "y_prob": float(pp),
                }
            )

    return pd.DataFrame(fold_rows), pd.DataFrame(pred_rows), used_df


def _find_best_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden = tpr - fpr
    best_idx = int(np.argmax(youden))
    return float(thresholds[best_idx])


def _evaluate_svm_nested_cv(
    data_df,
    feature_cols,
    model_name,
    n_splits=5,
    n_repeats=5,
    random_state=6753,
):
    keep_cols = ["subject_id", "Response_Group", "label", *feature_cols]
    used_df = data_df[keep_cols].copy().dropna(subset=feature_cols + ["label"])

    if used_df.empty:
        return pd.DataFrame(), pd.DataFrame(), used_df

    for c in feature_cols:
        used_df[c] = pd.to_numeric(used_df[c], errors="coerce")
    used_df = used_df.dropna(subset=feature_cols)

    class_counts = used_df["label"].value_counts()
    if len(class_counts) < 2:
        raise ValueError(f"{model_name}: only one class available after filtering.")
    if int(class_counts.min()) < int(n_splits):
        raise ValueError(
            f"{model_name}: smallest class count ({int(class_counts.min())}) is less than n_splits ({n_splits})."
        )

    X = used_df[feature_cols].to_numpy(dtype=float)
    y = used_df["label"].to_numpy(dtype=int)

    all_fold_rows = []
    all_pred_rows = []

    for repeat in range(n_repeats):
        outer_cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state + repeat,
        )

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
            x_train, x_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            pipe = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "svm",
                        SVC(
                            class_weight="balanced",
                            probability=True,
                            random_state=random_state,
                        ),
                    ),
                ]
            )

            param_grid = {
                "svm__kernel": ["linear", "rbf"],
                "svm__C": [0.01, 0.1, 1, 10],
                "svm__gamma": ["scale", 0.1, 1],
            }

            inner_cv = StratifiedKFold(
                n_splits=3,
                shuffle=True,
                random_state=random_state,
            )

            grid = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid,
                cv=inner_cv,
                scoring="roc_auc",
                n_jobs=-1,
            )
            grid.fit(x_train, y_train)
            best_model = grid.best_estimator_

            train_prob = best_model.predict_proba(x_train)[:, 1]
            best_threshold = _find_best_threshold(y_train, train_prob)

            y_prob = best_model.predict_proba(x_test)[:, 1]
            y_pred = (y_prob >= best_threshold).astype(int)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else np.nan

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[NEGATIVE_LABEL, POSITIVE_LABEL]).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

            all_fold_rows.append(
                {
                    "Model": model_name,
                    "Repeat": int(repeat),
                    "Fold": int(fold_idx),
                    "N_train": int(len(train_idx)),
                    "N_test": int(len(test_idx)),
                    "Accuracy": float(acc),
                    "AUC": float(auc) if np.isfinite(auc) else np.nan,
                    "Sensitivity": float(sensitivity) if np.isfinite(sensitivity) else np.nan,
                    "Specificity": float(specificity) if np.isfinite(specificity) else np.nan,
                    "F1": float(f1),
                    "BestThreshold": float(best_threshold),
                    "BestParams": str(grid.best_params_),
                    "TN": int(tn),
                    "FP": int(fp),
                    "FN": int(fn),
                    "TP": int(tp),
                }
            )

            fold_subjects = used_df.iloc[test_idx]["subject_id"].to_numpy()
            for sid, yy, yp, pp in zip(fold_subjects, y_test, y_pred, y_prob):
                all_pred_rows.append(
                    {
                        "Model": model_name,
                        "Repeat": int(repeat),
                        "Fold": int(fold_idx),
                        "subject_id": str(sid),
                        "y_true": int(yy),
                        "y_pred": int(yp),
                        "y_prob": float(pp),
                        "best_threshold": float(best_threshold),
                    }
                )

    return pd.DataFrame(all_fold_rows), pd.DataFrame(all_pred_rows), used_df


def _summarize_fold_metrics(fold_df):
    metrics = ["Accuracy", "AUC", "Sensitivity", "Specificity", "F1"]
    rows = []

    if fold_df.empty:
        return pd.DataFrame(columns=["Model", "Metric", "N", "Mean", "Std", "SE", "CI95_Low", "CI95_High"])

    for model_name in fold_df["Model"].dropna().unique():
        sub = fold_df[fold_df["Model"] == model_name].copy()
        n_folds = len(sub)

        for metric in metrics:
            vals = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                mean_val = np.nan
                std_val = np.nan
                se_val = np.nan
                ci = np.nan
            else:
                mean_val = float(np.mean(vals))
                std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                se_val = float(std_val / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
                ci = float(1.96 * se_val) if len(vals) > 1 else 0.0

            rows.append(
                {
                    "Model": model_name,
                    "Metric": metric,
                    "N_Folds": n_folds,
                    "N": len(vals),
                    "Mean": mean_val,
                    "Std": std_val,
                    "SE": se_val,
                    "CI95_Low": mean_val - ci if np.isfinite(mean_val) and np.isfinite(ci) else np.nan,
                    "CI95_High": mean_val + ci if np.isfinite(mean_val) and np.isfinite(ci) else np.nan,
                }
            )

    return pd.DataFrame(rows)


def _build_model_comparison(summary_df, baseline_name, enhanced_name):
    if summary_df.empty:
        return pd.DataFrame(columns=["Metric", "Baseline", "Enhanced", "Delta_Enhanced_minus_Baseline"])

    base = summary_df[summary_df["Model"] == baseline_name][["Metric", "Mean"]].rename(columns={"Mean": "Baseline"})
    enh = summary_df[summary_df["Model"] == enhanced_name][["Metric", "Mean"]].rename(columns={"Mean": "Enhanced"})
    cmp_df = pd.merge(base, enh, on="Metric", how="inner")
    cmp_df["Delta_Enhanced_minus_Baseline"] = cmp_df["Enhanced"] - cmp_df["Baseline"]
    return cmp_df


def _safe_binary_metrics(y_true, y_pred, y_prob):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[NEGATIVE_LABEL, POSITIVE_LABEL]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_prob)
    else:
        auc = np.nan

    return {
        "Accuracy": float(acc),
        "AUC": float(auc) if np.isfinite(auc) else np.nan,
        "Sensitivity": float(sensitivity) if np.isfinite(sensitivity) else np.nan,
        "Specificity": float(specificity) if np.isfinite(specificity) else np.nan,
        "F1": float(f1),
    }


def _align_prediction_pairs(base_preds, new_preds, pair_label, baseline_model, enhanced_model):
    key_cols = ["Repeat", "Fold", "subject_id"]

    left = base_preds.copy()
    right = new_preds.copy()

    for c in key_cols:
        if c in left.columns:
            left[c] = left[c].astype(str)
        if c in right.columns:
            right[c] = right[c].astype(str)

    merged = pd.merge(
        left[key_cols + ["y_true", "y_pred", "y_prob"]],
        right[key_cols + ["y_true", "y_pred", "y_prob"]],
        on=key_cols,
        how="inner",
        suffixes=("_base", "_new"),
    )

    if merged.empty:
        return pd.DataFrame(
            columns=[
                "Pair",
                "BaselineModel",
                "EnhancedModel",
                "Repeat",
                "Fold",
                "subject_id",
                "y_true",
                "y_pred_base",
                "y_prob_base",
                "y_pred_new",
                "y_prob_new",
            ]
        )

    # Prefer baseline y_true if there is a mismatch
    merged["y_true"] = merged["y_true_base"].astype(int)
    merged = merged.drop(columns=["y_true_base", "y_true_new"])
    merged["Pair"] = pair_label
    merged["BaselineModel"] = baseline_model
    merged["EnhancedModel"] = enhanced_model
    return merged


def bootstrap_metric_diffs(aligned_pred_df, n_boot=1000, random_state=6753):
    metrics = ["Accuracy", "AUC", "Sensitivity", "Specificity", "F1"]
    out_rows = []

    if aligned_pred_df is None or aligned_pred_df.empty:
        for metric in metrics:
            out_rows.append(
                {
                    "Pair": np.nan,
                    "BaselineModel": np.nan,
                    "EnhancedModel": np.nan,
                    "Metric": metric,
                    "N": 0,
                    "n_boot_used": 0,
                    "mean_diff": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                }
            )
        return pd.DataFrame(out_rows)

    y_true = aligned_pred_df["y_true"].to_numpy(dtype=int)
    y_pred_base = aligned_pred_df["y_pred_base"].to_numpy(dtype=int)
    y_prob_base = aligned_pred_df["y_prob_base"].to_numpy(dtype=float)
    y_pred_new = aligned_pred_df["y_pred_new"].to_numpy(dtype=int)
    y_prob_new = aligned_pred_df["y_prob_new"].to_numpy(dtype=float)

    n = len(y_true)
    rng = np.random.default_rng(random_state)
    boot_diffs = {m: [] for m in metrics}

    for _ in range(int(n_boot)):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        yb = y_true[idx]
        base_m = _safe_binary_metrics(yb, y_pred_base[idx], y_prob_base[idx])
        new_m = _safe_binary_metrics(yb, y_pred_new[idx], y_prob_new[idx])

        for metric in metrics:
            vb = base_m[metric]
            vn = new_m[metric]
            if np.isfinite(vb) and np.isfinite(vn):
                boot_diffs[metric].append(float(vn - vb))

    pair_label = aligned_pred_df["Pair"].iloc[0]
    baseline_model = aligned_pred_df["BaselineModel"].iloc[0]
    enhanced_model = aligned_pred_df["EnhancedModel"].iloc[0]

    for metric in metrics:
        vals = np.asarray(boot_diffs[metric], dtype=float)
        if vals.size == 0:
            out_rows.append(
                {
                    "Pair": pair_label,
                    "BaselineModel": baseline_model,
                    "EnhancedModel": enhanced_model,
                    "Metric": metric,
                    "N": int(n),
                    "n_boot_used": 0,
                    "mean_diff": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                }
            )
            continue

        out_rows.append(
            {
                "Pair": pair_label,
                "BaselineModel": baseline_model,
                "EnhancedModel": enhanced_model,
                "Metric": metric,
                "N": int(n),
                "n_boot_used": int(vals.size),
                "mean_diff": float(np.mean(vals)),
                "ci_low": float(np.percentile(vals, 2.5)),
                "ci_high": float(np.percentile(vals, 97.5)),
            }
        )

    return pd.DataFrame(out_rows)


def compute_nri(aligned_pred_df):
    if aligned_pred_df is None or aligned_pred_df.empty:
        return {
            "NRI": np.nan,
            "NRI_Z": np.nan,
            "NRI_p": np.nan,
            "NRI_Event": np.nan,
            "NRI_NonEvent": np.nan,
            "N_Event": 0,
            "N_NonEvent": 0,
            "N_Total": 0,
        }

    y_true = aligned_pred_df["y_true"].to_numpy(dtype=int)
    p_base = aligned_pred_df["y_prob_base"].to_numpy(dtype=float)
    p_new = aligned_pred_df["y_prob_new"].to_numpy(dtype=float)

    valid_mask = np.isfinite(y_true) & np.isfinite(p_base) & np.isfinite(p_new)
    y_true = y_true[valid_mask]
    p_base = p_base[valid_mask]
    p_new = p_new[valid_mask]

    if y_true.size == 0:
        return {
            "NRI": np.nan,
            "NRI_Z": np.nan,
            "NRI_p": np.nan,
            "NRI_Event": np.nan,
            "NRI_NonEvent": np.nan,
            "N_Event": 0,
            "N_NonEvent": 0,
            "N_Total": 0,
        }

    is_event = y_true == POSITIVE_LABEL
    is_nonevent = y_true == NEGATIVE_LABEL

    n_event = int(np.sum(is_event))
    n_nonevent = int(np.sum(is_nonevent))

    up = p_new > p_base
    down = p_new < p_base

    if n_event > 0:
        p_up_event = float(np.mean(up[is_event]))
        p_down_event = float(np.mean(down[is_event]))
        nri_event = p_up_event - p_down_event
        var_event = (p_up_event + p_down_event - (nri_event**2)) / n_event
    else:
        nri_event = np.nan
        var_event = np.nan

    if n_nonevent > 0:
        p_down_nonevent = float(np.mean(down[is_nonevent]))
        p_up_nonevent = float(np.mean(up[is_nonevent]))
        nri_nonevent = p_down_nonevent - p_up_nonevent
        var_nonevent = (p_down_nonevent + p_up_nonevent - (nri_nonevent**2)) / n_nonevent
    else:
        nri_nonevent = np.nan
        var_nonevent = np.nan

    if np.isfinite(nri_event) and np.isfinite(nri_nonevent):
        nri_total = float(nri_event + nri_nonevent)
    else:
        nri_total = np.nan

    if np.isfinite(var_event) and np.isfinite(var_nonevent) and (var_event + var_nonevent) > 0 and np.isfinite(nri_total):
        nri_z = float(nri_total / math.sqrt(var_event + var_nonevent))
        # Two-sided p-value from standard normal distribution
        nri_p = float(math.erfc(abs(nri_z) / math.sqrt(2.0)))
    else:
        nri_z = np.nan
        nri_p = np.nan

    return {
        "NRI": nri_total,
        "NRI_Z": nri_z,
        "NRI_p": nri_p,
        "NRI_Event": float(nri_event) if np.isfinite(nri_event) else np.nan,
        "NRI_NonEvent": float(nri_nonevent) if np.isfinite(nri_nonevent) else np.nan,
        "N_Event": n_event,
        "N_NonEvent": n_nonevent,
        "N_Total": int(y_true.size),
    }


def bootstrap_nri(aligned_pred_df, n_boot=1000, random_state=6753):
    if aligned_pred_df is None or aligned_pred_df.empty:
        return {
            "N": 0,
            "n_boot_used": 0,
            "mean_diff": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }

    n = int(len(aligned_pred_df))
    rng = np.random.default_rng(random_state)
    boot_vals = []

    for _ in range(int(n_boot)):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        sample_df = aligned_pred_df.iloc[idx].copy()
        nri_val = compute_nri(sample_df).get("NRI", np.nan)
        if np.isfinite(nri_val):
            boot_vals.append(float(nri_val))

    if not boot_vals:
        return {
            "N": n,
            "n_boot_used": 0,
            "mean_diff": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }

    vals = np.asarray(boot_vals, dtype=float)
    return {
        "N": n,
        "n_boot_used": int(vals.size),
        "mean_diff": float(np.mean(vals)),
        "ci_low": float(np.percentile(vals, 2.5)),
        "ci_high": float(np.percentile(vals, 97.5)),
    }


def _merge_comparison_bootstrap_nri(comparison_df, bootstrap_df, nri_rows_df):
    if comparison_df is None:
        comparison_df = pd.DataFrame()
    if bootstrap_df is None:
        bootstrap_df = pd.DataFrame()
    if nri_rows_df is None:
        nri_rows_df = pd.DataFrame()

    merge_keys = ["Pair", "BaselineModel", "EnhancedModel", "Metric"]
    merged = pd.merge(comparison_df, bootstrap_df, on=merge_keys, how="outer")

    if not nri_rows_df.empty:
        merged = pd.concat([merged, nri_rows_df], ignore_index=True)

    preferred_cols = [
        "Pair",
        "BaselineModel",
        "EnhancedModel",
        "Metric",
        "Baseline",
        "Enhanced",
        "Delta_Enhanced_minus_Baseline",
        "N",
        "n_boot_used",
        "mean_diff",
        "ci_low",
        "ci_high",
        "NRI",
        "NRI_Z",
        "NRI_p",
        "NRI_Event",
        "NRI_NonEvent",
        "N_Event",
        "N_NonEvent",
        "N_Total",
    ]
    cols = [c for c in preferred_cols if c in merged.columns] + [c for c in merged.columns if c not in preferred_cols]
    return merged[cols]


def _plot_roc_curves(model_preds, out_path):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    plotted_any = False
    for pred_df, model_name, color in model_preds:
        if pred_df is None or pred_df.empty:
            continue

        y_true = pred_df["y_true"].to_numpy(dtype=int)
        y_prob = pred_df["y_prob"].to_numpy(dtype=float)

        if len(np.unique(y_true)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{model_name} (AUC={auc:.3f})")
        plotted_any = True

    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("OOF ROC Curves (5-fold CV)")
    ax.grid(True, linestyle="--", alpha=0.3)
    if plotted_any:
        ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

from sklearn.model_selection import RepeatedStratifiedKFold
def run_permutation_test_for_model(data_df, feature_cols, model_name, n_splits=5, n_repeats=5, n_permutations=1000, random_state=6753, is_nested=True):
    keep_cols = ["subject_id", "label", *feature_cols]
    used_df = data_df[keep_cols].copy().dropna()

    if used_df.empty:
        return None

    for c in feature_cols:
        used_df[c] = pd.to_numeric(used_df[c], errors="coerce")
    used_df = used_df.dropna(subset=feature_cols)

    X = used_df[feature_cols].to_numpy(dtype=float)
    y = used_df["label"].to_numpy(dtype=int)

    if is_nested:
        pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("svm", SVC(class_weight="balanced", probability=True, random_state=random_state)),
            ]
        )
        param_grid = {
            "svm__kernel": ["linear", "rbf"],
            "svm__C": [0.01, 0.1, 1, 10],
            "svm__gamma": ["scale", 0.1, 1],
        }
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        estimator = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=inner_cv, scoring="roc_auc", n_jobs=1)
    else:
        estimator = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "svm",
                    SVC(kernel="linear", C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=random_state),
                ),
            ]
        )

    outer_cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    print(f"\n[INFO] Running {n_permutations} Permutation Tests for {model_name}... (this may take a while)")
    
    score, permutation_scores, pvalue = permutation_test_score(
        estimator=estimator,
        X=X,
        y=y,
        scoring="roc_auc",
        cv=outer_cv,
        n_permutations=n_permutations,
        n_jobs=-1,
        random_state=random_state,
        verbose=True,
    )

    print(f"[INFO] Permutation Test complete! True AUC: {score:.4f}, p-value: {pvalue:.4f}")

    return {
        "Model": model_name,
        "True_AUC": score,
        "p_value": pvalue,
        "Permutation_Scores": permutation_scores
    }

def plot_permutation_distribution(perm_results, out_path):
    if not perm_results:
        return

    true_auc = perm_results["True_AUC"]
    perm_scores = perm_results["Permutation_Scores"]
    p_value = perm_results["p_value"]
    model_name = perm_results["Model"]

    color_hist_fill = "#77b7f0"
    color_hist_edge = "#5364c0"
    color_true_line = "#c34fa2"
    color_true_text = "#77295d"

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    
    ax.hist(perm_scores, bins=20, density=True, color=color_hist_fill, alpha=0.7, 
            edgecolor=color_hist_edge, linewidth=1.2, label="Null distribution")
    
    ax.axvline(true_auc, color=color_true_line, linestyle='--', linewidth=2.5, 
               label=f"True AUC ({true_auc:.3f})")
    
    ax.text(true_auc * 0.95, ax.get_ylim()[1] * 0.8, f"Empirical p = {p_value:.4f}", 
            color=color_true_text, fontsize=12, fontweight='bold')

    ax.set_xlabel("AUC Score")
    ax.set_ylabel("Density")
    ax.set_title(f"Permutation Test: {model_name}\n(1000 Iterations)")
    
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.3, color="#5364c0")
    
    # ax.set_facecolor("#edf1fd") 

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_svm_response_prediction_pipeline(
    data_dir,
    output_dir,
    feature_specs=None,
    off_run="run-1",
    response_threshold_pct=30.0,
    n_splits=5,
    n_repeats=5,
    n_bootstrap=1000,
    random_state=6753,
    include_covariates=True,
    include_hy_stage=False,
    train_evaluator="_evaluate_svm_nested_cv",
    permutation_test=False,
    models_compare=True,
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if feature_specs is None:
        feature_specs = ["counts::CAP-2", "transition_probability::2.5"]

    model_df, info = _prepare_modeling_table(
        data_csv=str(data_dir / "01_merged_all_runs.csv"),
        feature_specs=feature_specs,
        off_run=off_run,
        response_threshold_pct=response_threshold_pct,
    )

    if model_df is None or model_df.empty:
        summary_path = out_dir / "cv_summary.csv"
        pd.DataFrame().to_csv(summary_path, index=False)
        print("[WARN] Failed to prepare modeling table.")
        if isinstance(info, dict) and "error" in info:
            print(f"[WARN] {info['error']}")
        return {
            "out_dir": out_dir,
            "summary_csv": summary_path,
        }

    model_df.to_csv(out_dir / "labels_used.csv", index=False)
    pd.DataFrame([info]).to_csv(out_dir / "dataset_summary.csv", index=False)

    base_feature_cols = ["UPDRS-III(OFF)"]
    covariate_candidates = ["Age", "Gender", "Years of Education"]
    if include_covariates:
        missing_covariates = [c for c in covariate_candidates if c not in model_df.columns]
        if missing_covariates:
            missing_text = ", ".join(missing_covariates)
            raise ValueError(f"Missing required covariates: {missing_text}")
        base_feature_cols.extend(covariate_candidates)

    if include_hy_stage:
        if "H-Y(OFF)" not in model_df.columns:
            raise ValueError("include_hy_stage=True but H-Y(OFF) is not available.")
        if "H-Y(OFF)" not in base_feature_cols:
            base_feature_cols.append("H-Y(OFF)")

    n_input_features = int(len(feature_specs))
    feature_cols_by_idx = {idx: f"feature_{idx}" for idx in range(1, n_input_features + 1)}

    model_defs = [  
        {
            "name": "base",
            "feature_idxs": tuple(),
            "feature_cols": list(base_feature_cols),
        }
    ]

    for r in range(1, n_input_features + 1):
        for combo in itertools.combinations(range(1, n_input_features + 1), r):
            model_name = "base_" + "_".join([str(i) for i in combo])
            combo_cols = list(base_feature_cols) + [feature_cols_by_idx[i] for i in combo]
            model_defs.append(
                {
                    "name": model_name,
                    "feature_idxs": tuple(combo),
                    "feature_cols": combo_cols,
                }
            )

    model_defs = sorted(model_defs, key=lambda x: (len(x["feature_idxs"]), x["feature_idxs"]))

    eval_key = str(train_evaluator).strip().lower()
    if eval_key in {"_evaluate_svm_nested_cv", "nested", "nested_cv"}:
        evaluator_name = "_evaluate_svm_nested_cv"

        def _run_eval(data_df, feature_cols, model_name):
            return _evaluate_svm_nested_cv(
                data_df=data_df,
                feature_cols=feature_cols,
                model_name=model_name,
                n_splits=n_splits,
                n_repeats=n_repeats,
                random_state=random_state,
            )

    elif eval_key in {"_evaluate_svm_cv", "cv", "simple_cv"}:
        evaluator_name = "_evaluate_svm_cv"

        def _run_eval(data_df, feature_cols, model_name):
            return _evaluate_svm_cv(
                data_df=data_df,
                feature_cols=feature_cols,
                model_name=model_name,
                n_splits=n_splits,
                random_state=random_state,
            )

    else:
        raise ValueError(
            "train_evaluator must be one of {'_evaluate_svm_nested_cv', 'nested', 'nested_cv', "
            "'_evaluate_svm_cv', 'cv', 'simple_cv'}"
        )


    folds_by_model = {}
    preds_by_model = {}
    used_by_model = {}
    fold_paths = {}
    pred_paths = {}
    used_rows = []

    total_models = len(model_defs)
    print(f"[INFO] Building {total_models} models (including base).")
    for idx, model_def in enumerate(model_defs, start=1):
        model_name = model_def["name"]
        model_cols = model_def["feature_cols"]
        feature_tag = "none" if not model_def["feature_idxs"] else ",".join([str(i) for i in model_def["feature_idxs"]])
        print(f"[INFO] [{idx}/{total_models}] Training {model_name} | feature_idxs={feature_tag}")

        folds_df, preds_df, used_df = _run_eval(model_df, model_cols, model_name)
        folds_by_model[model_name] = folds_df
        preds_by_model[model_name] = preds_df
        used_by_model[model_name] = used_df

        safe_name = _sanitize_name_for_path(model_name)
        fold_path = out_dir / f"{safe_name}_cv_folds.csv"
        pred_path = out_dir / f"{safe_name}_oof_predictions.csv"
        folds_df.to_csv(fold_path, index=False)
        preds_df.to_csv(pred_path, index=False)
        fold_paths[model_name] = fold_path
        pred_paths[model_name] = pred_path

        used_rows.append(
            {
                "Model": model_name,
                "N_subjects": int(used_df["subject_id"].nunique()) if not used_df.empty else 0,
                "N_responder": int((used_df["label"] == POSITIVE_LABEL).sum()) if not used_df.empty else 0,
                "N_non_responder": int((used_df["label"] == NEGATIVE_LABEL).sum()) if not used_df.empty else 0,
                "Feature_Idxs": feature_tag,
                "Features": ", ".join(model_cols),
            }
        )

    used_summary = pd.DataFrame(used_rows)
    used_summary_path = out_dir / "model_sample_summary.csv"
    used_summary.to_csv(used_summary_path, index=False)

    fold_frames = [df for df in folds_by_model.values() if df is not None and not df.empty]
    if not fold_frames:
        raise ValueError("All model fold results are empty; cannot summarize CV metrics.")
    fold_all_df = pd.concat(fold_frames, ignore_index=True)
    summary_df = _summarize_fold_metrics(fold_all_df)
    summary_path = out_dir / "cv_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    if models_compare:
        # Compare all subset->superset model pairs.
        pair_specs = []
        for i in range(len(model_defs)):
            base_def = model_defs[i]
            base_set = set(base_def["feature_idxs"])
            for j in range(i + 1, len(model_defs)):
                new_def = model_defs[j]
                new_set = set(new_def["feature_idxs"])
                if base_set.issubset(new_set) and len(new_set) > len(base_set):
                    pair_specs.append(
                        {
                            "pair": f"{new_def['name']}__vs__{base_def['name']}",
                            "baseline": base_def["name"],
                            "enhanced": new_def["name"],
                            "base_preds": preds_by_model[base_def["name"]],
                            "new_preds": preds_by_model[new_def["name"]],
                        }
                    )

        print(f"[INFO] Total comparison pairs (subset->superset): {len(pair_specs)}")

        comparison_rows_all = []
        bootstrap_rows_all = []
        nri_rows_all = []

        total_pairs = len(pair_specs)
        for pair_idx, spec in enumerate(pair_specs, start=1):
            pair = spec["pair"]
            baseline = spec["baseline"]
            enhanced = spec["enhanced"]
            print(f"[INFO] [Pair {pair_idx}/{total_pairs}] Comparing {enhanced} vs {baseline}")

            cmp_df = _build_model_comparison(summary_df, baseline, enhanced)
            cmp_df.insert(0, "Pair", pair)
            cmp_df.insert(1, "BaselineModel", baseline)
            cmp_df.insert(2, "EnhancedModel", enhanced)
            comparison_rows_all.append(cmp_df)

            aligned = _align_prediction_pairs(
                base_preds=spec["base_preds"],
                new_preds=spec["new_preds"],
                pair_label=pair,
                baseline_model=baseline,
                enhanced_model=enhanced,
            )

            print(f"[INFO] [Pair {pair_idx}/{total_pairs}] Bootstrap 5 metrics (n_boot={n_bootstrap})")
            boot_df = bootstrap_metric_diffs(
                aligned_pred_df=aligned,
                n_boot=n_bootstrap,
                random_state=random_state,
            )
            bootstrap_rows_all.append(boot_df)

            nri = compute_nri(aligned)
            print(f"[INFO] [Pair {pair_idx}/{total_pairs}] Bootstrap NRI (n_boot={n_bootstrap})")
            nri_boot = bootstrap_nri(
                aligned_pred_df=aligned,
                n_boot=n_bootstrap,
                random_state=random_state,
            )
            nri_rows_all.append(
                {
                    "Pair": pair,
                    "BaselineModel": baseline,
                    "EnhancedModel": enhanced,
                    "Metric": "NRI",
                    "Baseline": np.nan,
                    "Enhanced": np.nan,
                    "Delta_Enhanced_minus_Baseline": nri["NRI"],
                    "N": nri_boot["N"],
                    "n_boot_used": nri_boot["n_boot_used"],
                    "mean_diff": nri_boot["mean_diff"],
                    "ci_low": nri_boot["ci_low"],
                    "ci_high": nri_boot["ci_high"],
                    "NRI": nri["NRI"],
                    "NRI_Z": nri["NRI_Z"],
                    "NRI_p": nri["NRI_p"],
                    "NRI_Event": nri["NRI_Event"],
                    "NRI_NonEvent": nri["NRI_NonEvent"],
                    "N_Event": nri["N_Event"],
                    "N_NonEvent": nri["N_NonEvent"],
                    "N_Total": nri["N_Total"],
                }
            )

        if comparison_rows_all:
            comparison_all_df = pd.concat(comparison_rows_all, ignore_index=True)
        else:
            comparison_all_df = pd.DataFrame(
                columns=[
                    "Pair",
                    "BaselineModel",
                    "EnhancedModel",
                    "Metric",
                    "Baseline",
                    "Enhanced",
                    "Delta_Enhanced_minus_Baseline",
                ]
            )

        if bootstrap_rows_all:
            bootstrap_all_df = pd.concat(bootstrap_rows_all, ignore_index=True)
        else:
            bootstrap_all_df = pd.DataFrame(
                columns=[
                    "Pair",
                    "BaselineModel",
                    "EnhancedModel",
                    "Metric",
                    "N",
                    "n_boot_used",
                    "mean_diff",
                    "ci_low",
                    "ci_high",
                ]
            )

        nri_all_df = pd.DataFrame(nri_rows_all)
        integrated_results_df = _merge_comparison_bootstrap_nri(
            comparison_df=comparison_all_df,
            bootstrap_df=bootstrap_all_df,
            nri_rows_df=nri_all_df,
        )
        integrated_results_path = out_dir / "model_comparison_bootstrap_nri_all.csv"
        integrated_results_df.to_csv(integrated_results_path, index=False)

        roc_path = out_dir / "oof_roc_curves.png"
        color_cycle = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
        model_preds = []
        for idx, model_def in enumerate(model_defs):
            model_name = model_def["name"]
            model_preds.append((preds_by_model[model_name], model_name, color_cycle[idx % len(color_cycle)]))
        _plot_roc_curves(
            model_preds=model_preds,
            out_path=roc_path,
        )

    if permutation_test:
        final_model_def = model_defs[-1] 
        
        perm_results = run_permutation_test_for_model(
            data_df=model_df,
            feature_cols=final_model_def["feature_cols"],
            model_name=final_model_def["name"],
            n_splits=n_splits,
            n_repeats=n_repeats,
            n_permutations=1000,
            random_state=random_state
        )

        if perm_results:
            perm_summary_df = pd.DataFrame([{
                "Model": perm_results["Model"],
                "True_AUC": perm_results["True_AUC"],
                "p_value": perm_results["p_value"]
            }])
            perm_summary_path = out_dir / "permutation_test_summary.csv"
            perm_summary_df.to_csv(perm_summary_path, index=False)
            
            pd.DataFrame({"Permuted_AUC": perm_results["Permutation_Scores"]}).to_csv(out_dir / "permutation_null_distribution.csv", index=False)

            plot_permutation_distribution(perm_results, out_path=out_dir / "permutation_test_distribution.png")

    print(f"[INFO] SVM response prediction complete. Output dir: {out_dir}")
    print(
        f"[INFO] Evaluator: {evaluator_name}; n_splits={n_splits}; "
        f"n_repeats={n_repeats if evaluator_name == '_evaluate_svm_nested_cv' else 'N/A'}"
    )
    print(f"[INFO] Base feature columns: {base_feature_cols}")
    print(f"[INFO] Input feature specs: {feature_specs}")
    print(f"[INFO] Models trained: {len(model_defs)}")