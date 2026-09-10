from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, t, wilcoxon


# ----------------- File path configuration -----------------
CLINICAL_PATH = Path("data/merged_clinical_metrics.csv")
CONFOUND_DIR = Path("data/confound_rename")
OUTPUT_DIR = Path("outputs/clinical_statistics")

SUBJECT_FD_PATH = OUTPUT_DIR / "Subject_FD.csv"
TABLE3_PATH = OUTPUT_DIR / "Table3_QC_MeanFD_Long_Format.csv"
QC_SUMMARY_PATH = OUTPUT_DIR / "FD_QC_Inclusion_Summary.csv"

FD_COLUMN = "framewise_displacement"
THRESHOLDS = (0.5, 0.7, 0.9, 1.0)
MEAN_FD_UPPER_LIMIT = 0.5
MIN_FD_BELOW_1_PERCENT = 98.0
# MIN_FD_BELOW_09_PERCENT = 95.0
CONFOUND_PATTERN = re.compile(
    r"^sub-(?P<subject_id>.+?)_task-rest_run-(?P<run>\d+)_"
    r"desc-confounds_timeseries\.tsv$"
)


# ----------------- Helper functions -----------------
def format_pval(p_value):
    """Format p-values for the long-format Table 3 output."""
    if pd.isna(p_value):
        return "-"
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.3f}"


def calc_p_ind(df1, df2, column):
    """Two-sided Mann-Whitney U test for independent samples."""
    sample1 = pd.to_numeric(df1[column], errors="coerce").dropna()
    sample2 = pd.to_numeric(df2[column], errors="coerce").dropna()

    if sample1.empty or sample2.empty:
        return np.nan
    if sample1.nunique() == sample2.nunique() == 1:
        if sample1.iloc[0] == sample2.iloc[0]:
            return 1.0

    return mannwhitneyu(sample1, sample2, alternative="two-sided").pvalue


def calc_p_paired(df, column_off, column_on):
    """Two-sided Wilcoxon signed-rank test for paired OFF/ON samples."""
    paired = df[[column_off, column_on]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()

    if len(paired) < 2:
        return np.nan
    if (paired[column_off] == paired[column_on]).all():
        return 1.0

    return wilcoxon(
        paired[column_off], paired[column_on], alternative="two-sided"
    ).pvalue


def get_stats_dict(group_name, df, column):
    """Return mean, sample SD, and t-based 95% CI for one group."""
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    n_subjects = len(values)

    if n_subjects == 0:
        return {
            "Group": f"{group_name}(N=0)",
            "Mean_FD": "-",
            "95% CI_low": "-",
            "95% CI_high": "-",
            "p": "-",
        }

    mean_value = values.mean()
    sd_value = values.std(ddof=1)

    if n_subjects > 1:
        standard_error = sd_value / np.sqrt(n_subjects)
        ci_low, ci_high = t.interval(
            0.95,
            df=n_subjects - 1,
            loc=mean_value,
            scale=standard_error,
        )
        ci_low_string = f"{ci_low:.4f}"
        ci_high_string = f"{ci_high:.4f}"
    else:
        ci_low_string = "-"
        ci_high_string = "-"

    sd_string = "-" if pd.isna(sd_value) else f"{sd_value:.4f}"
    return {
        "Group": f"{group_name}(N={n_subjects})",
        "Mean_FD": f"{mean_value:.4f} ± {sd_string}",
        "95% CI_low": ci_low_string,
        "95% CI_high": ci_high_string,
        "p": "-",
    }


def get_p_row(comparison_name, p_value):
    """Build one comparison row for the long-format Table 3 output."""
    return {
        "Group": comparison_name,
        "Mean_FD": "-",
        "95% CI_low": "-",
        "95% CI_high": "-",
        "p": format_pval(p_value),
    }


def load_and_classify_clinical_data():
    """Load clinical data and classify subjects as HC, PD-BO, or paired PD."""
    required_columns = {
        "subject_id",
        "Group",
        "H-Y(ON)",
        "UPDRS-III(ON)",
        "Responder",
    }
    clinical = pd.read_csv(CLINICAL_PATH, dtype={"subject_id": "string"})

    missing_columns = required_columns - set(clinical.columns)
    if missing_columns:
        raise ValueError(
            "Clinical file is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    clinical["subject_id"] = clinical["subject_id"].str.strip()
    if clinical["subject_id"].isna().any() or clinical["subject_id"].eq("").any():
        raise ValueError("Clinical file contains an empty subject_id.")
    if clinical["subject_id"].duplicated().any():
        duplicate_ids = clinical.loc[
            clinical["subject_id"].duplicated(keep=False), "subject_id"
        ].unique()
        raise ValueError(
            "Clinical file contains duplicate subject_id values: "
            + ", ".join(duplicate_ids)
        )

    invalid_groups = sorted(set(clinical["Group"].dropna()) - {"HC", "PD"})
    if invalid_groups or clinical["Group"].isna().any():
        raise ValueError(
            "Group must contain only HC or PD. Invalid values: "
            + ", ".join(map(str, invalid_groups))
        )

    has_on_data = clinical["H-Y(ON)"].notna() | clinical["UPDRS-III(ON)"].notna()
    clinical["analysis_group"] = ""
    clinical.loc[clinical["Group"].eq("HC"), "analysis_group"] = "HC"
    clinical.loc[
        clinical["Group"].eq("PD") & ~has_on_data, "analysis_group"
    ] = "PD-BO"
    clinical.loc[
        clinical["Group"].eq("PD") & has_on_data, "analysis_group"
    ] = "PD-paired"

    clinical["subject_order"] = np.arange(len(clinical))
    return clinical


def expected_subject_runs(clinical):
    """Return the subject/run combinations implied by the clinical grouping."""
    expected = set()
    for row in clinical.itertuples(index=False):
        runs = (1, 2) if row.analysis_group == "PD-paired" else (0,)
        expected.update((row.subject_id, run_number) for run_number in runs)
    return expected


def calculate_subject_fd(clinical):
    """Read every confounds file and calculate subject/run-level FD metrics."""
    records = []
    observed_subject_runs = set()
    dropped_fd_values = 0

    confound_files = sorted(CONFOUND_DIR.glob("*.tsv"))
    if not confound_files:
        raise FileNotFoundError(f"No TSV files found in {CONFOUND_DIR}.")

    clinical_ids = set(clinical["subject_id"])
    group_by_subject = clinical.set_index("subject_id")["analysis_group"].to_dict()

    for confound_path in confound_files:
        match = CONFOUND_PATTERN.fullmatch(confound_path.name)
        if match is None:
            raise ValueError(f"Unexpected confounds filename: {confound_path.name}")

        subject_id = match.group("subject_id")
        run_number = int(match.group("run"))
        subject_run = (subject_id, run_number)

        if subject_id not in clinical_ids:
            raise ValueError(
                f"Subject {subject_id} from {confound_path.name} is absent from "
                f"{CLINICAL_PATH}."
            )
        if subject_run in observed_subject_runs:
            raise ValueError(
                f"Duplicate confounds file for {subject_id}, run-{run_number}."
            )
        observed_subject_runs.add(subject_run)

        expected_runs = (
            {1, 2} if group_by_subject[subject_id] == "PD-paired" else {0}
        )
        if run_number not in expected_runs:
            raise ValueError(
                f"Unexpected run-{run_number} for {subject_id} "
                f"({group_by_subject[subject_id]}). Expected: "
                + ", ".join(f"run-{run}" for run in sorted(expected_runs))
            )

        try:
            confounds = pd.read_csv(
                confound_path, sep="\t", usecols=[FD_COLUMN]
            )
        except ValueError as error:
            raise ValueError(
                f"{confound_path.name} does not contain the {FD_COLUMN} column."
            ) from error

        raw_fd = confounds[FD_COLUMN]
        fd_values = pd.to_numeric(raw_fd, errors="coerce")
        invalid_count = int(fd_values.isna().sum())
        dropped_fd_values += invalid_count
        fd_values = fd_values.dropna()

        if fd_values.empty:
            raise ValueError(f"No valid FD values found in {confound_path.name}.")
        if (fd_values < 0).any():
            raise ValueError(f"Negative FD values found in {confound_path.name}.")

        record = {
            "subject_id": subject_id,
            "run_number": run_number,
            "run": f"run-{run_number}",
            "mean_FD": fd_values.mean(),
            "max_FD": fd_values.max(),
        }
        for threshold in THRESHOLDS:
            record[f"FD<{threshold:.1f} %"] = (fd_values < threshold).mean() * 100
        records.append(record)

    expected = expected_subject_runs(clinical)
    missing = expected - observed_subject_runs
    unexpected = observed_subject_runs - expected
    if missing or unexpected:
        details = []
        if missing:
            details.append(
                "missing: "
                + ", ".join(
                    f"{subject_id}/run-{run_number}"
                    for subject_id, run_number in sorted(missing)
                )
            )
        if unexpected:
            details.append(
                "unexpected: "
                + ", ".join(
                    f"{subject_id}/run-{run_number}"
                    for subject_id, run_number in sorted(unexpected)
                )
            )
        raise ValueError("Clinical/confounds run mismatch; " + "; ".join(details))

    subject_fd = pd.DataFrame(records)
    subject_fd = subject_fd.merge(
        clinical[["subject_id", "analysis_group", "subject_order"]],
        on="subject_id",
        how="left",
        validate="many_to_one",
    ).sort_values(["subject_order", "run_number"])

    subject_fd["session_group"] = subject_fd["analysis_group"]
    subject_fd.loc[
        subject_fd["analysis_group"].eq("PD-paired")
        & subject_fd["run_number"].eq(1),
        "session_group",
    ] = "PD-OFF"
    subject_fd.loc[
        subject_fd["analysis_group"].eq("PD-paired")
        & subject_fd["run_number"].eq(2),
        "session_group",
    ] = "PD-ON"

    # A single run passes QC only when both strict criteria are satisfied.
    subject_fd["direct_qc_pass"] = (
        subject_fd["mean_FD"] < MEAN_FD_UPPER_LIMIT
    ) & (subject_fd["FD<1.0 %"] > MIN_FD_BELOW_1_PERCENT)
    # ) & (subject_fd["FD<0.9 %"] > MIN_FD_BELOW_09_PERCENT)
    subject_fd["included"] = subject_fd["direct_qc_pass"]

    # OFF/ON subjects are retained as a pair. If either run fails, mark both
    # runs as excluded so all downstream paired analyses use identical IDs.
    paired_mask = subject_fd["analysis_group"].eq("PD-paired")
    paired_inclusion = (
        subject_fd.loc[paired_mask]
        .groupby("subject_id")["direct_qc_pass"]
        .all()
    )
    subject_fd.loc[paired_mask, "included"] = subject_fd.loc[
        paired_mask, "subject_id"
    ].map(paired_inclusion)

    if dropped_fd_values:
        print(
            f"Warning: excluded {dropped_fd_values} non-numeric or missing FD "
            "values from the calculations."
        )

    return subject_fd.reset_index(drop=True)


def build_table3(clinical, subject_fd):
    """Build Table 3 using only runs that pass the final QC inclusion rule."""
    included_fd = subject_fd[subject_fd["included"]].copy()
    fd_wide = included_fd.pivot(
        index="subject_id", columns="run", values="mean_FD"
    ).rename(columns=lambda run: f"Mean_FD_{run}")

    merged = clinical.merge(
        fd_wide,
        left_on="subject_id",
        right_index=True,
        how="left",
        validate="one_to_one",
    )
    merged["Mean_FD_baseline"] = merged.get(
        "Mean_FD_run-1", pd.Series(index=merged.index, dtype=float)
    ).fillna(
        merged.get("Mean_FD_run-0", pd.Series(index=merged.index, dtype=float))
    )

    hc = merged[merged["analysis_group"].eq("HC")].copy()
    pd_all = merged[merged["Group"].eq("PD")].copy()
    pd_paired = merged[merged["analysis_group"].eq("PD-paired")].dropna(
        subset=["Mean_FD_run-1", "Mean_FD_run-2"]
    ).copy()

    responder_numeric = pd.to_numeric(pd_paired["Responder"], errors="coerce")
    invalid_responder = responder_numeric.isna() | ~responder_numeric.isin([0, 1])
    if invalid_responder.any():
        invalid_ids = pd_paired.loc[invalid_responder, "subject_id"].tolist()
        raise ValueError(
            "Paired PD subjects must have Responder equal to 0 or 1. Invalid "
            "subject_id values: " + ", ".join(invalid_ids)
        )

    responders = pd_paired[responder_numeric.eq(1)].copy()
    non_responders = pd_paired[responder_numeric.eq(0)].copy()

    table_data = [
        get_stats_dict("PD", pd_all, "Mean_FD_baseline"),
        get_stats_dict("HC", hc, "Mean_FD_baseline"),
        get_p_row(
            "PD vs. HC",
            calc_p_ind(pd_all, hc, "Mean_FD_baseline"),
        ),
        get_stats_dict("PD-OFF", pd_paired, "Mean_FD_run-1"),
        get_stats_dict("PD-ON", pd_paired, "Mean_FD_run-2"),
        get_p_row(
            "PD-OFF vs. PD-ON",
            calc_p_paired(pd_paired, "Mean_FD_run-1", "Mean_FD_run-2"),
        ),
        get_stats_dict("Responder-OFF", responders, "Mean_FD_run-1"),
        get_stats_dict(
            "Non-responder-OFF", non_responders, "Mean_FD_run-1"
        ),
        get_p_row(
            "Responder-OFF vs. Non-responder-OFF",
            calc_p_ind(responders, non_responders, "Mean_FD_run-1"),
        ),
        get_stats_dict("Responder-ON", responders, "Mean_FD_run-2"),
        get_stats_dict(
            "Non-responder-ON", non_responders, "Mean_FD_run-2"
        ),
        get_p_row(
            "Responder-ON vs. Non-responder-ON",
            calc_p_ind(responders, non_responders, "Mean_FD_run-2"),
        ),
    ]
    return pd.DataFrame(table_data)


def build_qc_summary(clinical, subject_fd):
    """Summarize direct QC failures and final pairwise exclusions."""
    rows = []
    group_order = ("HC", "PD-BO", "PD-OFF", "PD-ON")

    for group_name in group_order:
        group_data = subject_fd[subject_fd["session_group"].eq(group_name)]
        total_count = len(group_data)
        direct_failed_count = int((~group_data["direct_qc_pass"]).sum())
        final_excluded_count = int((~group_data["included"]).sum())
        direct_failed_ids = group_data.loc[
            ~group_data["direct_qc_pass"], "subject_id"
        ].tolist()
        excluded_ids = group_data.loc[~group_data["included"], "subject_id"].tolist()

        rows.append(
            {
                "Group": group_name,
                "Direct_failed_count": direct_failed_count,
                "Final_excluded_count": final_excluded_count,
                "Total_count": total_count,
                "Direct_failed/Total": f"{direct_failed_count}/{total_count}",
                "Final_excluded/Total": f"{final_excluded_count}/{total_count}",
                "Final_included_count": total_count - final_excluded_count,
                "Direct_failed_IDs": "; ".join(direct_failed_ids),
                "Excluded_IDs": "; ".join(excluded_ids),
            }
        )

    paired_data = subject_fd[subject_fd["analysis_group"].eq("PD-paired")]
    paired_by_subject = paired_data.groupby("subject_id", sort=False).agg(
        direct_qc_pass=("direct_qc_pass", "all"),
        included=("included", "all"),
    )
    total_pairs = len(paired_by_subject)
    failed_pairs = int((~paired_by_subject["direct_qc_pass"]).sum())
    excluded_pairs = int((~paired_by_subject["included"]).sum())
    excluded_pair_ids = paired_by_subject.index[~paired_by_subject["included"]].tolist()
    rows.append(
        {
            "Group": "OFF-ON final pairs",
            "Direct_failed_count": failed_pairs,
            "Final_excluded_count": excluded_pairs,
            "Total_count": total_pairs,
            "Direct_failed/Total": f"{failed_pairs}/{total_pairs}",
            "Final_excluded/Total": f"{excluded_pairs}/{total_pairs}",
            "Final_included_count": total_pairs - excluded_pairs,
            "Direct_failed_IDs": "; ".join(excluded_pair_ids),
            "Excluded_IDs": "; ".join(excluded_pair_ids),
        }
    )

    excluded_subjects = (
        subject_fd.loc[~subject_fd["included"], ["subject_id", "subject_order"]]
        .drop_duplicates("subject_id")
        .sort_values("subject_order")
    )
    total_subjects = clinical["subject_id"].nunique()
    total_excluded = len(excluded_subjects)
    rows.append(
        {
            "Group": "All excluded IDs",
            "Direct_failed_count": pd.NA,
            "Final_excluded_count": total_excluded,
            "Total_count": total_subjects,
            "Direct_failed/Total": "-",
            "Final_excluded/Total": f"{total_excluded}/{total_subjects}",
            "Final_included_count": total_subjects - total_excluded,
            "Direct_failed_IDs": "-",
            "Excluded_IDs": "; ".join(excluded_subjects["subject_id"]),
        }
    )

    return pd.DataFrame(rows)


def save_subject_fd(subject_fd):
    """Save Subject_FD.csv with the requested columns and precision."""
    output_columns = [
        "subject_id",
        "run",
        "mean_FD",
        "max_FD",
        "FD<0.5 %",
        "FD<0.7 %",
        "FD<0.9 %",
        "FD<1.0 %",
        "included",
    ]
    formatted = subject_fd[output_columns].copy()
    for column in ("mean_FD", "max_FD"):
        formatted[column] = formatted[column].map(lambda value: f"{value:.6f}")
    for column in ("FD<0.5 %", "FD<0.7 %", "FD<0.9 %", "FD<1.0 %"):
        formatted[column] = formatted[column].map(lambda value: f"{value:.2f}")
    formatted["included"] = formatted["included"].map({True: "Yes", False: "No"})

    formatted.to_csv(SUBJECT_FD_PATH, index=False, encoding="utf-8-sig")


# ----------------- Main logic -----------------
def main_fd_analysis():
    if not CLINICAL_PATH.is_file():
        raise FileNotFoundError(f"Clinical file not found: {CLINICAL_PATH}")
    if not CONFOUND_DIR.is_dir():
        raise FileNotFoundError(f"Confounds directory not found: {CONFOUND_DIR}")

    clinical = load_and_classify_clinical_data()
    subject_fd = calculate_subject_fd(clinical)
    table3 = build_table3(clinical, subject_fd)
    qc_summary = build_qc_summary(clinical, subject_fd)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_subject_fd(subject_fd)
    table3.to_csv(TABLE3_PATH, index=False, encoding="utf-8-sig")
    qc_summary.to_csv(QC_SUMMARY_PATH, index=False, encoding="utf-8-sig")

    print(f"Subject-level FD statistics exported: {SUBJECT_FD_PATH}")
    print(f"FD QC inclusion summary exported: {QC_SUMMARY_PATH}")
    print(f"Table 3 FD statistics exported: {TABLE3_PATH}")
    print(f"Subject_FD rows: {len(subject_fd)}")
    print("-" * 75)
    print(qc_summary.to_string(index=False))
    print("-" * 75)
    print(table3.to_string(index=False))
    print("-" * 75)


if __name__ == "__main__":
    main_fd_analysis()
