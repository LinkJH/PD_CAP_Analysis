#!/usr/bin/env python3
"""Merge NeuroCAPs metric CSV files into one long-format table.

Input directory default:
    outputs/neurocaps_full/03_metrics

Output file default:
    outputs/neurocaps_full/03_metrics/all_metrics_long.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

METRIC_KEYWORDS = {
    "counts": "counts",
    "persistence": "persistence",
    "temporal_fraction": "temporal_fraction",
    "transition_frequency": "transition_frequency",
    "transition_probability": "transition_probability",
}

OUTPUT_HEADER = ["Subject_ID", "Group", "Run", "Metric", "Feature", "Value"]


def infer_group(subject_id: str) -> str:
    """Infer HC/PD from Subject_ID string."""
    return "HC" if "hc" in subject_id.lower() else "PD"


def detect_metric_from_name(file_name: str) -> str | None:
    lowered = file_name.lower()
    for keyword, metric_name in METRIC_KEYWORDS.items():
        if keyword in lowered:
            return metric_name
    return None


def collect_metric_files(input_dir: Path) -> dict[str, Path]:
    metric_files: dict[str, Path] = {}
    for path in sorted(input_dir.glob("*.csv")):
        metric = detect_metric_from_name(path.name)
        if metric is None:
            continue
        if metric in metric_files:
            raise ValueError(
                f"Found multiple files for metric '{metric}': "
                f"{metric_files[metric].name}, {path.name}"
            )
        metric_files[metric] = path
    return metric_files


def parse_cap_rows(row: dict[str, str], metric: str) -> Iterable[dict[str, str]]:
    base = {
        "Subject_ID": row.get("Subject_ID", ""),
        "Group": infer_group(row.get("Subject_ID", "")),
        "Run": row.get("Run", ""),
        "Metric": metric,
    }
    for key, value in row.items():
        if key.startswith("CAP-"):
            yield {
                **base,
                "Feature": key,
                "Value": value,
            }


def parse_transition_frequency_rows(row: dict[str, str], metric: str) -> dict[str, str]:
    return {
        "Subject_ID": row.get("Subject_ID", ""),
        "Group": infer_group(row.get("Subject_ID", "")),
        "Run": row.get("Run", ""),
        "Metric": metric,
        "Feature": "",
        "Value": row.get("Transition_Frequency", ""),
    }


def is_transition_feature(name: str) -> bool:
    parts = name.split(".")
    return len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit()


def parse_transition_probability_rows(
    row: dict[str, str], metric: str
) -> Iterable[dict[str, str]]:
    base = {
        "Subject_ID": row.get("Subject_ID", ""),
        "Group": infer_group(row.get("Subject_ID", "")),
        "Run": row.get("Run", ""),
        "Metric": metric,
    }
    for key, value in row.items():
        if is_transition_feature(key):
            yield {
                **base,
                "Feature": key,
                "Value": value,
            }


def merge_files(
    input_dir: Path, output_path: Path | None = None
) -> list[dict[str, str]]:
    """Merge metric CSVs, write the long table, and return its rows."""
    input_dir = Path(input_dir)
    metric_files = collect_metric_files(input_dir)

    missing = [m for m in METRIC_KEYWORDS.values() if m not in metric_files]
    if missing:
        raise ValueError(f"Missing metric files for: {', '.join(missing)}")

    merged_rows: list[dict[str, str]] = []

    for metric in METRIC_KEYWORDS.values():
        path = metric_files[metric]
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                if metric in {"counts", "persistence", "temporal_fraction"}:
                    merged_rows.extend(parse_cap_rows(row, metric))
                elif metric == "transition_frequency":
                    merged_rows.append(parse_transition_frequency_rows(row, metric))
                elif metric == "transition_probability":
                    merged_rows.extend(parse_transition_probability_rows(row, metric))

    if output_path is None:
        output_path = input_dir / "all_metrics_long.csv"
    else:
        output_path = Path(output_path)

    write_output(merged_rows, output_path)
    print(f"[OK] Wrote {len(merged_rows)} rows to {output_path}")
    return merged_rows


def write_output(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge NeuroCAPs metric CSV files into one long-format table."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/neurocaps_full/03_metrics"),
        help="Directory containing metric CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/neurocaps_full/03_metrics/all_metrics_long.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        merge_files(args.input_dir, args.output)
    except Exception as exc:  # pragma: no cover - CLI error surface
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
