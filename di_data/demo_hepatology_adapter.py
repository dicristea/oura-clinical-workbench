#!/usr/bin/env python3
"""Terminal demo for HepatologyDataAdapter; no dashboard UI is required."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.hepatology_adapter import HepatologyDataAdapter  # noqa: E402


def _format_value(value: object) -> str:
    if value is None:
        return "missing"
    try:
        if value != value:  # NaN
            return "missing"
    except Exception:
        pass
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect converted hepatology JSON data.")
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default=str(Path(__file__).resolve().parent / "exports" / "hepatology_sample_10"),
        help="Directory containing manifest.json, participants.json, and daily_records.json.",
    )
    parser.add_argument(
        "--patient-id",
        help="Patient to show. Defaults to the first patient in the converted dataset.",
    )
    args = parser.parse_args()

    adapter = HepatologyDataAdapter(args.dataset_dir)
    summary = adapter.describe_dataset()
    patient_ids = adapter.list_patient_ids()
    if not patient_ids:
        raise SystemExit("The converted dataset contains no participants.")

    patient_id = args.patient_id or patient_ids[0]
    patient = adapter.load_patient(patient_id)
    valid_daily = adapter.load_patient_daily(patient_id, valid_only=True)

    print("Hepatology Dataset Adapter Demo")
    print("================================")
    for label, key in (
        ("Schema version", "schema_version"),
        ("Participants", "participants"),
        ("Participant-days", "participant_days"),
        ("Days passing valid80", "valid80_days"),
        ("Within-day records", "within_day_records"),
        ("Dictionary fields", "dictionary_fields"),
        ("Time axis", "time_axis"),
    ):
        print(f"{label}: {summary[key]}")

    print("\nSelected patient")
    print("----------------")
    print(f"Patient ID: {patient.patient_id}")
    print(f"Total daily records: {len(patient.time_series)}")
    print(f"Valid daily records: {len(valid_daily)}")
    print(f"Study-day range: {patient.time_series.index.min()}–{patient.time_series.index.max()}")
    print("Feature groups:")
    for group, fields in patient.feature_groups.items():
        print(f"  {group}: {', '.join(fields)}")

    print("\nSelected clinical fields")
    print("------------------------")
    for field in ("sex", "age_yrs", "ctp_cat", "meld", "anyhe", "anyOHE", "frail"):
        print(f"{field}: {_format_value(patient.static_features.get(field))}")

    print("\nDaily data preview")
    print("------------------")
    preview_fields = [
        field
        for field in (
            "step_count",
            "average_heart_rate",
            "average_hrv",
            "respiratory_rate",
            "spo2_pct",
            "valid80",
            "non_wear_pct",
        )
        if field in patient.time_series.columns
    ]
    print(patient.time_series[preview_fields].head(5).to_string())

    available_within = [
        record
        for record in adapter.within_day_records
        if str(record.get("patient_id")) == patient.patient_id
    ]
    if available_within:
        record = available_within[0]
        print("\nWithin-day example")
        print("------------------")
        print(f"Study day: {record['study_day']}")
        for field, length in record.get("series_lengths", {}).items():
            print(f"{field}: {length} values")
    else:
        print("\nNo within-day example was exported for this patient.")


if __name__ == "__main__":
    main()
