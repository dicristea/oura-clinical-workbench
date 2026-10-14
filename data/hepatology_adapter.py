from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path
from typing import Any

import pandas as pd

from data.base import DataSource, PatientTimeSeries
from data.feature_registry import get_feature_groups_for_source


# Only rename fields whose semantics are sufficiently clear. Ambiguous source
# fields such as average_heart_rate and average_hrv deliberately keep their
# source names until the clinical/data team confirms their definitions.
DAILY_FIELD_MAP: dict[str, str] = {
    "average_breath": "respiratory_rate",
    "spo2": "spo2_pct",
    "steps": "step_count",
}

REQUIRED_FILES = ("manifest.json", "participants.json", "daily_records.json")


class HepatologyDataAdapter:
    """Load converted de-identified hepatology JSON without a patient limit.

    The adapter consumes the output of ``di_data/convert_hepatology_rds.R``. It keeps
    the source study-day axis instead of inventing calendar dates and returns
    the shared ``PatientTimeSeries`` model with ``time_axis='study_day'``.

    Within-day records are loaded separately because the 1-minute arrays should
    only be read when a detailed activity view requests them.
    """

    def __init__(self, dataset_dir: str | Path):
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.is_dir():
            raise FileNotFoundError(f"Hepatology dataset directory not found: {self.dataset_dir}")

        missing = [name for name in REQUIRED_FILES if not (self.dataset_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Hepatology dataset is missing required files: {', '.join(missing)}"
            )

        self._validate_dataset()

    @staticmethod
    def _read_json(path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

    @cached_property
    def manifest(self) -> dict[str, Any]:
        value = self._read_json(self.dataset_dir / "manifest.json")
        if not isinstance(value, dict):
            raise ValueError("manifest.json must contain a JSON object.")
        return value

    @cached_property
    def participants(self) -> list[dict[str, Any]]:
        value = self._read_json(self.dataset_dir / "participants.json")
        if not isinstance(value, list):
            raise ValueError("participants.json must contain a JSON array.")
        return value

    @cached_property
    def daily_records(self) -> list[dict[str, Any]]:
        value = self._read_json(self.dataset_dir / "daily_records.json")
        if not isinstance(value, list):
            raise ValueError("daily_records.json must contain a JSON array.")
        return value

    @cached_property
    def within_day_records(self) -> list[dict[str, Any]]:
        path = self.dataset_dir / "within_day_records.json"
        if not path.is_file():
            return []
        value = self._read_json(path)
        if not isinstance(value, list):
            raise ValueError("within_day_records.json must contain a JSON array.")
        return value

    @cached_property
    def variable_dictionary(self) -> list[dict[str, Any]]:
        path = self.dataset_dir / "variable_dictionary.json"
        if not path.is_file():
            return []
        value = self._read_json(path)
        if not isinstance(value, list):
            raise ValueError("variable_dictionary.json must contain a JSON array.")
        return value

    @cached_property
    def _participants_by_id(self) -> dict[str, dict[str, Any]]:
        return {str(item["patient_id"]): item for item in self.participants}

    @cached_property
    def _daily_dataframe(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.daily_records)
        if frame.empty:
            return pd.DataFrame(columns=["patient_id", "study_day"])

        frame["patient_id"] = frame["patient_id"].astype(str)
        frame["study_day"] = pd.to_numeric(frame["study_day"], errors="raise").astype(int)
        return frame.sort_values(["patient_id", "study_day"]).reset_index(drop=True)

    def _validate_dataset(self) -> None:
        schema_version = str(self.manifest.get("schema_version", ""))
        if not schema_version.startswith("1."):
            raise ValueError(
                f"Unsupported hepatology schema version {schema_version!r}; expected 1.x."
            )

        patient_ids: list[str] = []
        for position, participant in enumerate(self.participants):
            if not isinstance(participant, dict):
                raise ValueError(f"Participant at position {position} is not an object.")
            patient_id = participant.get("patient_id")
            if patient_id is None or str(patient_id).strip() == "":
                raise ValueError(f"Participant at position {position} has no patient_id.")
            if not isinstance(participant.get("static_features", {}), dict):
                raise ValueError(f"Participant {patient_id} has invalid static_features.")
            patient_ids.append(str(patient_id))

        if len(patient_ids) != len(set(patient_ids)):
            raise ValueError("participants.json contains duplicate patient_id values.")

        known_ids = set(patient_ids)
        daily_keys: set[tuple[str, int]] = set()
        for position, record in enumerate(self.daily_records):
            if not isinstance(record, dict):
                raise ValueError(f"Daily record at position {position} is not an object.")
            patient_id = str(record.get("patient_id", ""))
            if patient_id not in known_ids:
                raise ValueError(f"Daily record references unknown patient_id {patient_id!r}.")
            try:
                study_day = int(record["study_day"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Daily record for {patient_id} has invalid study_day."
                ) from exc
            key = (patient_id, study_day)
            if key in daily_keys:
                raise ValueError(f"Duplicate daily record for {patient_id}, study day {study_day}.")
            daily_keys.add(key)

        for position, record in enumerate(self.within_day_records):
            patient_id = str(record.get("patient_id", ""))
            if patient_id not in known_ids:
                raise ValueError(
                    f"Within-day record at position {position} references unknown patient_id."
                )
            series = record.get("series", {})
            lengths = record.get("series_lengths", {})
            if not isinstance(series, dict) or not isinstance(lengths, dict):
                raise ValueError(f"Within-day record for {patient_id} has invalid series data.")
            for field, values in series.items():
                if not isinstance(values, list):
                    raise ValueError(f"Within-day field {field} for {patient_id} is not an array.")
                declared_length = lengths.get(field)
                if declared_length is not None and int(declared_length) != len(values):
                    raise ValueError(
                        f"Within-day length mismatch for {patient_id}, field {field}."
                    )

    def list_patient_ids(self) -> list[str]:
        """Return every available patient ID in file order."""
        return list(self._participants_by_id)

    def get_participant(self, patient_id: str) -> dict[str, Any]:
        """Return one participant-level record."""
        try:
            return self._participants_by_id[str(patient_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown hepatology patient_id: {patient_id}") from exc

    def load_participants(self) -> list[dict[str, Any]]:
        """Return all participant-level records; no display limit is applied."""
        return list(self.participants)

    def cohort_dataframe(self) -> pd.DataFrame:
        """Return one flat row per participant for cohort summaries and filters."""
        rows: list[dict[str, Any]] = []
        for participant in self.participants:
            row = {
                "patient_id": str(participant["patient_id"]),
                **participant.get("static_features", {}),
                **participant.get("coverage", {}),
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def load_patient_daily(self, patient_id: str, *, valid_only: bool = False) -> pd.DataFrame:
        """Return one patient's daily records indexed by relative study day."""
        patient_id = str(patient_id)
        self.get_participant(patient_id)

        frame = self._daily_dataframe[self._daily_dataframe["patient_id"] == patient_id].copy()
        if valid_only and "valid80" in frame.columns:
            frame = frame[pd.to_numeric(frame["valid80"], errors="coerce") == 1].copy()

        frame = frame.drop(columns=["patient_id"]).rename(columns=DAILY_FIELD_MAP)
        if frame.empty:
            empty = frame.drop(columns=["study_day"], errors="ignore")
            empty.index = pd.Index([], name="study_day", dtype="int64")
            return empty

        frame = frame.set_index("study_day").sort_index()
        frame.index = frame.index.astype(int)
        frame.index.name = "study_day"
        return frame

    def load_patient(self, patient_id: str, *, valid_only: bool = False) -> PatientTimeSeries:
        """Return one patient in the project's shared time-series container."""
        participant = self.get_participant(patient_id)
        daily = self.load_patient_daily(patient_id, valid_only=valid_only)
        feature_groups = {
            group: [feature.name for feature in features if feature.name in daily.columns]
            for group, features in get_feature_groups_for_source(DataSource.HEPATOLOGY).items()
        }
        feature_groups = {group: fields for group, fields in feature_groups.items() if fields}

        return PatientTimeSeries(
            patient_id=str(patient_id),
            data_source=DataSource.HEPATOLOGY,
            static_features=dict(participant.get("static_features", {})),
            time_series=daily,
            metadata={
                "data_source_label": "De-identified hepatology dataset",
                "data_points_count": len(daily),
                "coverage": dict(participant.get("coverage", {})),
                "schema_version": self.manifest.get("schema_version"),
                "loaded_from": str(self.dataset_dir),
                "time_axis": "study_day",
            },
            feature_groups=feature_groups,
            time_axis="study_day",
        )

    def load_all(self, *, valid_only: bool = False) -> list[PatientTimeSeries]:
        """Load every available patient; this method never truncates the cohort."""
        return [self.load_patient(patient_id, valid_only=valid_only) for patient_id in self.list_patient_ids()]

    def load_within_day(self, patient_id: str, study_day: int) -> dict[str, Any]:
        """Return full within-day arrays for one available patient-day."""
        patient_id = str(patient_id)
        self.get_participant(patient_id)
        study_day = int(study_day)
        for record in self.within_day_records:
            if str(record.get("patient_id")) == patient_id and int(record.get("study_day")) == study_day:
                return record
        raise KeyError(
            f"No within-day record available for {patient_id}, study day {study_day}."
        )

    def describe_dataset(self) -> dict[str, Any]:
        """Return compact counts suitable for a terminal or notebook demo."""
        daily = self._daily_dataframe
        valid_days = 0
        if not daily.empty and "valid80" in daily.columns:
            valid_days = int((pd.to_numeric(daily["valid80"], errors="coerce") == 1).sum())

        return {
            "schema_version": self.manifest.get("schema_version"),
            "participants": len(self.participants),
            "participant_days": len(self.daily_records),
            "valid80_days": valid_days,
            "within_day_records": len(self.within_day_records),
            "dictionary_fields": len(self.variable_dictionary),
            "time_axis": "study_day",
        }
