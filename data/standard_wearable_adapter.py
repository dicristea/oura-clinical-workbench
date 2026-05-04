from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.base import DataSource, PatientTimeSeries
from data.feature_registry import get_feature_groups_for_source


def _body_value(body: dict[str, Any], field: str) -> float | None:
    value = body.get(field)
    if isinstance(value, dict):
        value = value.get("value")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _interval(body: dict[str, Any]) -> dict[str, Any]:
    return body.get("effective_time_frame", {}).get("time_interval", {})


def _record_date(body: dict[str, Any]) -> pd.Timestamp | None:
    interval = _interval(body)
    date_text = interval.get("start_date_time") or interval.get("date_time")
    if not date_text:
        return None
    try:
        return pd.to_datetime(date_text, utc=True).tz_convert(None).normalize()
    except Exception:
        return None


def _record_start_label(body: dict[str, Any]) -> str | None:
    start = _interval(body).get("start_date_time")
    if not start:
        return None
    try:
        return pd.to_datetime(start, utc=True).tz_convert(None).strftime("%I:%M %p").lstrip("0")
    except Exception:
        return None


def _record_end_label(body: dict[str, Any]) -> str | None:
    end = _interval(body).get("end_date_time")
    if not end:
        return None
    try:
        return pd.to_datetime(end, utc=True).tz_convert(None).strftime("%I:%M %p").lstrip("0")
    except Exception:
        return None


def _compute_risk_level(ts: pd.DataFrame) -> str:
    hrv_mean = ts["hrv_balance"].mean() if "hrv_balance" in ts.columns else np.nan
    rem_mean = ts["rem_sleep_pct"].mean() if "rem_sleep_pct" in ts.columns else np.nan

    if not pd.isna(hrv_mean) and hrv_mean < 30:
        return "high"
    if not pd.isna(rem_mean) and rem_mean < 16:
        return "high"
    if not pd.isna(hrv_mean) and hrv_mean <= 45:
        return "medium"
    if not pd.isna(rem_mean) and rem_mean <= 20:
        return "medium"
    return "low"


class StandardWearableAdapter:
    """Read OMH/IEEE wearable records into the dashboard's PatientTimeSeries model."""

    def load_from_records(self, filepath: str | Path, patient_id: str | None = None) -> PatientTimeSeries:
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Standard wearable records not found: {path}")

        payload = json.loads(path.read_text(encoding="utf-8"))
        patient = payload.get("patient", {})
        resolved_patient_id = str(patient_id or patient.get("patient_id") or path.parent.name)
        records_by_schema = payload.get("records_by_schema", {})

        daily: dict[pd.Timestamp, dict[str, Any]] = {}

        def row_for(body: dict[str, Any]) -> dict[str, Any] | None:
            date = _record_date(body)
            if date is None:
                return None
            return daily.setdefault(date, {})

        for record in records_by_schema.get("ieee:sleep-episode:1.0", []):
            body = record.get("body", {})
            row = row_for(body)
            if row is None:
                continue

            total_sec = _body_value(body, "total_sleep_time")
            rem_sec = _body_value(body, "rem_sleep_duration")
            deep_sec = _body_value(body, "deep_sleep_duration")
            light_sec = _body_value(body, "light_sleep_duration")
            latency_sec = _body_value(body, "latency_to_sleep_onset")
            waso_sec = _body_value(body, "wake_after_sleep_onset")
            efficiency = _body_value(body, "sleep_efficiency_percentage")

            if total_sec and total_sec > 0:
                row["total_sleep_time_hours"] = round(total_sec / 3600, 2)
                row["sleep_duration_hours"] = round(total_sec / 3600, 2)
                if rem_sec is not None:
                    row["rem_sleep_pct"] = round(rem_sec / total_sec * 100, 1)
                    row["rem_sleep_hours"] = round(rem_sec / 3600, 2)
                if deep_sec is not None:
                    row["deep_sleep_pct"] = round(deep_sec / total_sec * 100, 1)
                    row["deep_sleep_hours"] = round(deep_sec / 3600, 2)
                if light_sec is not None:
                    row["light_sleep_hours"] = round(light_sec / 3600, 2)
            if latency_sec is not None:
                row["sleep_latency"] = round(latency_sec / 60, 1)
            if waso_sec is not None:
                row["waso_minutes"] = round(waso_sec / 60, 1)
                row["awake_hours"] = round(waso_sec / 3600, 2)
            if efficiency is not None:
                row["sleep_efficiency_pct"] = round(efficiency, 1)

            start_label = _record_start_label(body)
            end_label = _record_end_label(body)
            if start_label:
                row["sleep_start_label"] = start_label
            if end_label:
                row["sleep_end_label"] = end_label

        for record in records_by_schema.get("omh:sleep-episode:1.1", []):
            body = record.get("body", {})
            row = row_for(body)
            if row is None:
                continue
            efficiency = _body_value(body, "sleep_maintenance_efficiency_percentage")
            if efficiency is not None:
                row.setdefault("sleep_efficiency_pct", round(efficiency, 1))

        for record in records_by_schema.get("ieee:time-in-bed:1.0", []):
            body = record.get("body", {})
            row = row_for(body)
            tib_sec = _body_value(body, "time_in_bed")
            if row is not None and tib_sec is not None:
                row["time_in_bed_hours"] = round(tib_sec / 3600, 2)

        for record in records_by_schema.get("omh:total-sleep-time:1.0", []):
            body = record.get("body", {})
            row = row_for(body)
            tst_sec = _body_value(body, "total_sleep_time")
            if row is not None and tst_sec is not None:
                row.setdefault("total_sleep_time_hours", round(tst_sec / 3600, 2))
                row.setdefault("sleep_duration_hours", round(tst_sec / 3600, 2))

        for record in records_by_schema.get("ieee:physical-activity:1.0", []):
            body = record.get("body", {})
            row = row_for(body)
            if row is None:
                continue
            steps = _body_value(body, "base_movement_quantity")
            distance = _body_value(body, "distance")
            kcal = _body_value(body, "kcal_burned")
            if steps is not None:
                row["step_count"] = int(round(steps))
            if distance is not None:
                row["distance_m"] = round(distance, 1)
            if kcal is not None:
                row["active_kcal"] = round(kcal, 1)

        for record in records_by_schema.get("omh:heart-rate:2.0", []):
            body = record.get("body", {})
            row = row_for(body)
            heart_rate = _body_value(body, "heart_rate")
            if row is not None and heart_rate is not None:
                row["resting_hr"] = round(heart_rate, 1)

        for record in records_by_schema.get("omh:respiratory-rate:2.0", []):
            body = record.get("body", {})
            row = row_for(body)
            respiratory_rate = _body_value(body, "respiratory_rate")
            if row is not None and respiratory_rate is not None:
                row["respiratory_rate"] = round(respiratory_rate, 1)

        for record in records_by_schema.get("omh:oxygen-saturation:2.0", []):
            body = record.get("body", {})
            row = row_for(body)
            spo2 = _body_value(body, "oxygen_saturation")
            if row is not None and spo2 is not None:
                row["spo2_pct"] = round(spo2, 1)

        if not daily:
            raise ValueError(f"No usable OMH/IEEE wearable records found in {path}")

        time_series = pd.DataFrame.from_dict(daily, orient="index").sort_index()
        time_series.index = pd.DatetimeIndex(time_series.index)

        conditions = patient.get("conditions", [])
        if isinstance(conditions, list):
            condition_text = ", ".join(str(c) for c in conditions)
        else:
            condition_text = str(conditions or "")

        static_features = {
            "name": patient.get("display_name") or resolved_patient_id,
            "age": patient.get("age"),
            "sex": patient.get("gender") or patient.get("sex"),
            "conditions": condition_text,
        }

        risk_level = _compute_risk_level(time_series)
        feature_groups = {
            group: [fc.name for fc in fcs]
            for group, fcs in get_feature_groups_for_source(DataSource.OURA).items()
        }

        return PatientTimeSeries(
            patient_id=resolved_patient_id,
            data_source=DataSource.OURA,
            static_features=static_features,
            time_series=time_series,
            metadata={
                "data_source_label": "OMH/IEEE Wearable Records",
                "data_points_count": len(time_series),
                "risk_level": risk_level,
                "loaded_from": str(path),
                "standard_record_counts": payload.get("counts_by_schema", {}),
            },
            feature_groups=feature_groups,
        )

    def load_all_from_dir(self, dataset_dir: str | Path) -> list[PatientTimeSeries]:
        root = Path(dataset_dir)
        if not root.is_dir():
            return []
        record_files = sorted(root.glob("patients/*/records.json"))
        results: list[PatientTimeSeries] = []
        for path in record_files:
            try:
                results.append(self.load_from_records(path))
            except Exception as exc:
                print(f"[StandardWearableAdapter] Skipping {path}: {exc}")
        return results
