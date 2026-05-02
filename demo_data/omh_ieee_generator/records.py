from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

from demo_data.omh_ieee_generator.fhir import SyntheaPatientProfile
from demo_data.omh_ieee_generator.simulate import SyntheticPatientDay
from demo_data.omh_ieee_generator.record_builder import (
    build_header,
    build_heart_rate_series,
    build_physical_activity_record,
)


def build_patient_records(
    profile: SyntheaPatientProfile,
    patient_days: list[SyntheticPatientDay],
    *,
    source_name: str,
) -> dict[str, list[dict[str, Any]]]:
    records_by_schema: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for patient_day in patient_days:
        for record in _build_day_records(profile, patient_day, source_name=source_name):
            schema_id = _record_schema_id(record)
            records_by_schema[schema_id].append(record)
    return dict(records_by_schema)


def _build_day_records(
    profile: SyntheaPatientProfile,
    patient_day: SyntheticPatientDay,
    *,
    source_name: str,
) -> list[dict[str, Any]]:
    del profile  # profile is available for future extension; currently not embedded in records.
    return [
        _build_ieee_physical_activity(patient_day, source_name=source_name),
        _build_ieee_heart_rate_series(patient_day, source_name=source_name),
        _build_ieee_sleep_episode(patient_day, source_name=source_name),
        _build_ieee_time_in_bed(patient_day, source_name=source_name),
        _build_omh_heart_rate_summary(patient_day, source_name=source_name),
        _build_omh_respiratory_rate(patient_day, source_name=source_name),
        _build_omh_oxygen_saturation(patient_day, source_name=source_name),
        _build_omh_total_sleep_time(patient_day, source_name=source_name),
        _build_omh_sleep_episode(patient_day, source_name=source_name),
    ]


def _build_ieee_physical_activity(
    patient_day: SyntheticPatientDay,
    *,
    source_name: str,
) -> dict[str, Any]:
    day_start = patient_day.sleep_end.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start.replace(hour=23, minute=59, second=59)
    total_duration = (
        patient_day.light_activity_sec
        + patient_day.moderate_activity_sec
        + patient_day.vigorous_activity_sec
    )
    return build_physical_activity_record(
        activity_name="Total Daily Physical Activity",
        start=day_start,
        end=day_end,
        standard="ieee",
        source_name=source_name,
        base_movement_quantity=patient_day.steps,
        distance_m=patient_day.distance_m,
        kcal_burned=patient_day.active_kcal,
        duration_sec=total_duration,
        duration_light_sec=patient_day.light_activity_sec,
        duration_moderate_sec=patient_day.moderate_activity_sec,
        duration_vigorous_sec=patient_day.vigorous_activity_sec,
        descriptive_statistic="sum",
        descriptive_statistic_denominator="d",
    )


def _build_ieee_heart_rate_series(
    patient_day: SyntheticPatientDay,
    *,
    source_name: str,
) -> dict[str, Any]:
    return build_heart_rate_series(
        patient_day.heart_rate_samples,
        standard="ieee",
        source_name=source_name,
    )


def _build_ieee_sleep_episode(
    patient_day: SyntheticPatientDay,
    *,
    source_name: str,
) -> dict[str, Any]:
    record = build_header("ieee", "ieee:sleep-episode:1.0", source_name=source_name)
    record["body"] = {
        "latency_to_sleep_onset": _duration_unit_value(patient_day.latency_to_sleep_onset_sec),
        "total_sleep_time": _duration_unit_value(patient_day.total_sleep_time_sec),
        "light_sleep_duration": _duration_unit_value(patient_day.light_sleep_duration_sec),
        "deep_sleep_duration": _duration_unit_value(patient_day.deep_sleep_duration_sec),
        "rem_sleep_duration": _duration_unit_value(patient_day.rem_sleep_duration_sec),
        "wake_after_sleep_onset": _duration_unit_value(patient_day.wake_after_sleep_onset_sec),
        "is_main_sleep": True,
        "sleep_efficiency_percentage": _unit_value(patient_day.sleep_efficiency_pct, "%"),
        "effective_time_frame": {
            "time_interval": _time_interval(patient_day.sleep_start, patient_day.sleep_end)
        },
    }
    return record


def _build_ieee_time_in_bed(
    patient_day: SyntheticPatientDay,
    *,
    source_name: str,
) -> dict[str, Any]:
    record = build_header("ieee", "ieee:time-in-bed:1.0", source_name=source_name)
    record["body"] = {
        "time_in_bed": _duration_unit_value(patient_day.time_in_bed_sec),
        "effective_time_frame": {
            "time_interval": _time_interval(patient_day.sleep_start, patient_day.sleep_end)
        },
        "is_main_sleep": True,
    }
    return record


def _build_omh_heart_rate_summary(
    patient_day: SyntheticPatientDay,
    *,
    source_name: str,
) -> dict[str, Any]:
    record = build_header("omh", "omh:heart-rate:2.0", source_name=source_name)
    record["body"] = {
        "heart_rate": _unit_value(patient_day.average_heart_rate_bpm, "beats/min"),
        "descriptive_statistic": "average",
        "temporal_relationship_to_sleep": "during",
        "effective_time_frame": {
            "time_interval": _time_interval(patient_day.sleep_start, patient_day.sleep_end)
        },
    }
    return record


def _build_omh_respiratory_rate(
    patient_day: SyntheticPatientDay,
    *,
    source_name: str,
) -> dict[str, Any]:
    record = build_header("omh", "omh:respiratory-rate:2.0", source_name=source_name)
    record["body"] = {
        "respiratory_rate": _unit_value(patient_day.respiratory_rate_bpm, "breaths/min"),
        "descriptive_statistic": "average",
        "effective_time_frame": {
            "time_interval": _time_interval(patient_day.sleep_start, patient_day.sleep_end)
        },
    }
    return record


def _build_omh_oxygen_saturation(
    patient_day: SyntheticPatientDay,
    *,
    source_name: str,
) -> dict[str, Any]:
    record = build_header("omh", "omh:oxygen-saturation:2.0", source_name=source_name)
    record["body"] = {
        "oxygen_saturation": _unit_value(patient_day.oxygen_saturation_pct, "%"),
        "descriptive_statistic": "average",
        "measurement_method": "pulse oximetry",
        "system": "peripheral capillary",
        "effective_time_frame": {
            "time_interval": _time_interval(patient_day.sleep_start, patient_day.sleep_end)
        },
    }
    return record


def _build_omh_total_sleep_time(
    patient_day: SyntheticPatientDay,
    *,
    source_name: str,
) -> dict[str, Any]:
    record = build_header("omh", "omh:total-sleep-time:1.0", source_name=source_name)
    record["body"] = {
        "total_sleep_time": _duration_unit_value(patient_day.total_sleep_time_sec),
        "effective_time_frame": {
            "time_interval": _time_interval(patient_day.sleep_start, patient_day.sleep_end)
        },
    }
    return record


def _build_omh_sleep_episode(
    patient_day: SyntheticPatientDay,
    *,
    source_name: str,
) -> dict[str, Any]:
    record = build_header("omh", "omh:sleep-episode:1.1", source_name=source_name)
    record["body"] = {
        "latency_to_sleep_onset": _duration_unit_value(patient_day.latency_to_sleep_onset_sec),
        "total_sleep_time": _duration_unit_value(patient_day.total_sleep_time_sec),
        "wake_after_sleep_onset": _duration_unit_value(patient_day.wake_after_sleep_onset_sec),
        "is_main_sleep": True,
        "sleep_maintenance_efficiency_percentage": _unit_value(patient_day.sleep_efficiency_pct, "%"),
        "effective_time_frame": {
            "time_interval": _time_interval(patient_day.sleep_start, patient_day.sleep_end)
        },
    }
    return record


def _time_interval(start: datetime, end: datetime) -> dict[str, Any]:
    return {
        "start_date_time": _isoformat(start),
        "end_date_time": _isoformat(end),
        "duration": _duration_unit_value((end - start).total_seconds()),
    }


def _unit_value(value: float, unit: str) -> dict[str, Any]:
    return {"value": round(float(value), 1), "unit": unit}


def _duration_unit_value(value: float) -> dict[str, Any]:
    return {"value": int(round(float(value))), "unit": "sec"}


def _record_schema_id(record: dict[str, Any]) -> str:
    schema = record["header"]["schema_id"]
    return f"{schema['namespace']}:{schema['name']}:{schema['version']}"


def _isoformat(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")
