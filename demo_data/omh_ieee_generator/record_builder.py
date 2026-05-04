from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Literal

Standard = Literal["ieee", "omh"]
DEFAULT_SOURCE_NAME = "demo-omh-ieee-generator"


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def build_header(
    standard: Standard,
    schema_id: str,
    source_name: str = DEFAULT_SOURCE_NAME,
) -> dict[str, Any]:
    """Build the shared OMH / IEEE record header for generated demo records."""
    del standard  # The namespace is already encoded in schema_id.
    namespace, name, version = schema_id.split(":", 2)
    return {
        "header": {
            "uuid": str(uuid.uuid4()),
            "source_creation_date_time": _iso(datetime.now(timezone.utc)),
            "schema_id": {
                "namespace": namespace,
                "name": name,
                "version": version,
            },
            "modality": "sensed",
            "acquisition_provenance": {
                "source_name": source_name,
            },
        }
    }


def unit_value(value: float, unit: str, digits: int | None = None) -> dict[str, Any]:
    if digits is not None:
        value = round(value, digits)
    return {"value": value, "unit": unit}


def time_interval(start: datetime, end: datetime) -> dict[str, Any]:
    return {
        "time_interval": {
            "start_date_time": _iso(start),
            "end_date_time": _iso(end),
        }
    }


def date_time_frame(dt: datetime) -> dict[str, Any]:
    return {"date_time": _iso(dt)}


@dataclass(frozen=True)
class HeartRateSample:
    timestamp: datetime
    bpm: float


def build_heart_rate_series(
    samples: Iterable[HeartRateSample],
    standard: Standard = "ieee",
    source_name: str = DEFAULT_SOURCE_NAME,
) -> dict[str, Any]:
    schema_id = "ieee:heart-rate:1.0" if standard == "ieee" else "omh:heart-rate:2.0"
    body = [
        {
            "heart_rate": unit_value(sample.bpm, "beats/min", digits=1),
            "effective_time_frame": date_time_frame(sample.timestamp),
        }
        for sample in samples
    ]

    record = build_header(standard, schema_id, source_name=source_name)
    record["body"] = body
    return record


def build_physical_activity_record(
    activity_name: str,
    start: datetime,
    end: datetime,
    *,
    standard: Standard = "ieee",
    source_name: str = DEFAULT_SOURCE_NAME,
    base_movement_quantity: int | None = None,
    distance_m: float | None = None,
    kcal_burned: float | None = None,
    duration_sec: int | None = None,
    duration_light_sec: int | None = None,
    duration_moderate_sec: int | None = None,
    duration_vigorous_sec: int | None = None,
    descriptive_statistic: str | None = None,
    descriptive_statistic_denominator: str | None = None,
) -> dict[str, Any]:
    schema_id = "ieee:physical-activity:1.0" if standard == "ieee" else "omh:physical-activity:1.0"

    record = build_header(standard, schema_id, source_name=source_name)
    body: dict[str, Any] = {
        "activity_name": activity_name,
        "effective_time_frame": time_interval(start, end),
    }

    if base_movement_quantity is not None:
        body["base_movement_quantity"] = unit_value(base_movement_quantity, "steps")
    if distance_m is not None:
        body["distance"] = unit_value(distance_m, "m", digits=1)
    if kcal_burned is not None:
        body["kcal_burned"] = unit_value(kcal_burned, "kcal", digits=1)
    if duration_sec is not None:
        body["duration"] = unit_value(duration_sec, "sec")
    if duration_light_sec is not None:
        body["duration_light_activity"] = unit_value(duration_light_sec, "sec")
    if duration_moderate_sec is not None:
        body["duration_moderate_activity"] = unit_value(duration_moderate_sec, "sec")
    if duration_vigorous_sec is not None:
        body["duration_vigorous_activity"] = unit_value(duration_vigorous_sec, "sec")
    if descriptive_statistic is not None:
        body["descriptive_statistic"] = descriptive_statistic
    if descriptive_statistic_denominator is not None:
        body["descriptive_statistic_denominator"] = descriptive_statistic_denominator

    record["body"] = body
    return record
