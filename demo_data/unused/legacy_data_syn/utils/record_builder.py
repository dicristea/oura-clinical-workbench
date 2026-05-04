from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Literal, Optional

Standard = Literal["ieee", "omh"]
DEFAULT_SOURCE_NAME = "data_syn"


# -----------------------------
# Generic helpers
# -----------------------------

def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def _deep_merge(target: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
    return target


def set_by_path(obj: Dict[str, Any], path: str, value: Any) -> Dict[str, Any]:
    """Set nested dictionary value by dotted path, e.g. body.heart_rate.value"""
    parts = path.split(".")
    cur = obj
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value
    return obj


def build_header(standard: Standard, schema_id: str, source_name: str = DEFAULT_SOURCE_NAME) -> Dict[str, Any]:
    """Build the shared record header used by the data_syn conversion pipeline."""
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


def unit_value(value: float, unit: str, digits: Optional[int] = None) -> Dict[str, Any]:
    if digits is not None:
        value = round(value, digits)
    return {"value": value, "unit": unit}


def time_interval(start: datetime, end: datetime) -> Dict[str, Any]:
    return {
        "time_interval": {
            "start_date_time": _iso(start),
            "end_date_time": _iso(end),
        }
    }


def date_time_frame(dt: datetime) -> Dict[str, Any]:
    return {"date_time": _iso(dt)}


# -----------------------------
# Heart rate builders
# -----------------------------

@dataclass
class HeartRateSample:
    timestamp: datetime
    bpm: float


def build_heart_rate_measurement(
    bpm: float,
    timestamp: datetime,
    standard: Standard = "ieee",
    descriptive_statistic: Optional[str] = None,
    relationship_to_sleep: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a single heart-rate record."""
    schema_id = "ieee:heart-rate:1.0" if standard == "ieee" else "omh:heart-rate:2.0"

    record = build_header(standard, schema_id)
    body = {
        "heart_rate": unit_value(bpm, "beats/min", digits=1),
        "effective_time_frame": date_time_frame(timestamp),
    }
    if descriptive_statistic:
        body["descriptive_statistic"] = descriptive_statistic
    if relationship_to_sleep:
        body["temporal_relationship_to_sleep"] = relationship_to_sleep

    record["body"] = body
    return record


def build_heart_rate_series(
    samples: Iterable[HeartRateSample],
    standard: Standard = "ieee",
    source_name: str = DEFAULT_SOURCE_NAME,
) -> Dict[str, Any]:
    """
    Build a metadata data-series record.

    For IEEE 1752 data-series instances, the header identifies the schema and the
    body is an array of point measurements that each conform to that schema.
    """
    schema_id = "ieee:heart-rate:1.0" if standard == "ieee" else "omh:heart-rate:2.0"
    body = []
    for sample in samples:
        body.append(
            {
                "heart_rate": unit_value(sample.bpm, "beats/min", digits=1),
                "effective_time_frame": date_time_frame(sample.timestamp),
            }
        )

    record = build_header(standard, schema_id, source_name=source_name)
    record["body"] = body
    return record


# -----------------------------
# Physical activity builders
# -----------------------------

def build_physical_activity_record(
    activity_name: str,
    start: datetime,
    end: datetime,
    *,
    standard: Standard = "ieee",
    source_name: str = DEFAULT_SOURCE_NAME,
    base_movement_quantity: Optional[int] = None,
    distance_m: Optional[float] = None,
    kcal_burned: Optional[float] = None,
    duration_sec: Optional[int] = None,
    duration_light_sec: Optional[int] = None,
    duration_moderate_sec: Optional[int] = None,
    duration_vigorous_sec: Optional[int] = None,
    average_speed_mps: Optional[float] = None,
    reported_intensity: Optional[str] = None,
    descriptive_statistic: Optional[str] = None,
    descriptive_statistic_denominator: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a physical activity record.
    Works for both:
      - a single workout episode
      - a daily aggregate like 'all daily activities'
    """
    schema_id = "ieee:physical-activity:1.0" if standard == "ieee" else "omh:physical-activity:1.0"

    record = build_header(standard, schema_id, source_name=source_name)
    body: Dict[str, Any] = {
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
    if average_speed_mps is not None:
        body["average_speed"] = unit_value(average_speed_mps, "m/s", digits=2)
    if reported_intensity is not None:
        body["reported_activity_intensity"] = reported_intensity
    if descriptive_statistic is not None:
        body["descriptive_statistic"] = descriptive_statistic
    if descriptive_statistic_denominator is not None:
        body["descriptive_statistic_denominator"] = descriptive_statistic_denominator

    record["body"] = body
    return record
