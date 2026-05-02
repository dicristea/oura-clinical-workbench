from __future__ import annotations

from typing import Any


def validate_records(records_by_schema: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Run lightweight semantic checks on generated records."""
    warnings: list[str] = []

    for schema_id, records in records_by_schema.items():
        for index, record in enumerate(records):
            body = record.get("body")
            if body is None:
                warnings.append(f"{schema_id}[{index}] is missing body")
                continue

            if schema_id == "ieee:time-in-bed:1.0":
                warnings.extend(_validate_time_in_bed(schema_id, index, body))
            elif schema_id in {"ieee:sleep-episode:1.0", "omh:sleep-episode:1.1"}:
                warnings.extend(_validate_sleep_episode(schema_id, index, body))
            elif schema_id == "ieee:physical-activity:1.0":
                warnings.extend(_validate_physical_activity(schema_id, index, body))
            elif schema_id == "ieee:heart-rate:1.0":
                warnings.extend(_validate_heart_rate_series(schema_id, index, body))
            elif schema_id == "omh:heart-rate:2.0":
                warnings.extend(_validate_unit_value(schema_id, index, body, "heart_rate", 35, 180))
            elif schema_id == "omh:respiratory-rate:2.0":
                warnings.extend(_validate_unit_value(schema_id, index, body, "respiratory_rate", 8, 30))
            elif schema_id == "omh:oxygen-saturation:2.0":
                warnings.extend(_validate_unit_value(schema_id, index, body, "oxygen_saturation", 85, 100))
            elif schema_id == "omh:total-sleep-time:1.0":
                warnings.extend(_validate_duration(schema_id, index, body, "total_sleep_time"))

    return warnings


def _validate_time_in_bed(schema_id: str, index: int, body: dict[str, Any]) -> list[str]:
    warnings = _validate_duration(schema_id, index, body, "time_in_bed")
    interval = body.get("effective_time_frame", {}).get("time_interval", {})
    if not interval.get("start_date_time") or not interval.get("end_date_time"):
        warnings.append(f"{schema_id}[{index}] is missing time interval bounds")
    return warnings


def _validate_sleep_episode(schema_id: str, index: int, body: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for field in ("total_sleep_time", "wake_after_sleep_onset"):
        warnings.extend(_validate_duration(schema_id, index, body, field))

    total_sleep = _duration_value(body.get("total_sleep_time"))
    time_interval = body.get("effective_time_frame", {}).get("time_interval", {})
    interval_duration = _duration_value(time_interval.get("duration"))
    if total_sleep is not None and interval_duration is not None and total_sleep > interval_duration:
        warnings.append(f"{schema_id}[{index}] total sleep exceeds time interval duration")

    return warnings


def _validate_physical_activity(schema_id: str, index: int, body: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for field in (
        "base_movement_quantity",
        "distance",
        "kcal_burned",
        "duration",
        "duration_light_activity",
        "duration_moderate_activity",
        "duration_vigorous_activity",
    ):
        unit_value = body.get(field)
        if not isinstance(unit_value, dict):
            continue
        value = unit_value.get("value")
        if isinstance(value, (int, float)) and value < 0:
            warnings.append(f"{schema_id}[{index}] has negative value for {field}")
    return warnings


def _validate_heart_rate_series(schema_id: str, index: int, body: Any) -> list[str]:
    if not isinstance(body, list) or not body:
        return [f"{schema_id}[{index}] must contain a non-empty body list"]

    warnings: list[str] = []
    for sample_index, sample in enumerate(body):
        if not isinstance(sample, dict):
            warnings.append(f"{schema_id}[{index}] sample {sample_index} is not an object")
            continue
        warnings.extend(
            _validate_unit_value(
                f"{schema_id}[{index}]",
                sample_index,
                sample,
                "heart_rate",
                35,
                180,
            )
        )
    return warnings


def _validate_unit_value(
    schema_id: str,
    index: int,
    body: dict[str, Any],
    field_name: str,
    low: float,
    high: float,
) -> list[str]:
    field = body.get(field_name)
    if not isinstance(field, dict):
        return [f"{schema_id}[{index}] is missing {field_name}"]

    value = field.get("value")
    if not isinstance(value, (int, float)):
        return [f"{schema_id}[{index}] has a non-numeric {field_name}"]
    if not low <= float(value) <= high:
        return [f"{schema_id}[{index}] {field_name}={value} is outside [{low}, {high}]"]
    return []


def _validate_duration(
    schema_id: str,
    index: int,
    body: dict[str, Any],
    field_name: str,
) -> list[str]:
    duration = _duration_value(body.get(field_name))
    if duration is None:
        return [f"{schema_id}[{index}] is missing duration field {field_name}"]
    if duration <= 0:
        return [f"{schema_id}[{index}] {field_name} must be positive"]
    return []


def _duration_value(payload: Any) -> int | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get("value")
    if not isinstance(value, (int, float)):
        return None
    return int(round(float(value)))
