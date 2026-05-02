from __future__ import annotations

"""
Convert saved Oura endpoint payloads into:
1. focused extracts of project-relevant source fields, and
2. OMH / IEEE standard records driven by mapping_config.json.

The converter reads raw JSON files from data_syn/outputs/raw/oura/,
applies the configured field mappings, emits schema-grouped standard
records, and reports mapping coverage / gaps in mapping_audit.
"""

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional

try:
    from data_syn.utils.paths import CONVERTED_RECORDS_PATH, MAPPING_CONFIG_PATH, RAW_OURA_DIR
    from data_syn.utils.record_builder import (
        HeartRateSample,
        build_header,
        build_heart_rate_series,
        build_physical_activity_record,
        set_by_path,
    )
except ModuleNotFoundError:
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from data_syn.utils.paths import CONVERTED_RECORDS_PATH, MAPPING_CONFIG_PATH, RAW_OURA_DIR
    from data_syn.utils.record_builder import (
        HeartRateSample,
        build_header,
        build_heart_rate_series,
        build_physical_activity_record,
        set_by_path,
    )

StandardChoice = Literal["omh", "ieee", "both"]

DEFAULT_SOURCE_NAME = "oura-ring-v2"
DEFAULT_MAPPING_PATH = MAPPING_CONFIG_PATH
DEFAULT_INPUT_DIR = RAW_OURA_DIR
DEFAULT_OUTPUT_PATH = CONVERTED_RECORDS_PATH

COMBINED_SCHEMAS = {
    "omh:calories-burned:2.0",
    "omh:oxygen-saturation:2.0",
    "omh:sleep-episode:1.1",
    "omh:total-sleep-time:1.0",
    "ieee:sleep-episode:1.0",
    "ieee:time-in-bed:1.0",
}

CUSTOM_FOCUSED_FIELDS: Dict[str, List[str]] = {
    "daily_activity": [
        "steps",
        "equivalent_walking_distance",
        "low_activity_time",
        "medium_activity_time",
        "high_activity_time",
    ]
}


@dataclass(frozen=True)
class MappingTarget:
    standard: str
    schema_id: str
    field_path: Optional[str]
    field_type: Optional[str]
    notes: Optional[str]


@dataclass(frozen=True)
class MappingRule:
    mapping_id: str
    status: str
    endpoint: str
    source_path: str
    source_field: str
    field_required: bool
    source_schema_name: Optional[str]
    targets: tuple[MappingTarget, ...]
    general_notes: Optional[str]


def _parse_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str):
        return None

    normalized = value.strip()
    if not normalized:
        return None
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _isoformat(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _parse_day(value: Any) -> Optional[date]:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _canonical_schema_id(raw_schema_id: Optional[str], standard_hint: str) -> Optional[str]:
    if not raw_schema_id:
        return None
    if raw_schema_id.startswith(("omh:", "ieee:")):
        return raw_schema_id

    match = re.match(r"^(?P<name>.+)-(?P<version>\d+\.\d+)$", raw_schema_id)
    if not match:
        return None
    return f"{standard_hint}:{match.group('name')}:{match.group('version')}"


def _endpoint_from_source_path(source_path: str) -> str:
    endpoint = source_path.split("/v2/usercollection/", 1)[-1]
    return endpoint.split("/", 1)[0]


def _normalize_field_path(schema_id: str, field_path: str) -> str:
    """
    mapping_config.json stores paths with a schema-specific wrapper under `body`.
    The actual record body is the schema object itself, so:
      body.sleep_episode.wake_after_sleep_onset -> body.wake_after_sleep_onset
      body.calories_burned.kcal_burned        -> body.kcal_burned
      body.heart_rate.heart_rate              -> body.heart_rate
    """
    parts = field_path.split(".")
    if len(parts) < 3 or parts[0] != "body":
        return field_path

    wrapper_name = schema_id.split(":")[1].replace("-", "_")
    if parts[1] == wrapper_name and len(parts) > 2:
        return ".".join(["body", *parts[2:]])
    return field_path


def _new_record(schema_id: str, source_name: str) -> Dict[str, Any]:
    record = build_header(schema_id.split(":", 1)[0], schema_id, source_name=source_name)
    record["body"] = {}
    return record


def _unit_value(value: Any, unit: str, digits: Optional[int] = None) -> Dict[str, Any]:
    numeric = float(value)
    if digits is not None:
        numeric = round(numeric, digits)
    return {"value": numeric, "unit": unit}


def _duration_unit_value(value: Any) -> Dict[str, Any]:
    return {"value": int(round(float(value))), "unit": "sec"}


def _kcal_unit_value(value: Any) -> Dict[str, Any]:
    return {"value": float(value), "unit": "kcal"}


def _day_interval(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    day_value = _parse_day(item.get("day"))
    if day_value is None:
        return None

    tz_source = (
        _parse_datetime(item.get("timestamp"))
        or _parse_datetime(item.get("bedtime_start"))
        or _parse_datetime(item.get("bedtime_end"))
    )
    tzinfo = tz_source.tzinfo if tz_source else timezone.utc

    start = datetime.combine(day_value, time.min, tzinfo=tzinfo)
    end = start + timedelta(days=1)
    return {
        "start_date_time": _isoformat(start),
        "end_date_time": _isoformat(end),
        "duration": _duration_unit_value((end - start).total_seconds()),
    }


def _sleep_interval(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    start_dt = _parse_datetime(item.get("bedtime_start"))
    end_dt = _parse_datetime(item.get("bedtime_end"))

    if start_dt is None and end_dt is None:
        return None

    interval: Dict[str, Any] = {}
    if start_dt is not None:
        interval["start_date_time"] = _isoformat(start_dt)
    if end_dt is not None:
        interval["end_date_time"] = _isoformat(end_dt)

    time_in_bed = item.get("time_in_bed")
    if time_in_bed is not None:
        interval["duration"] = _duration_unit_value(time_in_bed)
    elif start_dt is not None and end_dt is not None:
        interval["duration"] = _duration_unit_value((end_dt - start_dt).total_seconds())

    return interval


def _extract_source_value(rule: MappingRule, item: Dict[str, Any]) -> Any:
    value = item.get(rule.source_field)

    if rule.mapping_id == "dailyspo2model_spo2_percentage" and isinstance(value, dict):
        return value.get("average")

    return value


def _convert_value(rule: MappingRule, target: MappingTarget, raw_value: Any) -> Any:
    if raw_value is None:
        return None

    field_type = (target.field_type or "").strip()
    if field_type == "date-time":
        dt = _parse_datetime(raw_value)
        return _isoformat(dt) if dt else None
    if field_type == "duration_unit_value":
        return _duration_unit_value(raw_value)
    if field_type == "kcal_unit_value":
        return _kcal_unit_value(raw_value)
    if field_type in {"unit_value", "percent_unit_value", "number --> unit_value"}:
        if rule.mapping_id == "sleepmodel_average_breath":
            return _unit_value(raw_value, "breaths/min", digits=1)
        if rule.mapping_id in {"sleepmodel_average_heart_rate", "sleepmodel_lowest_heart_rate"}:
            return _unit_value(raw_value, "beats/min", digits=1)
        if rule.mapping_id == "sleepmodel_efficiency":
            return _unit_value(raw_value, "%")
        if rule.mapping_id == "dailyspo2model_spo2_percentage":
            return _unit_value(raw_value, "%", digits=3)
        return raw_value

    return raw_value


def _mapping_statistic(target: MappingTarget) -> Optional[str]:
    notes = target.notes or ""
    match = re.search(r'descriptive_statistic:\s*"([^"]+)"', notes)
    return match.group(1) if match else None


def _record_group_key(rule: MappingRule, target: MappingTarget) -> str:
    if target.schema_id in COMBINED_SCHEMAS:
        return target.schema_id
    return f"{target.schema_id}::{rule.mapping_id}"


def _record_has_measurement(schema_id: str, body: Dict[str, Any]) -> bool:
    if schema_id == "omh:calories-burned:2.0":
        return body.get("kcal_burned") is not None
    if schema_id == "omh:oxygen-saturation:2.0":
        return body.get("oxygen_saturation") is not None
    if schema_id == "omh:heart-rate:2.0":
        return body.get("heart_rate") is not None
    if schema_id == "omh:respiratory-rate:2.0":
        return body.get("respiratory_rate") is not None
    if schema_id == "omh:total-sleep-time:1.0":
        return body.get("total_sleep_time") is not None
    if schema_id == "ieee:time-in-bed:1.0":
        return body.get("time_in_bed") is not None
    if schema_id in {"omh:sleep-episode:1.1", "ieee:sleep-episode:1.0"}:
        return bool(body)
    return bool(body)


def _is_main_sleep(item: Dict[str, Any]) -> Optional[bool]:
    sleep_type = item.get("type")
    if sleep_type == "long_sleep":
        return True
    if sleep_type == "sleep":
        return False
    return None


def _apply_endpoint_defaults(endpoint: str, schema_id: str, record: Dict[str, Any], item: Dict[str, Any]) -> None:
    body = record["body"]

    if endpoint == "daily_activity" and schema_id == "omh:calories-burned:2.0":
        interval = _day_interval(item)
        if interval:
            set_by_path(record, "body.effective_time_frame.time_interval", interval)
        body.setdefault("descriptive_statistic", "sum")
        body.setdefault("descriptive_statistic_denominator", "d")
        body.setdefault("activity_name", "all daily activities")
        return

    if endpoint == "daily_spo2" and schema_id == "omh:oxygen-saturation:2.0":
        interval = _day_interval(item)
        if interval:
            set_by_path(record, "body.effective_time_frame.time_interval", interval)
        body.setdefault("descriptive_statistic", "average")
        return

    if endpoint == "sleep":
        sleep_interval = _sleep_interval(item)
        if schema_id in {
            "omh:respiratory-rate:2.0",
            "omh:heart-rate:2.0",
            "omh:sleep-episode:1.1",
            "omh:total-sleep-time:1.0",
            "ieee:sleep-episode:1.0",
            "ieee:time-in-bed:1.0",
        }:
            if sleep_interval and "effective_time_frame" not in body:
                set_by_path(record, "body.effective_time_frame.time_interval", sleep_interval)

        if schema_id in {"omh:sleep-episode:1.1", "ieee:sleep-episode:1.0", "ieee:time-in-bed:1.0"}:
            main_sleep = _is_main_sleep(item)
            if main_sleep is not None:
                body.setdefault("is_main_sleep", main_sleep)


def _expand_sleep_heart_rate_series(item: Dict[str, Any], source_name: str) -> List[Dict[str, Any]]:
    block = item.get("heart_rate")
    if not isinstance(block, dict):
        return []

    base_timestamp = _parse_datetime(block.get("timestamp"))
    items = block.get("items")
    interval = block.get("interval")
    if base_timestamp is None or not isinstance(items, list):
        return []

    try:
        interval_seconds = int(float(interval))
    except (TypeError, ValueError):
        return []

    records: List[Dict[str, Any]] = []
    for index, bpm in enumerate(items):
        if bpm is None:
            continue
        sample_time = base_timestamp + timedelta(seconds=index * interval_seconds)
        record = _new_record("omh:heart-rate:2.0", source_name=source_name)
        set_by_path(record, "body.heart_rate", _unit_value(bpm, "beats/min", digits=1))
        set_by_path(record, "body.effective_time_frame.date_time", _isoformat(sample_time))
        records.append(record)

    return records


def _build_ieee_sleep_heart_rate_series(item: Dict[str, Any], source_name: str) -> Optional[Dict[str, Any]]:
    block = item.get("heart_rate")
    if not isinstance(block, dict):
        return None

    base_timestamp = _parse_datetime(block.get("timestamp"))
    items = block.get("items")
    interval = block.get("interval")
    if base_timestamp is None or not isinstance(items, list):
        return None

    try:
        interval_seconds = int(float(interval))
    except (TypeError, ValueError):
        return None

    samples: List[HeartRateSample] = []
    for index, bpm in enumerate(items):
        if bpm is None:
            continue
        samples.append(
            HeartRateSample(
                timestamp=base_timestamp + timedelta(seconds=index * interval_seconds),
                bpm=float(bpm),
            )
        )

    if not samples:
        return None

    return build_heart_rate_series(samples, standard="ieee", source_name=source_name)


def _build_total_daily_physical_activity_record(
    item: Dict[str, Any],
    source_name: str,
) -> Optional[Dict[str, Any]]:
    day_value = _parse_day(item.get("day"))
    if day_value is None:
        return None

    tz_source = _parse_datetime(item.get("timestamp"))
    tzinfo = tz_source.tzinfo if tz_source else timezone.utc
    start = datetime.combine(day_value, time.min, tzinfo=tzinfo)
    end = start + timedelta(days=1)

    steps = item.get("steps")
    distance_m = item.get("equivalent_walking_distance")
    kcal_burned = item.get("active_calories")
    duration_light_sec = item.get("low_activity_time")
    duration_moderate_sec = item.get("medium_activity_time")
    duration_vigorous_sec = item.get("high_activity_time")

    if all(value is None for value in (steps, distance_m, kcal_burned)):
        return None

    total_duration = None
    duration_values = [
        value
        for value in (duration_light_sec, duration_moderate_sec, duration_vigorous_sec)
        if value is not None
    ]
    if duration_values:
        total_duration = int(sum(float(value) for value in duration_values))

    return build_physical_activity_record(
        activity_name="Total Daily Physical Activity",
        start=start,
        end=end,
        standard="ieee",
        source_name=source_name,
        base_movement_quantity=int(steps) if steps is not None else None,
        distance_m=float(distance_m) if distance_m is not None else None,
        kcal_burned=float(kcal_burned) if kcal_burned is not None else None,
        duration_sec=total_duration,
        duration_light_sec=int(duration_light_sec) if duration_light_sec is not None else None,
        duration_moderate_sec=int(duration_moderate_sec) if duration_moderate_sec is not None else None,
        duration_vigorous_sec=int(duration_vigorous_sec) if duration_vigorous_sec is not None else None,
        descriptive_statistic="sum",
        descriptive_statistic_denominator="d",
    )


def _summarize_nested_value(value: Any) -> Any:
    if isinstance(value, dict) and isinstance(value.get("items"), list):
        items = value["items"]
        preview = items[:10]
        summary = {key: item for key, item in value.items() if key != "items"}
        summary["item_count"] = len(items)
        summary["non_null_count"] = sum(1 for item in items if item is not None)
        summary["items_preview"] = preview
        return summary
    return value


def _payload_items(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return [item for item in payload["data"] if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_mapping_rules(
    mapping_path: str | Path = DEFAULT_MAPPING_PATH,
    *,
    standard: StandardChoice = "both",
) -> List[MappingRule]:
    config = _read_json(Path(mapping_path))
    rules: List[MappingRule] = []
    requested_standards = {"omh", "ieee"} if standard == "both" else {standard}

    for entry in config.get("mappings", []):
        source = entry.get("source", {})
        source_path = source.get("path")
        source_field = source.get("field")
        if not source_path or not source_field:
            continue

        targets: List[MappingTarget] = []
        for standard_name in requested_standards:
            target_node = entry.get(standard_name)
            if not isinstance(target_node, dict):
                continue

            schema_id = _canonical_schema_id(target_node.get("schema_id"), standard_name)
            if not schema_id:
                continue

            targets.append(
                MappingTarget(
                    standard=standard_name,
                    schema_id=schema_id,
                    field_path=target_node.get("field_path"),
                    field_type=target_node.get("field_type"),
                    notes=target_node.get("notes"),
                )
            )

        rules.append(
            MappingRule(
                mapping_id=entry["id"],
                status=entry.get("status", ""),
                endpoint=_endpoint_from_source_path(source_path),
                source_path=source_path,
                source_field=source_field,
                field_required=bool(source.get("field_required")),
                source_schema_name=source.get("response_schema_name"),
                targets=tuple(targets),
                general_notes=(entry.get("notes") or {}).get("general"),
            )
        )

    return rules


def load_source_payloads(input_dir: str | Path, endpoints: Iterable[str]) -> tuple[Dict[str, Any], List[str]]:
    payloads: Dict[str, Any] = {}
    missing_files: List[str] = []
    input_dir = Path(input_dir)

    for endpoint in sorted(set(endpoints)):
        path = input_dir / f"{endpoint}.json"
        if not path.exists():
            missing_files.append(str(path))
            continue
        payloads[endpoint] = _read_json(path)

    return payloads, missing_files


def build_focused_data(
    payloads: Dict[str, Any],
    rules: Iterable[MappingRule],
) -> Dict[str, Dict[str, Any]]:
    rules_by_endpoint: Dict[str, List[MappingRule]] = defaultdict(list)
    for rule in rules:
        rules_by_endpoint[rule.endpoint].append(rule)

    focused: Dict[str, Dict[str, Any]] = {}
    for endpoint, endpoint_rules in sorted(rules_by_endpoint.items()):
        payload = payloads.get(endpoint)
        if payload is None:
            continue

        fields = sorted({rule.source_field for rule in endpoint_rules} | set(CUSTOM_FOCUSED_FIELDS.get(endpoint, [])))
        items: List[Dict[str, Any]] = []
        for raw_item in _payload_items(payload):
            focused_item: Dict[str, Any] = {
                key: raw_item[key]
                for key in ("id", "day", "timestamp", "bedtime_start", "bedtime_end")
                if key in raw_item
            }
            for field in fields:
                if field in raw_item:
                    focused_item[field] = _summarize_nested_value(raw_item[field])
            items.append(focused_item)

        focused[endpoint] = {
            "source_file": f"{endpoint}.json",
            "field_count": len(fields),
            "fields": fields,
            "item_count": len(items),
            "items": items,
        }

    return focused


def _ensure_fieldless_record(
    endpoint: str,
    rule: MappingRule,
    target: MappingTarget,
    records_by_group: Dict[str, Dict[str, Any]],
    source_name: str,
) -> bool:
    if endpoint == "sleep" and target.schema_id == "ieee:time-in-bed:1.0":
        group_key = _record_group_key(rule, target)
        records_by_group.setdefault(group_key, _new_record(target.schema_id, source_name))
        return True
    return False


def convert_endpoint_items(
    endpoint: str,
    items: Iterable[Dict[str, Any]],
    rules: Iterable[MappingRule],
    *,
    source_name: str,
) -> Dict[str, Any]:
    records_by_schema: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    converted_mapping_ids: set[str] = set()
    missing_source_fields: set[str] = set()
    unimplemented_targets: List[Dict[str, Any]] = []

    endpoint_rules = list(rules)
    if not endpoint_rules:
        return {
            "records_by_schema": {},
            "converted_mapping_ids": [],
            "missing_source_fields": [],
            "unimplemented_targets": [],
        }

    for item in items:
        records_by_group: Dict[str, Dict[str, Any]] = {}
        emitted_custom_records: set[str] = set()

        for rule in endpoint_rules:
            source_present = rule.source_field in item and item.get(rule.source_field) is not None
            if not source_present:
                missing_source_fields.add(f"{endpoint}.{rule.source_field}")

            for target in rule.targets:
                if not source_present:
                    if target.field_path is None:
                        _ensure_fieldless_record(endpoint, rule, target, records_by_group, source_name)
                    continue

                if endpoint == "daily_activity" and target.schema_id == "ieee:physical-activity:1.0":
                    if target.schema_id not in emitted_custom_records:
                        record = _build_total_daily_physical_activity_record(item, source_name=source_name)
                        if record is not None:
                            records_by_schema[target.schema_id].append(record)
                            emitted_custom_records.add(target.schema_id)
                    if target.schema_id in emitted_custom_records:
                        converted_mapping_ids.add(rule.mapping_id)
                    continue

                if (
                    endpoint == "sleep"
                    and rule.mapping_id == "sleepmodel_heart_rate"
                    and target.schema_id == "ieee:heart-rate:1.0"
                ):
                    if target.schema_id not in emitted_custom_records:
                        series_record = _build_ieee_sleep_heart_rate_series(item, source_name=source_name)
                        if series_record is not None:
                            records_by_schema[target.schema_id].append(series_record)
                            emitted_custom_records.add(target.schema_id)
                    if target.schema_id in emitted_custom_records:
                        converted_mapping_ids.add(rule.mapping_id)
                    continue

                if target.field_path is None:
                    if _ensure_fieldless_record(endpoint, rule, target, records_by_group, source_name):
                        converted_mapping_ids.add(rule.mapping_id)
                        continue

                    unimplemented_targets.append(
                        {
                            "mapping_id": rule.mapping_id,
                            "endpoint": endpoint,
                            "schema_id": target.schema_id,
                            "reason": "schema target exists but field_path is not defined in mapping_config.json",
                            "notes": target.notes,
                        }
                    )
                    continue

                raw_value = _extract_source_value(rule, item)
                converted_value = _convert_value(rule, target, raw_value)
                if converted_value is None:
                    continue

                group_key = _record_group_key(rule, target)
                record = records_by_group.setdefault(group_key, _new_record(target.schema_id, source_name))
                normalized_path = _normalize_field_path(target.schema_id, target.field_path)
                set_by_path(record, normalized_path, converted_value)

                statistic = _mapping_statistic(target)
                if statistic:
                    prefix = ".".join(normalized_path.split(".")[:-1]) or "body"
                    set_by_path(record, f"{prefix}.descriptive_statistic", statistic)

                converted_mapping_ids.add(rule.mapping_id)

        for record in records_by_group.values():
            schema_id = ":".join(
                [
                    record["header"]["schema_id"]["namespace"],
                    record["header"]["schema_id"]["name"],
                    record["header"]["schema_id"]["version"],
                ]
            )
            _apply_endpoint_defaults(endpoint, schema_id, record, item)
            if _record_has_measurement(schema_id, record["body"]):
                records_by_schema[schema_id].append(record)

    return {
        "records_by_schema": dict(records_by_schema),
        "converted_mapping_ids": sorted(converted_mapping_ids),
        "missing_source_fields": sorted(missing_source_fields),
        "unimplemented_targets": unimplemented_targets,
    }


def convert_oura_directory(
    *,
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    mapping_path: str | Path = DEFAULT_MAPPING_PATH,
    standard: StandardChoice = "both",
    source_name: str = DEFAULT_SOURCE_NAME,
) -> Dict[str, Any]:
    rules = load_mapping_rules(mapping_path, standard=standard)
    endpoints = [rule.endpoint for rule in rules]
    payloads, missing_files = load_source_payloads(input_dir, endpoints)
    focused_data = build_focused_data(payloads, rules)

    rules_by_endpoint: Dict[str, List[MappingRule]] = defaultdict(list)
    for rule in rules:
        rules_by_endpoint[rule.endpoint].append(rule)

    records_by_schema: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    converted_mapping_ids: set[str] = set()
    missing_source_fields: set[str] = set()
    unimplemented_targets: List[Dict[str, Any]] = []

    for endpoint, endpoint_rules in sorted(rules_by_endpoint.items()):
        payload = payloads.get(endpoint)
        if payload is None:
            continue

        result = convert_endpoint_items(
            endpoint,
            _payload_items(payload),
            endpoint_rules,
            source_name=source_name,
        )
        for schema_id, records in result["records_by_schema"].items():
            records_by_schema[schema_id].extend(records)
        converted_mapping_ids.update(result["converted_mapping_ids"])
        missing_source_fields.update(result["missing_source_fields"])
        unimplemented_targets.extend(result["unimplemented_targets"])

    deduped_unimplemented_targets: List[Dict[str, Any]] = []
    seen_unimplemented: set[tuple[str, str, str]] = set()
    for item in unimplemented_targets:
        key = (item["mapping_id"], item["endpoint"], item["schema_id"])
        if key in seen_unimplemented:
            continue
        seen_unimplemented.add(key)
        deduped_unimplemented_targets.append(item)

    configured_mapping_ids = {rule.mapping_id for rule in rules}
    mapped_target_ids = {rule.mapping_id for rule in rules if rule.targets}
    config_gap_ids = sorted(rule.mapping_id for rule in rules if not rule.targets)

    summary = {
        "configured_mappings": len(configured_mapping_ids),
        "mappings_with_schema_targets": len(mapped_target_ids),
        "converted_mapping_count": len(converted_mapping_ids),
        "config_gap_count": len(config_gap_ids),
        "missing_source_file_count": len(missing_files),
        "missing_source_field_count": len(missing_source_fields),
        "record_count": sum(len(records) for records in records_by_schema.values()),
        "schema_count": len(records_by_schema),
    }

    return {
        "summary": summary,
        "focused_data": focused_data,
        "records_by_schema": dict(records_by_schema),
        "counts_by_schema": {schema_id: len(records) for schema_id, records in records_by_schema.items()},
        "mapping_audit": {
            "converted_mapping_ids": sorted(converted_mapping_ids),
            "not_yet_mapped_in_config": config_gap_ids,
            "missing_source_files": missing_files,
            "missing_source_fields": sorted(missing_source_fields),
            "unimplemented_targets": deduped_unimplemented_targets,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert saved Oura JSON files into focused extracts plus OMH/IEEE standard records."
    )
    parser.add_argument("--input-dir", type=str, default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--mapping-config", type=str, default=str(DEFAULT_MAPPING_PATH))
    parser.add_argument("--source-name", type=str, default=DEFAULT_SOURCE_NAME)
    parser.add_argument("--standard", choices=["omh", "ieee", "both"], default="both")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    converted = convert_oura_directory(
        input_dir=args.input_dir,
        mapping_path=args.mapping_config,
        standard=args.standard,
        source_name=args.source_name,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(converted, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved converted records -> {output_path}")


if __name__ == "__main__":
    main()
