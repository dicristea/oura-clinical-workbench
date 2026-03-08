from __future__ import annotations

import json
import math
import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Literal, Optional

Standard = Literal["ieee", "omh"]


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


def build_header(standard: Standard, schema_id: str, source_name: str = "synthetic-oura-simulator") -> Dict[str, Any]:
    """Lightweight header wrapper; both IEEE/OMH examples in your project can be carried this way."""
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
    source_name: str = "synthetic-oura-simulator",
) -> Dict[str, Any]:
    """Build a date-series record like the ambient-temperature example Simona shared."""
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


def simulate_sleep_heart_rate_series(
    start: datetime,
    duration_hours: float = 8.0,
    interval_minutes: int = 5,
    resting_bpm: int = 56,
    variability: float = 4.0,
    seed: int = 42,
) -> List[HeartRateSample]:
    """
    Generate plausible sleep heart-rate samples.
    Pattern: early decline, mid-sleep trough, pre-wake rise, small noise.
    """
    rng = random.Random(seed)
    total_points = max(1, int(duration_hours * 60 / interval_minutes))
    samples: List[HeartRateSample] = []

    for i in range(total_points):
        t = start + timedelta(minutes=i * interval_minutes)
        progress = i / max(1, total_points - 1)

        # Smooth nightly shape: starts slightly higher, dips in middle, rises before wake.
        nightly_curve = (
            2.5 * math.cos(progress * math.pi * 2)  # cyclical movement
            - 2.0 * math.sin(progress * math.pi)     # lower in mid-sleep
            + 1.5 * progress                         # slight rise near wake
        )
        noise = rng.uniform(-variability, variability)
        bpm = max(38, resting_bpm + nightly_curve + noise)
        samples.append(HeartRateSample(timestamp=t, bpm=round(bpm, 1)))

    return samples


# -----------------------------
# Physical activity builders
# -----------------------------

def build_physical_activity_record(
    activity_name: str,
    start: datetime,
    end: datetime,
    *,
    standard: Standard = "ieee",
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

    record = build_header(standard, schema_id)
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


def simulate_daily_activity(
    day: datetime,
    *,
    seed: int = 42,
    steps: Optional[int] = None,
    kcal: Optional[float] = None,
    distance_m: Optional[float] = None,
) -> Dict[str, Any]:
    """Simulate one day of aggregated activity."""
    rng = random.Random(seed)
    start = day.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1) - timedelta(seconds=1)

    if steps is None:
        steps = rng.randint(4500, 11000)
    if distance_m is None:
        # very rough walking conversion
        distance_m = steps * rng.uniform(0.68, 0.80)
    if kcal is None:
        kcal = rng.uniform(180, 650)

    light = rng.randint(30, 120) * 60
    moderate = rng.randint(10, 60) * 60
    vigorous = rng.randint(0, 25) * 60
    duration = light + moderate + vigorous

    return build_physical_activity_record(
        activity_name="all daily activities",
        start=start,
        end=end,
        standard="ieee",
        base_movement_quantity=steps,
        distance_m=distance_m,
        kcal_burned=kcal,
        duration_sec=duration,
        duration_light_sec=light,
        duration_moderate_sec=moderate,
        duration_vigorous_sec=vigorous,
        descriptive_statistic="sum",
        descriptive_statistic_denominator="d",
    )


def simulate_workout_episode(
    start: datetime,
    *,
    activity_name: str = "running",
    duration_min: int = 42,
    seed: int = 7,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    end = start + timedelta(minutes=duration_min)
    distance_m = duration_min * rng.uniform(110, 180)
    kcal = duration_min * rng.uniform(7, 12)
    steps = int(distance_m / rng.uniform(0.75, 1.05))
    speed = distance_m / (duration_min * 60)

    return build_physical_activity_record(
        activity_name=activity_name,
        start=start,
        end=end,
        standard="ieee",
        base_movement_quantity=steps,
        distance_m=distance_m,
        kcal_burned=kcal,
        duration_sec=duration_min * 60,
        average_speed_mps=speed,
        reported_intensity="moderate" if activity_name == "walking" else "vigorous",
    )


# -----------------------------
# Demo / example output
# -----------------------------

def demo() -> Dict[str, Any]:
    sleep_start = datetime(2026, 3, 1, 22, 30, tzinfo=timezone.utc)
    hr_samples = simulate_sleep_heart_rate_series(
        start=sleep_start,
        duration_hours=8,
        interval_minutes=5,
        resting_bpm=54,
        seed=123,
    )
    heart_rate_series_record = build_heart_rate_series(hr_samples, standard="ieee")

    daily_activity_record = simulate_daily_activity(
        datetime(2026, 3, 2, tzinfo=timezone.utc),
        seed=123,
    )

    workout_record = simulate_workout_episode(
        datetime(2026, 3, 2, 17, 45, tzinfo=timezone.utc),
        activity_name="running",
        duration_min=35,
        seed=123,
    )

    return {
        "heart_rate_series": heart_rate_series_record,
        "daily_activity": daily_activity_record,
        "workout_episode": workout_record,
    }


if __name__ == "__main__":
    examples = demo()

    with open("synthetic_examples.json", "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    print("Saved to synthetic_examples.json")
