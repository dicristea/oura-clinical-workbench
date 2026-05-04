from __future__ import annotations

import json
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from data_syn.utils.paths import PROCESSED_DIR, SYNTHETIC_EXAMPLES_PATH
    from data_syn.utils.record_builder import (
        HeartRateSample,
        build_heart_rate_series,
        build_physical_activity_record,
    )
except ModuleNotFoundError:
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from data_syn.utils.paths import PROCESSED_DIR, SYNTHETIC_EXAMPLES_PATH
    from data_syn.utils.record_builder import (
        HeartRateSample,
        build_heart_rate_series,
        build_physical_activity_record,
    )


def simulate_sleep_heart_rate_series(
    start: datetime,
    duration_hours: float = 8.0,
    interval_minutes: int = 5,
    resting_bpm: int = 56,
    variability: float = 4.0,
    seed: int = 42,
) -> List[HeartRateSample]:
    """Generate plausible sleep heart-rate samples for demos."""
    rng = random.Random(seed)
    total_points = max(1, int(duration_hours * 60 / interval_minutes))
    samples: List[HeartRateSample] = []

    for i in range(total_points):
        timestamp = start + timedelta(minutes=i * interval_minutes)
        progress = i / max(1, total_points - 1)
        nightly_curve = (
            2.5 * math.cos(progress * math.pi * 2)
            - 2.0 * math.sin(progress * math.pi)
            + 1.5 * progress
        )
        noise = rng.uniform(-variability, variability)
        bpm = max(38, resting_bpm + nightly_curve + noise)
        samples.append(HeartRateSample(timestamp=timestamp, bpm=round(bpm, 1)))

    return samples


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
        distance_m = steps * rng.uniform(0.68, 0.80)
    if kcal is None:
        kcal = rng.uniform(180, 650)

    light = rng.randint(30, 120) * 60
    moderate = rng.randint(10, 60) * 60
    vigorous = rng.randint(0, 25) * 60
    duration = light + moderate + vigorous

    return build_physical_activity_record(
        activity_name="Total Daily Physical Activity",
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
    """Simulate one workout episode for schema examples."""
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


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    examples = demo()
    SYNTHETIC_EXAMPLES_PATH.write_text(
        json.dumps(examples, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved synthetic examples -> {SYNTHETIC_EXAMPLES_PATH}")


if __name__ == "__main__":
    main()
