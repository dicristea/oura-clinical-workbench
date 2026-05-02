from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone

from demo_data.omh_ieee_generator.fhir import SyntheaPatientProfile
from demo_data.omh_ieee_generator.record_builder import HeartRateSample


@dataclass(frozen=True)
class SyntheticPatientDay:
    day: date
    sleep_start: datetime
    sleep_end: datetime
    time_in_bed_sec: int
    total_sleep_time_sec: int
    wake_after_sleep_onset_sec: int
    latency_to_sleep_onset_sec: int
    deep_sleep_duration_sec: int
    rem_sleep_duration_sec: int
    light_sleep_duration_sec: int
    sleep_efficiency_pct: float
    average_heart_rate_bpm: float
    respiratory_rate_bpm: float
    oxygen_saturation_pct: float
    steps: int
    distance_m: float
    active_kcal: float
    light_activity_sec: int
    moderate_activity_sec: int
    vigorous_activity_sec: int
    heart_rate_samples: tuple[HeartRateSample, ...]


def generate_patient_days(
    profile: SyntheaPatientProfile,
    *,
    days: int,
    end_date: date,
    seed: int,
) -> list[SyntheticPatientDay]:
    if days <= 0:
        raise ValueError(f"days must be positive, got {days}")

    rng = random.Random(f"{seed}:{profile.patient_id}")
    risk_score = _risk_score(profile)

    records: list[SyntheticPatientDay] = []
    start_day = end_date - timedelta(days=days - 1)
    for offset in range(days):
        day = start_day + timedelta(days=offset)
        records.append(_generate_day(profile, day=day, day_index=offset, risk_score=risk_score, rng=rng))
    return records


def _generate_day(
    profile: SyntheaPatientProfile,
    *,
    day: date,
    day_index: int,
    risk_score: float,
    rng: random.Random,
) -> SyntheticPatientDay:
    age = profile.age or 50
    weekday_factor = 0.35 if day.weekday() >= 5 else 0.0
    trend = (day_index % 14) / 13 if day_index else 0.0

    bedtime_hour = 22 + rng.uniform(0.0, 1.5) + 0.25 * risk_score
    bedtime_minute = int((bedtime_hour % 1) * 60)
    sleep_start = datetime.combine(
        day - timedelta(days=1),
        time(hour=int(bedtime_hour), minute=bedtime_minute, tzinfo=timezone.utc),
    )

    latency_sec = int((10 + 25 * risk_score + weekday_factor * 5 + rng.uniform(-3, 8)) * 60)
    wake_after_sec = int((20 + 40 * risk_score + rng.uniform(-5, 15)) * 60)
    time_in_bed_sec = int((7.6 - 0.01 * max(age - 45, 0) - 0.4 * risk_score + rng.uniform(-0.5, 0.5)) * 3600)
    time_in_bed_sec = max(6 * 3600, min(9 * 3600, time_in_bed_sec))

    total_sleep_time_sec = time_in_bed_sec - latency_sec - wake_after_sec
    total_sleep_time_sec = max(4 * 3600, min(time_in_bed_sec - 300, total_sleep_time_sec))

    deep_pct = max(0.12, min(0.24, 0.20 - 0.06 * risk_score + rng.uniform(-0.02, 0.02)))
    rem_pct = max(0.14, min(0.28, 0.22 - 0.05 * risk_score + rng.uniform(-0.02, 0.02)))
    deep_sleep_duration_sec = int(total_sleep_time_sec * deep_pct)
    rem_sleep_duration_sec = int(total_sleep_time_sec * rem_pct)
    light_sleep_duration_sec = max(0, total_sleep_time_sec - deep_sleep_duration_sec - rem_sleep_duration_sec)

    sleep_efficiency_pct = round(total_sleep_time_sec / time_in_bed_sec * 100, 1)
    sleep_end = sleep_start + timedelta(seconds=time_in_bed_sec)

    activity_drag = 1200 * risk_score + max(age - 55, 0) * 55
    steps = int(max(1500, 9000 - activity_drag + 700 * math.sin(day_index / 4) + rng.uniform(-1800, 1800)))
    distance_m = round(steps * rng.uniform(0.68, 0.80), 1)
    active_kcal = round(max(120.0, 260 + steps * 0.035 + rng.uniform(-60, 80)), 1)

    light_activity_sec = int(max(20, 95 - 25 * risk_score + rng.uniform(-20, 20)) * 60)
    moderate_activity_sec = int(max(5, 35 - 15 * risk_score + rng.uniform(-10, 15)) * 60)
    vigorous_activity_sec = int(max(0, 12 - 10 * risk_score + rng.uniform(-6, 8)) * 60)

    average_heart_rate_bpm = round(55 + 10 * risk_score + max(age - 50, 0) * 0.12 + rng.uniform(-4, 4), 1)
    respiratory_rate_bpm = round(13 + 2.5 * risk_score + rng.uniform(-1.0, 1.5), 1)
    oxygen_saturation_pct = round(max(92.5, 98.2 - 1.8 * risk_score + rng.uniform(-0.7, 0.4)), 1)

    heart_rate_samples = tuple(
        _generate_heart_rate_samples(
            start=sleep_start + timedelta(seconds=latency_sec),
            duration_sec=total_sleep_time_sec,
            average_bpm=average_heart_rate_bpm,
            risk_score=risk_score,
            trend=trend,
            rng=rng,
        )
    )

    return SyntheticPatientDay(
        day=day,
        sleep_start=sleep_start,
        sleep_end=sleep_end,
        time_in_bed_sec=time_in_bed_sec,
        total_sleep_time_sec=total_sleep_time_sec,
        wake_after_sleep_onset_sec=wake_after_sec,
        latency_to_sleep_onset_sec=latency_sec,
        deep_sleep_duration_sec=deep_sleep_duration_sec,
        rem_sleep_duration_sec=rem_sleep_duration_sec,
        light_sleep_duration_sec=light_sleep_duration_sec,
        sleep_efficiency_pct=sleep_efficiency_pct,
        average_heart_rate_bpm=average_heart_rate_bpm,
        respiratory_rate_bpm=respiratory_rate_bpm,
        oxygen_saturation_pct=oxygen_saturation_pct,
        steps=steps,
        distance_m=distance_m,
        active_kcal=active_kcal,
        light_activity_sec=light_activity_sec,
        moderate_activity_sec=moderate_activity_sec,
        vigorous_activity_sec=vigorous_activity_sec,
        heart_rate_samples=heart_rate_samples,
    )


def _risk_score(profile: SyntheaPatientProfile) -> float:
    age = profile.age or 50
    keywords = {
        "diabetes": 0.10,
        "chronic kidney": 0.12,
        "hypertension": 0.08,
        "heart": 0.10,
        "copd": 0.12,
        "asthma": 0.06,
        "depression": 0.05,
        "sleep": 0.10,
        "obesity": 0.10,
    }

    score = 0.12 + max(age - 40, 0) / 200
    for condition in profile.conditions:
        lowered = condition.lower()
        for keyword, weight in keywords.items():
            if keyword in lowered:
                score += weight
    score += min(len(profile.conditions), 5) * 0.04
    return max(0.0, min(0.85, score))


def _generate_heart_rate_samples(
    *,
    start: datetime,
    duration_sec: int,
    average_bpm: float,
    risk_score: float,
    trend: float,
    rng: random.Random,
) -> list[HeartRateSample]:
    interval_minutes = 5
    total_points = max(12, int(duration_sec / (interval_minutes * 60)))
    samples: list[HeartRateSample] = []

    for index in range(total_points):
        progress = index / max(1, total_points - 1)
        timestamp = start + timedelta(minutes=index * interval_minutes)
        nightly_curve = (
            -4.0 * math.sin(progress * math.pi)
            + 2.0 * math.cos(progress * math.pi * 2)
            + trend
        )
        noise = rng.uniform(-2.2 - risk_score, 2.2 + risk_score)
        bpm = max(42.0, average_bpm + nightly_curve + noise)
        samples.append(HeartRateSample(timestamp=timestamp, bpm=round(bpm, 1)))

    return samples
