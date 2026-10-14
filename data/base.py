from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class DataSource(Enum):
    OURA = "oura"
    SYNTHEA = "synthea"
    HEPATOLOGY = "hepatology"
    OPEN_WEARABLES = "open_wearables"   # Open Wearables unified API (Oura Q1 2026)


# Default feature groups keyed by DataSource.
# Adapters may override these when constructing a PatientTimeSeries.
DEFAULT_FEATURE_GROUPS: dict[DataSource, dict[str, list[str]]] = {
    DataSource.OURA: {
        "Sleep Architecture": ["rem_sleep_pct", "deep_sleep_pct", "sleep_latency"],
        "Physiological": ["hrv_balance", "body_temp_deviation", "resting_hr"],
        "Activity": ["step_count", "inactivity_alerts"],
    },
    DataSource.SYNTHEA: {
        "Vital Signs":       ["heart_rate", "systolic_bp", "diastolic_bp", "respiratory_rate", "body_temperature"],
        "Body Composition":  ["body_weight_kg", "bmi"],
        "Metabolic":         ["glucose_mgdl", "hba1c_pct", "total_cholesterol_mgdl", "ldl_cholesterol_mgdl"],
    },
    DataSource.HEPATOLOGY: {
        "Physiological": [
            "average_heart_rate", "average_hrv", "respiratory_rate", "spo2_pct",
        ],
        "Activity": ["step_count", "average_met_minutes", "total_calories"],
        "Data Quality": ["valid80", "non_wear_pct", "non_wear_time"],
    },
    # Open Wearables normalized field names — same groups as Oura, different API keys.
    # Wire up by swapping OuraAdapter → OpenWearablesAdapter once Oura integration ships (Q1 2026).
    DataSource.OPEN_WEARABLES: {
        "Sleep Architecture": ["rem_sleep_pct", "deep_sleep_pct", "sleep_latency"],
        "Physiological":      ["hrv_rmssd", "body_temp_deviation", "resting_heart_rate"],
        "Activity":           ["steps", "inactivity_alerts"],
    },
}


@dataclass
class PatientTimeSeries:
    """Unified patient data container used by all data adapters and the ML layer.

    All data sources (Oura, Synthea) produce PatientTimeSeries objects so
    the rest of the application can remain source-agnostic.

    Attributes:
        patient_id:     Unique identifier for this patient (MRN, etc.).
        data_source:    Which data source produced this record.
        static_features: Time-invariant attributes — demographics, genetic status,
                         baseline clinical scores, etc.
                         Example: {"age": 65, "sex": "M"}
        time_series:    DataFrame where each column is a feature and each row is one
                         observation. Calendar-based sources use a DatetimeIndex;
                         relative longitudinal studies may use an integer study-day
                         index. Missing values are NaN.
        time_axis:      ``"calendar_date"`` for a DatetimeIndex or ``"study_day"``
                         for a relative integer day index.
        metadata:       Free-form dict for cohort info, risk level, enrollment
                         dates, study arm, etc.
                         Example: {"cohort": "de_novo", "risk_level": "high",
                                   "enrollment_date": "2021-03-15"}
        feature_groups: Maps group label → list of column names in time_series.
                         Controls how the UI renders feature checkboxes and the
                         data explorer.  Defaults to the canonical groups for
                         this data source if not provided.
    """

    patient_id: str
    data_source: DataSource
    static_features: dict = field(default_factory=dict)
    time_series: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: dict = field(default_factory=dict)
    feature_groups: dict[str, list[str]] = field(default_factory=dict)
    time_axis: str = "calendar_date"

    def __post_init__(self) -> None:
        if self.time_axis not in {"calendar_date", "study_day"}:
            raise ValueError(
                "time_axis must be 'calendar_date' or 'study_day', "
                f"got {self.time_axis!r}."
            )

        # Validate the time index while allowing empty DataFrames.
        if not self.time_series.empty:
            if self.time_axis == "calendar_date" and not isinstance(
                self.time_series.index, pd.DatetimeIndex
            ):
                raise ValueError(
                    "calendar_date time series must have a DatetimeIndex. "
                    f"Got {type(self.time_series.index).__name__}."
                )
            if self.time_axis == "study_day":
                numeric_index = pd.to_numeric(self.time_series.index, errors="coerce")
                if pd.isna(numeric_index).any():
                    raise ValueError("study_day time series must have a numeric index.")

        # Apply default feature groups when none are supplied
        if not self.feature_groups:
            self.feature_groups = DEFAULT_FEATURE_GROUPS.get(self.data_source, {})

    # ── Public helpers ────────────────────────────────────────────────────────

    def get_feature_list(self) -> list[str]:
        """Return all feature names across every group (preserves group order)."""
        seen: set[str] = set()
        features: list[str] = []
        for group_features in self.feature_groups.values():
            for f in group_features:
                if f not in seen:
                    seen.add(f)
                    features.append(f)
        return features

    def get_analysis_window(self, days: int) -> pd.DataFrame:
        """Return a copy of time_series limited to the most recent *days* days.

        Args:
            days: Number of calendar days to include (counting back from the
                  latest date present in the DataFrame).

        Returns:
            A filtered DataFrame with the same columns and a DatetimeIndex.
            Returns an empty DataFrame (same columns) if time_series is empty
            or no rows fall within the window.

        Raises:
            ValueError: If *days* is not a positive integer.
        """
        if days <= 0:
            raise ValueError(f"days must be a positive integer, got {days}.")

        if self.time_series.empty:
            return self.time_series.copy()

        if self.time_axis == "study_day":
            cutoff = self.time_series.index.max() - (days - 1)
        else:
            cutoff = self.time_series.index.max() - pd.Timedelta(days=days - 1)
        return self.time_series.loc[self.time_series.index >= cutoff].copy()
