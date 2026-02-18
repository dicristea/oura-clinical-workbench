from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class DataSource(Enum):
    OURA = "oura"
    PPMI = "ppmi"
    SYNTHEA = "synthea"


# Default feature groups keyed by DataSource.
# Adapters may override these when constructing a PatientTimeSeries.
DEFAULT_FEATURE_GROUPS: dict[DataSource, dict[str, list[str]]] = {
    DataSource.OURA: {
        "Sleep Architecture": ["rem_sleep_pct", "deep_sleep_pct", "sleep_latency"],
        "Physiological": ["hrv_balance", "body_temp_deviation", "resting_hr"],
        "Activity": ["step_count", "inactivity_alerts"],
    },
    DataSource.PPMI: {
        "Genetic": ["gba_mutation", "lrrk2_mutation", "apoe_status"],
        "Biofluid": ["csf_alpha_synuclein", "amyloid_beta", "total_tau"],
        "Clinical": ["baseline_updrs", "epworth_sleep", "schwab_england_adl"],
        "Imaging": ["datscan"],
    },
    DataSource.SYNTHEA: {},
}


@dataclass
class PatientTimeSeries:
    """Unified patient data container used by all data adapters and the ML layer.

    All data sources (Oura, PPMI, Synthea) produce PatientTimeSeries objects so
    the rest of the application can remain source-agnostic.

    Attributes:
        patient_id:     Unique identifier for this patient (MRN, PPMI ID, etc.).
        data_source:    Which data source produced this record.
        static_features: Time-invariant attributes — demographics, genetic status,
                         baseline clinical scores, etc.
                         Example: {"age": 65, "gba_mutation": True, "sex": "M"}
        time_series:    DatetimeIndex DataFrame where each column is a feature and
                         each row is one observation (daily for Oura, per-visit for
                         PPMI).  Missing values are NaN.
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

    def __post_init__(self) -> None:
        # Validate time_series index is datetime (allow empty DataFrame)
        if not self.time_series.empty and not isinstance(
            self.time_series.index, pd.DatetimeIndex
        ):
            raise ValueError(
                "PatientTimeSeries.time_series must have a DatetimeIndex. "
                f"Got {type(self.time_series.index).__name__}."
            )

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

        cutoff = self.time_series.index.max() - pd.Timedelta(days=days - 1)
        return self.time_series.loc[self.time_series.index >= cutoff].copy()
