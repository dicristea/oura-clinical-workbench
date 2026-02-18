from __future__ import annotations

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from data.base import DataSource, PatientTimeSeries
from data.feature_registry import get_feature_groups_for_source


# ---------------------------------------------------------------------------
# Column-name mapping: registry feature name → candidate Excel column names
# (checked left-to-right; first match wins)
# ---------------------------------------------------------------------------
_EXCEL_COLUMN_MAP: dict[str, list[str]] = {
    "rem_sleep_pct":       ["rem_sleep_pct", "rem_pct", "rem_sleep_percentage", "rem_%"],
    "deep_sleep_pct":      ["deep_sleep_pct", "deep_pct", "deep_sleep_percentage", "deep_%"],
    "sleep_latency":       ["sleep_latency", "latency_min", "sleep_onset_latency"],
    "hrv_balance":         ["hrv_balance", "hrv", "hrv_rmssd", "heart_rate_variability"],
    "body_temp_deviation": ["body_temp_deviation", "temp_deviation", "temperature_deviation"],
    "resting_hr":          ["resting_hr", "resting_heart_rate", "lowest_hr", "min_heart_rate"],
    "step_count":          ["step_count", "steps", "total_steps", "daily_steps"],
    "inactivity_alerts":   ["inactivity_alerts", "inactivity_alert_count", "inactivity_count"],
}

# Candidate column names for the observation date column
_DATE_COLUMN_CANDIDATES = [
    "flowsheet_record_date", "record_date", "date", "observation_date", "entry_date",
]

# Candidate column names for the patient MRN
_MRN_COLUMN_CANDIDATES = ["mrn", "patient_mrn", "patient_id", "MRN"]


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column name that exists in *df*, or None."""
    cols_lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in cols_lower:
            return cols_lower[candidate.lower()]
    return None


def _compute_risk_level(ts: pd.DataFrame) -> str:
    """Derive a clinical risk level from HRV and REM sleep trends.

    Rules (mirrors the clinical context in CLAUDE.md):
        high   – mean HRV < 30 ms  OR  mean REM % < 16 %
        medium – mean HRV 30–45 ms OR  mean REM % 16–20 %
        low    – mean HRV > 45 ms  AND mean REM % > 20 %

    Falls back gracefully when either feature is absent.
    """
    hrv_mean = ts["hrv_balance"].mean() if "hrv_balance" in ts.columns else None
    rem_mean = ts["rem_sleep_pct"].mean() if "rem_sleep_pct" in ts.columns else None

    high_flags = []
    medium_flags = []

    if hrv_mean is not None and not np.isnan(hrv_mean):
        if hrv_mean < 30:
            high_flags.append(True)
        elif hrv_mean <= 45:
            medium_flags.append(True)

    if rem_mean is not None and not np.isnan(rem_mean):
        if rem_mean < 16:
            high_flags.append(True)
        elif rem_mean <= 20:
            medium_flags.append(True)

    if high_flags:
        return "high"
    if medium_flags:
        return "medium"
    return "low"


class OuraAdapter:
    """Loads Oura Ring data from an Excel flowsheet or generates demo data.

    All output is a ``PatientTimeSeries`` using the feature names defined in
    ``data.feature_registry``, so the ML layer and UI stay source-agnostic.
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def load_from_excel(self, filepath: str, patient_mrn: str) -> PatientTimeSeries:
        """Build a PatientTimeSeries from a flowsheet Excel file.

        The method is intentionally lenient: it maps whatever Oura-related
        columns it finds in the sheet to the registry feature names and leaves
        the rest as NaN.  This preserves compatibility with the existing
        ``load_patient_data()`` function in app.py while letting new code use
        the structured PatientTimeSeries interface.

        Args:
            filepath:    Path to the ``.xlsx`` flowsheet file.
            patient_mrn: MRN string used to filter the sheet to one patient.

        Returns:
            A PatientTimeSeries with a DatetimeIndex time_series.

        Raises:
            FileNotFoundError: If *filepath* does not exist.
            ValueError: If *patient_mrn* is not found in the sheet, or if no
                        usable date column can be identified.
        """
        df = pd.read_excel(filepath)

        # -- Locate the MRN column and filter to this patient -----------------
        mrn_col = _find_column(df, _MRN_COLUMN_CANDIDATES)
        if mrn_col is None:
            raise ValueError(
                f"Could not find an MRN column in {filepath}. "
                f"Expected one of: {_MRN_COLUMN_CANDIDATES}"
            )

        # Normalise MRN comparison (both to string, stripped)
        df[mrn_col] = df[mrn_col].astype(str).str.strip()
        patient_df = df[df[mrn_col] == str(patient_mrn).strip()].copy()

        if patient_df.empty:
            raise ValueError(
                f"MRN '{patient_mrn}' not found in {filepath}."
            )

        # -- Locate date column and build DatetimeIndex -----------------------
        date_col = _find_column(patient_df, _DATE_COLUMN_CANDIDATES)
        if date_col is None:
            raise ValueError(
                f"Could not find a date column for patient {patient_mrn}. "
                f"Expected one of: {_DATE_COLUMN_CANDIDATES}"
            )

        patient_df[date_col] = pd.to_datetime(patient_df[date_col], errors="coerce")
        patient_df = patient_df.dropna(subset=[date_col])
        patient_df = patient_df.set_index(date_col).sort_index()
        patient_df.index = pd.DatetimeIndex(patient_df.index)

        # -- Map raw columns → registry feature names -------------------------
        ts_data: dict[str, pd.Series] = {}
        for feature_name, candidates in _EXCEL_COLUMN_MAP.items():
            col = _find_column(patient_df, candidates)
            if col is not None:
                ts_data[feature_name] = pd.to_numeric(patient_df[col], errors="coerce")

        time_series = pd.DataFrame(ts_data, index=patient_df.index)

        # -- Static features from first row -----------------------------------
        first = patient_df.iloc[0]
        static_features: dict = {}
        for candidate in ["age", "sex", "gender", "diagnosis"]:
            col = _find_column(patient_df, [candidate])
            if col is not None:
                static_features[candidate] = first[col]

        # Confounders flagged in CLAUDE.md
        for confounder in ["sleep_apnea", "narcotics_use", "alcohol_use"]:
            col = _find_column(patient_df, [confounder])
            if col is not None:
                static_features[confounder] = bool(first[col])

        # -- Metadata ---------------------------------------------------------
        risk_level = _compute_risk_level(time_series)
        metadata = {
            "data_source_label": "Oura V2 API",
            "data_points_count": len(time_series),
            "risk_level": risk_level,
            "loaded_from": filepath,
        }

        # -- Feature groups from registry -------------------------------------
        feature_groups = {
            group: [fc.name for fc in fcs]
            for group, fcs in get_feature_groups_for_source(DataSource.OURA).items()
        }

        return PatientTimeSeries(
            patient_id=str(patient_mrn),
            data_source=DataSource.OURA,
            static_features=static_features,
            time_series=time_series,
            metadata=metadata,
            feature_groups=feature_groups,
        )

    def load_demo_data(self, patient_id: str) -> PatientTimeSeries:
        """Generate 30 days of realistic synthetic Oura data for demo/testing.

        Uses ``random.seed(hash(patient_id))`` so the same patient always gets
        the same data — matching the seeding strategy already in app.py.

        Realistic ranges (sourced from the existing patient_detail route):
            resting_hr           48–72 bpm
            hrv_balance          20–80 ms
            body_temp_deviation  −1.0 – +1.0 °C
            step_count           2 000–15 000 steps/day
            rem_sleep_pct        15–25 %
            deep_sleep_pct       10–20 %
            sleep_latency        5–30 min
            inactivity_alerts    0–5 alerts/day

        Args:
            patient_id: Arbitrary string identifier for the demo patient.

        Returns:
            A PatientTimeSeries with 30 rows of synthetic daily Oura data.
        """
        rng = random.Random(hash(patient_id))

        end_date = datetime(2024, 12, 30)
        dates = [end_date - timedelta(days=i) for i in range(29, -1, -1)]
        index = pd.DatetimeIndex(dates)

        # -- Generate correlated daily values ---------------------------------
        # HRV and REM are mildly correlated (both reflect sleep quality);
        # add small day-to-day autocorrelation via a smoothed noise term.
        def _smooth_series(low: float, high: float, n: int = 30) -> list[float]:
            """Uniform random walk clipped to [low, high]."""
            mid = (low + high) / 2
            vals = [mid]
            for _ in range(n - 1):
                step = rng.uniform(-(high - low) * 0.15, (high - low) * 0.15)
                vals.append(max(low, min(high, vals[-1] + step)))
            return vals

        hrv_series       = _smooth_series(20, 80)
        rem_series        = _smooth_series(15, 25)
        deep_series       = _smooth_series(10, 20)
        resting_hr_series = _smooth_series(48, 72)
        temp_series       = _smooth_series(-1.0, 1.0)
        steps_series      = [int(rng.uniform(2000, 15000)) for _ in range(30)]
        latency_series    = [round(rng.uniform(5, 30), 1) for _ in range(30)]
        inactivity_series = [rng.randint(0, 5) for _ in range(30)]

        time_series = pd.DataFrame(
            {
                "rem_sleep_pct":       [round(v, 1) for v in rem_series],
                "deep_sleep_pct":      [round(v, 1) for v in deep_series],
                "sleep_latency":       latency_series,
                "hrv_balance":         [round(v, 1) for v in hrv_series],
                "body_temp_deviation": [round(v, 2) for v in temp_series],
                "resting_hr":          [round(v, 0) for v in resting_hr_series],
                "step_count":          steps_series,
                "inactivity_alerts":   inactivity_series,
            },
            index=index,
        )

        # -- Static features (plausible demo patient) -------------------------
        static_features = {
            "age": rng.randint(45, 75),
            "sex": rng.choice(["M", "F"]),
            # Confounders flagged in CLAUDE.md — False for clean demo data
            "sleep_apnea": False,
            "narcotics_use": False,
            "alcohol_use": False,
        }

        # -- Metadata ---------------------------------------------------------
        risk_level = _compute_risk_level(time_series)
        metadata = {
            "data_source_label": "Oura V2 API",
            "data_points_count": len(time_series),
            "risk_level": risk_level,
            "cohort": "demo",
            "enrollment_date": (end_date - timedelta(days=29)).strftime("%Y-%m-%d"),
        }

        # -- Feature groups from registry -------------------------------------
        feature_groups = {
            group: [fc.name for fc in fcs]
            for group, fcs in get_feature_groups_for_source(DataSource.OURA).items()
        }

        return PatientTimeSeries(
            patient_id=patient_id,
            data_source=DataSource.OURA,
            static_features=static_features,
            time_series=time_series,
            metadata=metadata,
            feature_groups=feature_groups,
        )
