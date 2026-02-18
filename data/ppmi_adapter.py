from __future__ import annotations

import os
import random
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd

from data.base import DataSource, PatientTimeSeries
from data.feature_registry import get_feature_groups_for_source


# ---------------------------------------------------------------------------
# PPMI visit schedule: EVENT_ID → month offset from enrollment
# These are the standard PPMI curated data visit codes.
# ---------------------------------------------------------------------------
_VISIT_MONTHS: dict[str, int] = {
    "BL":  0,   # Baseline
    "V04": 6,   # Month 6
    "V06": 12,  # Month 12
    "V08": 24,  # Month 24
    "V10": 36,  # Month 36
}

# Cohort codes in PPMI Demographics / Enrollment CSVs.
# APPRDX column: 1 = Parkinson's Disease (de novo target cohort)
_DE_NOVO_COHORT_CODES = {1, "1", "PD", "de_novo"}

# ---------------------------------------------------------------------------
# Column-name maps: registry feature → candidate PPMI CSV column names
# (checked left-to-right; first match in the loaded DataFrame wins)
# ---------------------------------------------------------------------------

# Motor_Assessments.csv (or MDS_UPDRS_Part_III.csv)
_MOTOR_COLUMN_MAP: dict[str, list[str]] = {
    "baseline_updrs":   ["NP3TOT", "UPDRS_PART3_TOTAL", "MDS_UPDRS_III", "TOTAL"],
    "epworth_sleep":    ["ESS", "ESS_TOTAL", "EPWORTH_TOTAL", "ESSNV"],
    "schwab_england_adl": ["MSEADLG", "SE_ADL", "SCHWAB_ENGLAND", "SEADL"],
}

# Biospecimen_Analysis.csv
_BIOSPECIMEN_COLUMN_MAP: dict[str, list[str]] = {
    "csf_alpha_synuclein": ["ALPHA_SYN", "CSF_ALPHA_SYN", "ALPHASY", "DATSCAN_ALPHA"],
    "amyloid_beta":        ["ABETA", "ABETA42", "AB42", "CSF_ABETA"],
    "total_tau":           ["TTAU", "TOTAL_TAU", "CSF_TAU", "TAU"],
}

# DaTscan / Imaging_Data.csv (PPMI uses mean striatal binding ratio)
_IMAGING_COLUMN_MAP: dict[str, list[str]] = {
    "datscan": ["CAUDATE_R", "DATSCAN_CAUDATE", "SBR_CAUDATE", "MEAN_SBR", "PUTAMEN_R"],
}

# Genetics.csv — time-invariant, extracted to static_features
_GENETICS_COLUMN_MAP: dict[str, list[str]] = {
    "gba_mutation":  ["GBA_MUTATION", "MUTGBA", "GBA", "GBA_STATUS"],
    "lrrk2_mutation": ["LRRK2_MUTATION", "MUTLRRK2", "LRRK2", "LRRK2_STATUS"],
    "apoe_status":   ["APOE", "APOE_STATUS", "APOE_GENOTYPE", "APOE4"],
}

# Demographics.csv
_DEMOGRAPHICS_COLUMN_MAP: dict[str, list[str]] = {
    "age":              ["AGE", "AGE_AT_VISIT", "ENROLL_AGE"],
    "sex":              ["SEX", "GENDER"],
    "enrollment_date":  ["ENROLL_DATE", "ENRDT", "ENROLLMENT_DATE"],
    "cohort_code":      ["APPRDX", "COHORT", "ENROLL_STATUS"],
}

# Candidate column names for the patient ID and visit fields
_PATNO_CANDIDATES   = ["PATNO", "PATIENT_ID", "SUBJECT_ID", "patno"]
_EVENT_ID_CANDIDATES = ["EVENT_ID", "VISIT", "VISIT_CODE", "event_id"]

# Expected CSV filenames inside data_dir (tried in order, first found wins)
_CSV_FILES: dict[str, list[str]] = {
    "motor":       ["Motor_Assessments.csv", "MDS_UPDRS_Part_III.csv",
                    "UPDRS.csv", "motor_assessments.csv"],
    "biospecimen": ["Biospecimen_Analysis.csv", "Biospecimen.csv",
                    "CSF_Biomarkers.csv", "biospecimen.csv"],
    "imaging":     ["DaTscan_Analysis.csv", "Imaging_Data.csv",
                    "SPECT_Analysis.csv", "datscan.csv"],
    "genetics":    ["Genetics.csv", "Genetic_Status.csv",
                    "genetics.csv", "Genetic_Data.csv"],
    "demographics": ["Demographics.csv", "Enrollment.csv",
                     "demographics.csv", "Subject_Characteristics.csv"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column name present in *df* (case-insensitive)."""
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None


def _find_file(data_dir: str, candidates: list[str]) -> str | None:
    """Return the first candidate filename that exists in *data_dir*, or None."""
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.isfile(path):
            return path
    return None


def _load_csv(data_dir: str, key: str) -> pd.DataFrame | None:
    """Load a PPMI CSV by category key; return None if the file is not found."""
    path = _find_file(data_dir, _CSV_FILES[key])
    if path is None:
        return None
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        print(f"[PPMIAdapter] Warning: could not read {path}: {exc}")
        return None


def _filter_to_patient(df: pd.DataFrame, patient_id: str) -> pd.DataFrame:
    """Filter *df* to rows matching *patient_id* in the PATNO column."""
    col = _find_column(df, _PATNO_CANDIDATES)
    if col is None:
        return pd.DataFrame()
    df = df.copy()
    df[col] = df[col].astype(str).str.strip()
    return df[df[col] == str(patient_id).strip()]


def _visit_to_date(enrollment_date: datetime, event_id: str) -> datetime | None:
    """Convert a PPMI EVENT_ID to an absolute date given an enrollment date."""
    months = _VISIT_MONTHS.get(str(event_id).strip().upper())
    if months is None:
        return None
    # Approximate month offset as 30.44 days/month to stay close to real dates
    return enrollment_date + timedelta(days=round(months * 30.44))


def _extract_longitudinal(
    df: pd.DataFrame,
    column_map: dict[str, list[str]],
    enrollment_date: datetime,
) -> pd.DataFrame:
    """Extract visit-level features from a PPMI CSV into a dated DataFrame.

    Returns a DataFrame indexed by visit date with one column per mapped feature.
    Rows whose EVENT_ID is not in ``_VISIT_MONTHS`` are silently dropped.
    """
    event_col = _find_column(df, _EVENT_ID_CANDIDATES)
    if event_col is None:
        return pd.DataFrame()

    rows: list[dict] = []
    for _, row in df.iterrows():
        date = _visit_to_date(enrollment_date, row[event_col])
        if date is None:
            continue
        record: dict = {"date": date}
        for feature_name, candidates in column_map.items():
            col = _find_column(df, candidates)
            if col is not None:
                record[feature_name] = pd.to_numeric(row[col], errors="coerce")
        rows.append(record)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).set_index("date")
    result.index = pd.DatetimeIndex(result.index)
    return result.sort_index()


def _compute_risk_level(progressor_type: str, updrs_slope: float | None = None) -> str:
    """Map progressor type / UPDRS slope to a clinical risk level."""
    if progressor_type == "rapid":
        return "high"
    if progressor_type == "slow":
        return "low"
    # For real data: classify by slope of UPDRS over the observed visits
    if updrs_slope is not None:
        if updrs_slope > 5.0:    # >5 points/year → rapid
            return "high"
        if updrs_slope > 2.0:
            return "medium"
    return "low"


def _build_feature_groups() -> dict[str, list[str]]:
    """Return feature_groups dict (group → column name list) from the registry."""
    return {
        group: [fc.name for fc in fcs]
        for group, fcs in get_feature_groups_for_source(DataSource.PPMI).items()
    }


# ---------------------------------------------------------------------------
# PPMIAdapter
# ---------------------------------------------------------------------------

class PPMIAdapter:
    """Loads PPMI data from curated CSVs or generates synthetic demo data.

    PPMI data structure differs from Oura in two key ways:
      1. Observations are sparse — one row per visit (0, 6, 12, 24, 36 months),
         not one row per day.
      2. Genetic features (GBA, LRRK2, APOE) are time-invariant and live in
         ``static_features``, while biofluid / clinical / imaging measurements
         are longitudinal and live in ``time_series``.
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def load_from_csvs(self, data_dir: str, patient_id: str) -> PatientTimeSeries:
        """Build a PatientTimeSeries from PPMI curated data CSV files.

        Reads up to five CSV files from *data_dir*:
            Motor_Assessments.csv  — UPDRS III, Epworth, Schwab & England
            Biospecimen_Analysis.csv — CSF alpha-syn, amyloid-beta, total tau
            DaTscan_Analysis.csv   — dopamine transporter binding ratio
            Genetics.csv           — GBA / LRRK2 / APOE status (static)
            Demographics.csv       — age, sex, enrollment date, cohort code

        Missing files are silently skipped; the resulting time_series will have
        NaN in any columns whose source file was absent.

        Args:
            data_dir:   Directory containing PPMI curated CSVs.
            patient_id: PPMI PATNO string (e.g. "3000").

        Returns:
            A PatientTimeSeries with a visit-date DatetimeIndex.

        Raises:
            FileNotFoundError: If *data_dir* does not exist.
            ValueError: If *patient_id* is not found in any CSV, or if the
                        patient does not belong to the de novo PD cohort.
        """
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"PPMI data directory not found: {data_dir}")

        # -- Demographics: enrollment date + cohort validation ----------------
        demo_df = _load_csv(data_dir, "demographics")
        enrollment_date = datetime(2020, 1, 1)   # fallback if not found
        static_features: dict = {}

        if demo_df is not None:
            patient_demo = _filter_to_patient(demo_df, patient_id)
            if patient_demo.empty:
                raise ValueError(
                    f"Patient '{patient_id}' not found in Demographics CSV."
                )

            first = patient_demo.iloc[0]

            # Validate de novo cohort membership
            cohort_col = _find_column(patient_demo, _DEMOGRAPHICS_COLUMN_MAP["cohort_code"])
            if cohort_col is not None:
                cohort_val = first[cohort_col]
                if cohort_val not in _DE_NOVO_COHORT_CODES:
                    raise ValueError(
                        f"Patient '{patient_id}' cohort '{cohort_val}' is not "
                        f"the de novo PD cohort. Expected one of: {_DE_NOVO_COHORT_CODES}"
                    )

            # Enrollment date
            enroll_col = _find_column(patient_demo, _DEMOGRAPHICS_COLUMN_MAP["enrollment_date"])
            if enroll_col is not None:
                try:
                    enrollment_date = pd.to_datetime(first[enroll_col]).to_pydatetime()
                except Exception:
                    pass

            # Demographics → static_features
            for key, candidates in _DEMOGRAPHICS_COLUMN_MAP.items():
                if key == "cohort_code":
                    continue
                col = _find_column(patient_demo, candidates)
                if col is not None:
                    static_features[key] = first[col]

        # -- Genetics → static_features (time-invariant) ----------------------
        genetics_df = _load_csv(data_dir, "genetics")
        if genetics_df is not None:
            patient_gen = _filter_to_patient(genetics_df, patient_id)
            if not patient_gen.empty:
                first_gen = patient_gen.iloc[0]
                for feature_name, candidates in _GENETICS_COLUMN_MAP.items():
                    col = _find_column(patient_gen, candidates)
                    if col is not None:
                        static_features[feature_name] = first_gen[col]

        # -- Longitudinal measurements → time_series --------------------------
        frames: list[pd.DataFrame] = []

        motor_df = _load_csv(data_dir, "motor")
        if motor_df is not None:
            patient_motor = _filter_to_patient(motor_df, patient_id)
            if not patient_motor.empty:
                frames.append(
                    _extract_longitudinal(patient_motor, _MOTOR_COLUMN_MAP, enrollment_date)
                )

        bio_df = _load_csv(data_dir, "biospecimen")
        if bio_df is not None:
            patient_bio = _filter_to_patient(bio_df, patient_id)
            if not patient_bio.empty:
                frames.append(
                    _extract_longitudinal(patient_bio, _BIOSPECIMEN_COLUMN_MAP, enrollment_date)
                )

        imaging_df = _load_csv(data_dir, "imaging")
        if imaging_df is not None:
            patient_img = _filter_to_patient(imaging_df, patient_id)
            if not patient_img.empty:
                frames.append(
                    _extract_longitudinal(patient_img, _IMAGING_COLUMN_MAP, enrollment_date)
                )

        # Outer-join all longitudinal frames on the date index
        if frames:
            time_series = frames[0]
            for frame in frames[1:]:
                time_series = time_series.join(frame, how="outer")
            time_series = time_series.sort_index()
        else:
            time_series = pd.DataFrame()

        # -- Risk level from UPDRS slope --------------------------------------
        updrs_slope: float | None = None
        if not time_series.empty and "baseline_updrs" in time_series.columns:
            scores = time_series["baseline_updrs"].dropna()
            if len(scores) >= 2:
                # Slope in points/year using first and last valid visits
                date_diff_years = (scores.index[-1] - scores.index[0]).days / 365.25
                if date_diff_years > 0:
                    updrs_slope = (scores.iloc[-1] - scores.iloc[0]) / date_diff_years

        progressor_type = (
            "rapid" if (updrs_slope is not None and updrs_slope > 5.0) else "slow"
        )

        metadata = {
            "data_source_label": "PPMI (LONI IDA)",
            "cohort": "de_novo",
            "risk_level": _compute_risk_level(progressor_type, updrs_slope),
            "prediction_horizon": "48 months",
            "data_points_count": len(time_series),
            "enrollment_date": enrollment_date.strftime("%Y-%m-%d"),
            "progressor_type": progressor_type,
        }

        return PatientTimeSeries(
            patient_id=str(patient_id),
            data_source=DataSource.PPMI,
            static_features=static_features,
            time_series=time_series,
            metadata=metadata,
            feature_groups=_build_feature_groups(),
        )

    def load_demo_data(
        self,
        patient_id: str,
        progressor_type: Literal["rapid", "slow"] = "rapid",
    ) -> PatientTimeSeries:
        """Generate synthetic PPMI data for demo / unit-testing purposes.

        Produces five visits (months 0, 6, 12, 24, 36) with clinically
        plausible trajectories for each progressor type.

        Rapid progressor trajectory:
            UPDRS III:     ~15 → ~38–42 (accelerating, GBA mutation = 1)
            CSF alpha-syn: 800–1200 pg/mL (low — correlates with fast decline)
            DaTscan SBR:   1.2 → 0.9 (declining dopamine transporter binding)
            Schwab & England: 90% → 72% (functional decline)

        Slow progressor trajectory:
            UPDRS III:     ~12 → ~19–22 (mild, LRRK2 mutation = 0)
            CSF alpha-syn: 1500–2000 pg/mL (high — correlates with slow decline)
            DaTscan SBR:   2.0 → 1.6 (stable-ish binding)
            Schwab & England: 95% → 88%

        Args:
            patient_id:      Arbitrary identifier for this demo patient.
            progressor_type: "rapid" or "slow" — controls trajectory shape
                             and default genetic profile.

        Returns:
            A PatientTimeSeries with five visit rows indexed by absolute date.
        """
        rng = random.Random(hash((patient_id, progressor_type)))

        enrollment_date = datetime(2020, 1, 1)
        visit_months = sorted(_VISIT_MONTHS.values())   # [0, 6, 12, 24, 36]
        visit_dates = [
            enrollment_date + timedelta(days=round(m * 30.44))
            for m in visit_months
        ]
        index = pd.DatetimeIndex(visit_dates)
        n = len(visit_months)

        def _jitter(value: float, pct: float = 0.08) -> float:
            """Add ±pct relative noise to a value."""
            return value * (1 + rng.uniform(-pct, pct))

        # ── Per-visit trajectories ────────────────────────────────────────────
        if progressor_type == "rapid":
            # UPDRS III: starts ~15, accelerates toward 38–42 by month 36
            updrs_targets = [15, 19, 24, 31, rng.uniform(38, 42)]
            csf_alpha_syn_base = rng.uniform(800, 1200)
            amyloid_base       = rng.uniform(300, 550)
            tau_base           = rng.uniform(200, 400)
            datscan_start      = rng.uniform(1.1, 1.4)
            datscan_end        = rng.uniform(0.8, 1.0)
            schwab_start       = rng.uniform(87, 93)
            schwab_end         = rng.uniform(68, 76)
            epworth_base       = rng.uniform(10, 16)
        else:  # slow
            updrs_targets = [12, 13, 15, 17, rng.uniform(19, 22)]
            csf_alpha_syn_base = rng.uniform(1500, 2000)
            amyloid_base       = rng.uniform(600, 1000)
            tau_base           = rng.uniform(100, 220)
            datscan_start      = rng.uniform(1.8, 2.2)
            datscan_end        = rng.uniform(1.5, 1.8)
            schwab_start       = rng.uniform(93, 98)
            schwab_end         = rng.uniform(86, 91)
            epworth_base       = rng.uniform(4, 9)

        def _interpolate(start: float, end: float) -> list[float]:
            """Linear interpolation across *n* visits with small noise."""
            return [
                round(_jitter(start + (end - start) * i / (n - 1)), 2)
                for i in range(n)
            ]

        time_series = pd.DataFrame(
            {
                # Clinical (primary target + functional scales)
                "baseline_updrs":     [round(_jitter(v), 1) for v in updrs_targets],
                "epworth_sleep":      [round(_jitter(epworth_base, 0.15), 1) for _ in range(n)],
                "schwab_england_adl": _interpolate(schwab_start, schwab_end),

                # Biofluid (mostly stable — re-measured at select visits, NaN otherwise)
                # Biospecimen typically collected at BL, V06, V10 in PPMI
                "csf_alpha_synuclein": [
                    round(_jitter(csf_alpha_syn_base, 0.05), 1) if i in (0, 2, 4) else float("nan")
                    for i in range(n)
                ],
                "amyloid_beta": [
                    round(_jitter(amyloid_base, 0.05), 1) if i in (0, 2, 4) else float("nan")
                    for i in range(n)
                ],
                "total_tau": [
                    round(_jitter(tau_base, 0.05), 1) if i in (0, 2, 4) else float("nan")
                    for i in range(n)
                ],

                # Imaging — DaTscan measured at BL, V06, V10
                "datscan": [
                    round(_jitter(datscan_start + (datscan_end - datscan_start) * i / (n - 1), 0.04), 3)
                    if i in (0, 2, 4) else float("nan")
                    for i in range(n)
                ],
            },
            index=index,
        )

        # ── Static features (genetic + demographics) ──────────────────────────
        # Rapid progressors are more likely to carry GBA mutation (per literature)
        if progressor_type == "rapid":
            gba_status  = rng.choices([0, 1, 2], weights=[0.55, 0.40, 0.05])[0]
            lrrk2_status = 0
            apoe_alleles = rng.choices(["e3/e3", "e3/e4", "e4/e4"], weights=[0.50, 0.38, 0.12])[0]
        else:
            gba_status  = rng.choices([0, 1], weights=[0.85, 0.15])[0]
            lrrk2_status = rng.choices([0, 1], weights=[0.80, 0.20])[0]
            apoe_alleles = rng.choices(["e2/e3", "e3/e3", "e3/e4"], weights=[0.15, 0.65, 0.20])[0]

        static_features: dict = {
            "gba_mutation":  gba_status,
            "lrrk2_mutation": lrrk2_status,
            "apoe_status":   apoe_alleles,
            "age":           rng.randint(50, 78),
            "sex":           rng.choice(["M", "F"]),
        }

        metadata = {
            "data_source_label": "PPMI (LONI IDA)",
            "cohort": "de_novo",
            "risk_level": _compute_risk_level(progressor_type),
            "prediction_horizon": "48 months",
            "data_points_count": len(time_series),
            "enrollment_date": enrollment_date.strftime("%Y-%m-%d"),
            "progressor_type": progressor_type,
        }

        return PatientTimeSeries(
            patient_id=patient_id,
            data_source=DataSource.PPMI,
            static_features=static_features,
            time_series=time_series,
            metadata=metadata,
            feature_groups=_build_feature_groups(),
        )
