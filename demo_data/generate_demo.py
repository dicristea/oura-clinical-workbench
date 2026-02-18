#!/usr/bin/env python3
"""Generate demo data files for development and testing.

Run from the project root:
    python demo_data/generate_demo.py

Outputs
-------
demo_data/demo_oura.xlsx
    10 patients × 30 days of Oura data in the format that
    OuraAdapter.load_from_excel() can re-read.

demo_data/demo_ppmi/
    Motor_Assessments.csv, Biospecimen_Analysis.csv, DaTscan_Analysis.csv,
    Genetics.csv, Demographics.csv — one row per patient per visit, in the
    column-name format that PPMIAdapter.load_from_csvs() can re-read.

No real patient data is used or required.
"""

from __future__ import annotations

import random
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — allow running from the project root or the demo_data/ dir
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.oura_adapter import OuraAdapter          # noqa: E402 (after sys.path)
from data.ppmi_adapter import PPMIAdapter          # noqa: E402

# ---------------------------------------------------------------------------
# Output locations
# ---------------------------------------------------------------------------
DEMO_DATA_DIR  = PROJECT_ROOT / "demo_data"
DEMO_PPMI_DIR  = DEMO_DATA_DIR / "demo_ppmi"
DEMO_OURA_FILE = DEMO_DATA_DIR / "demo_oura.xlsx"

# ---------------------------------------------------------------------------
# Patient rosters
# ---------------------------------------------------------------------------
# (patient_id, desired_risk_level)
# Distribution: 3 high / 4 medium / 3 low
OURA_PATIENTS: list[tuple[str, str]] = [
    ("PT-1001", "high"),
    ("PT-1002", "high"),
    ("PT-1003", "high"),
    ("PT-1004", "medium"),
    ("PT-1005", "medium"),
    ("PT-1006", "medium"),
    ("PT-1007", "medium"),
    ("PT-1008", "low"),
    ("PT-1009", "low"),
    ("PT-1010", "low"),
]

# (patient_id, progressor_type)
# Distribution: 4 rapid / 4 slow
PPMI_PATIENTS: list[tuple[str, str]] = [
    ("PT-2001", "rapid"),
    ("PT-2002", "rapid"),
    ("PT-2003", "rapid"),
    ("PT-2004", "rapid"),
    ("PT-2005", "slow"),
    ("PT-2006", "slow"),
    ("PT-2007", "slow"),
    ("PT-2008", "slow"),
]

# ---------------------------------------------------------------------------
# PPMI visit schedule (mirrors ppmi_adapter._VISIT_MONTHS — defined locally
# so we don't import a private symbol from the adapter module)
# ---------------------------------------------------------------------------
_VISIT_MONTHS: dict[str, int] = {
    "BL":  0,
    "V04": 6,
    "V06": 12,
    "V08": 24,
    "V10": 36,
}
_MONTHS_TO_EVENT: dict[int, str] = {v: k for k, v in _VISIT_MONTHS.items()}

# Must match the enrollment_date hardcoded in PPMIAdapter.load_demo_data
_PPMI_ENROLLMENT_DATE = datetime(2020, 1, 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scale_to_risk_level(
    ts: pd.DataFrame, target: str, rng: random.Random
) -> pd.DataFrame:
    """Rescale HRV and REM columns so the time_series yields the desired risk.

    Thresholds mirror OuraAdapter._compute_risk_level:
        high   – mean HRV < 30  OR  mean REM % < 16
        medium – mean HRV 30-45 OR  mean REM % 16-20
        low    – mean HRV > 45  AND mean REM % > 20
    """
    HRV_TARGET_RANGES: dict[str, tuple[float, float]] = {
        "high":   (22.0, 28.0),   # forces mean < 30
        "medium": (34.0, 42.0),   # inside 30-45
        "low":    (50.0, 65.0),   # forces mean > 45
    }
    REM_TARGET_RANGES: dict[str, tuple[float, float]] = {
        "high":   (11.0, 15.0),   # forces mean < 16
        "medium": (17.0, 20.0),   # inside 16-20
        "low":    (21.0, 24.0),   # forces mean > 20
    }

    ts = ts.copy()

    if "hrv_balance" in ts.columns:
        cur = ts["hrv_balance"].mean()
        if cur > 0:
            lo, hi = HRV_TARGET_RANGES[target]
            scale = rng.uniform(lo, hi) / cur
            ts["hrv_balance"] = (
                ts["hrv_balance"].mul(scale).clip(lower=5.0, upper=100.0).round(1)
            )

    if "rem_sleep_pct" in ts.columns:
        cur = ts["rem_sleep_pct"].mean()
        if cur > 0:
            lo, hi = REM_TARGET_RANGES[target]
            scale = rng.uniform(lo, hi) / cur
            ts["rem_sleep_pct"] = (
                ts["rem_sleep_pct"].mul(scale).clip(lower=5.0, upper=35.0).round(1)
            )

    return ts


def _verify_risk(ts: pd.DataFrame) -> str:
    """Recompute risk level from a scaled time_series (local replica of adapter logic)."""
    hrv_mean = ts["hrv_balance"].mean() if "hrv_balance" in ts.columns else float("nan")
    rem_mean = ts["rem_sleep_pct"].mean() if "rem_sleep_pct" in ts.columns else float("nan")
    if hrv_mean < 30 or rem_mean < 16:
        return "high"
    if hrv_mean <= 45 or rem_mean <= 20:
        return "medium"
    return "low"


def _date_to_event_id(dt: datetime) -> str:
    """Map a visit datetime → PPMI EVENT_ID by finding the nearest known visit month."""
    approx_months = (dt - _PPMI_ENROLLMENT_DATE).days / 30.44
    closest_month = min(_MONTHS_TO_EVENT, key=lambda m: abs(m - approx_months))
    return _MONTHS_TO_EVENT[closest_month]


# ---------------------------------------------------------------------------
# Oura Excel generator
# ---------------------------------------------------------------------------

def generate_oura_excel() -> None:
    """Write demo_oura.xlsx — 10 patients × 30 daily rows, risk-adjusted."""
    adapter = OuraAdapter()
    blocks: list[pd.DataFrame] = []

    print(f"\n{'─' * 48}")
    print(f"  Oura patients  →  {DEMO_OURA_FILE.relative_to(PROJECT_ROOT)}")
    print(f"{'─' * 48}")
    print(f"  {'Patient ID':<12} {'Target':<9} {'Actual':<9}  mean HRV  mean REM%")

    for patient_id, target_risk in OURA_PATIENTS:
        pts = adapter.load_demo_data(patient_id)
        # Use a separate, deterministic RNG for the scaling step so the
        # patient's own data stays reproducible.
        scale_rng = random.Random(hash((patient_id, "scale")))
        scaled_ts = _scale_to_risk_level(pts.time_series, target_risk, scale_rng)
        actual_risk = _verify_risk(scaled_ts)

        hrv_mean = scaled_ts["hrv_balance"].mean()
        rem_mean = scaled_ts["rem_sleep_pct"].mean()
        status = "✓" if actual_risk == target_risk else "!"
        print(
            f"  {patient_id:<12} {target_risk:<9} {actual_risk:<9}  "
            f"{hrv_mean:>6.1f} ms  {rem_mean:>5.1f}%  {status}"
        )

        # Flatten to one row-per-day; use the index name trick so reset_index
        # produces a column with the exact name the adapter expects.
        scaled_ts.index.name = "flowsheet_record_date"
        block = scaled_ts.reset_index()
        block.insert(0, "mrn", patient_id)
        # Include static demographics so load_from_excel can populate them
        block["age"] = pts.static_features.get("age", "")
        block["sex"] = pts.static_features.get("sex", "")
        blocks.append(block)

    combined = pd.concat(blocks, ignore_index=True)
    DEMO_OURA_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_excel(DEMO_OURA_FILE, index=False)

    print(f"\n  {len(combined)} rows written  ({len(OURA_PATIENTS)} patients × 30 days)")


# ---------------------------------------------------------------------------
# PPMI CSV generator
# ---------------------------------------------------------------------------

def generate_ppmi_csvs() -> None:
    """Write five PPMI CSVs to demo_data/demo_ppmi/ for 8 demo patients."""
    adapter = PPMIAdapter()
    DEMO_PPMI_DIR.mkdir(parents=True, exist_ok=True)

    motor_rows:       list[dict] = []
    bio_rows:         list[dict] = []
    imaging_rows:     list[dict] = []
    genetics_rows:    list[dict] = []
    demographics_rows: list[dict] = []

    print(f"\n{'─' * 70}")
    print(f"  PPMI patients  →  {DEMO_PPMI_DIR.relative_to(PROJECT_ROOT)}/")
    print(f"{'─' * 70}")
    print(f"  {'Patient ID':<12} {'Type':<9} {'Risk':<7}  UPDRS trajectory (months 0→6→12→24→36)")

    for patient_id, progressor_type in PPMI_PATIENTS:
        pts = adapter.load_demo_data(patient_id, progressor_type)
        ts  = pts.time_series
        sf  = pts.static_features

        # UPDRS trajectory summary
        updrs = ts["baseline_updrs"].tolist()
        traj  = " → ".join(f"{v:.1f}" for v in updrs)
        print(
            f"  {patient_id:<12} {progressor_type:<9} "
            f"{pts.metadata['risk_level']:<7}  {traj}"
        )

        for visit_date, row in ts.iterrows():
            event_id = _date_to_event_id(visit_date)

            # Motor / clinical assessments — measured at every visit
            motor_rows.append({
                "PATNO":    patient_id,
                "EVENT_ID": event_id,
                "NP3TOT":   row.get("baseline_updrs"),
                "ESS":      row.get("epworth_sleep"),
                "MSEADLG":  row.get("schwab_england_adl"),
            })

            # Biospecimen — NaN at non-collection visits (V04, V08)
            bio_rows.append({
                "PATNO":     patient_id,
                "EVENT_ID":  event_id,
                "ALPHA_SYN": row.get("csf_alpha_synuclein"),
                "ABETA":     row.get("amyloid_beta"),
                "TTAU":      row.get("total_tau"),
            })

            # DaTscan imaging — NaN at non-collection visits (V04, V08)
            imaging_rows.append({
                "PATNO":     patient_id,
                "EVENT_ID":  event_id,
                "CAUDATE_R": row.get("datscan"),
            })

        # Genetics — one row per patient, no EVENT_ID (time-invariant)
        genetics_rows.append({
            "PATNO":          patient_id,
            "GBA_MUTATION":   sf.get("gba_mutation"),
            "LRRK2_MUTATION": sf.get("lrrk2_mutation"),
            "APOE":           sf.get("apoe_status"),
        })

        # Demographics — one row per patient
        demographics_rows.append({
            "PATNO":       patient_id,
            "AGE":         sf.get("age"),
            "SEX":         sf.get("sex"),
            "ENROLL_DATE": pts.metadata.get("enrollment_date"),
            "APPRDX":      1,     # de novo PD cohort code (1 = PD in PPMI)
        })

    # Write all CSVs
    csv_manifest: dict[str, pd.DataFrame] = {
        "Motor_Assessments.csv":    pd.DataFrame(motor_rows),
        "Biospecimen_Analysis.csv": pd.DataFrame(bio_rows),
        "DaTscan_Analysis.csv":     pd.DataFrame(imaging_rows),
        "Genetics.csv":             pd.DataFrame(genetics_rows),
        "Demographics.csv":         pd.DataFrame(demographics_rows),
    }

    print(f"\n  {'File':<35} {'Rows':>6}  {'Cols':>5}")
    for filename, df in csv_manifest.items():
        path = DEMO_PPMI_DIR / filename
        df.to_csv(path, index=False)
        print(f"  {filename:<35} {len(df):>6}  {len(df.columns):>5}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  oura-clinical-workbench  —  demo data generator")
    print("=" * 60)

    generate_oura_excel()
    generate_ppmi_csvs()

    print(f"\n{'─' * 60}")
    print("  All files written. Load them with:")
    print("    OuraAdapter().load_from_excel('demo_data/demo_oura.xlsx', 'PT-1001')")
    print("    PPMIAdapter().load_from_csvs('demo_data/demo_ppmi', 'PT-2001')")
    print(f"{'─' * 60}\n")


if __name__ == "__main__":
    main()
