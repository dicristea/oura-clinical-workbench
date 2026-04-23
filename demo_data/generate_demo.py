#!/usr/bin/env python3
"""Generate demo data files for development and testing.

Run from the project root:
    python demo_data/generate_demo.py

Outputs
-------
demo_data/demo_oura.xlsx
    10 patients × 30 days of Oura data in the format that
    OuraAdapter.load_from_excel() can re-read.

demo_data/demo_synthea/
    One FHIR R4 Bundle JSON per Synthea demo patient.

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
from data.synthea_adapter import SyntheaAdapter   # noqa: E402

# ---------------------------------------------------------------------------
# Output locations
# ---------------------------------------------------------------------------
DEMO_DATA_DIR    = PROJECT_ROOT / "demo_data"
DEMO_SYNTHEA_DIR = DEMO_DATA_DIR / "demo_synthea"
DEMO_OURA_FILE   = DEMO_DATA_DIR / "demo_oura.xlsx"

# ---------------------------------------------------------------------------
# Patient rosters
# ---------------------------------------------------------------------------
# (patient_id, risk_level)  — 2 high / 2 medium / 1 low
SYNTHEA_PATIENTS: list[tuple[str, str]] = [
    ("PT-3001", "high"),
    ("PT-3002", "high"),
    ("PT-3003", "medium"),
    ("PT-3004", "medium"),
    ("PT-3005", "low"),
]

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
# Synthea FHIR JSON generator
# ---------------------------------------------------------------------------

def _make_loinc_coding(loinc_code: str, display: str) -> dict:
    return {"coding": [{"system": "http://loinc.org", "code": loinc_code, "display": display}],
            "text": display}


def _make_observation(patient_ref: str, loinc_code: str, display: str,
                      value: float, unit: str, unit_code: str, date: str) -> dict:
    return {
        "resourceType": "Observation",
        "status": "final",
        "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                   "code": "vital-signs", "display": "Vital Signs"}]}],
        "code": _make_loinc_coding(loinc_code, display),
        "subject": {"reference": f"urn:uuid:{patient_ref}"},
        "effectiveDateTime": date,
        "valueQuantity": {"value": value, "unit": unit, "system": "http://unitsofmeasure.org", "code": unit_code},
    }


def _make_bp_observation(patient_ref: str, sbp: float, dbp: float, date: str) -> dict:
    """Blood pressure is a two-component Observation in FHIR."""
    return {
        "resourceType": "Observation",
        "status": "final",
        "code": _make_loinc_coding("55284-4", "Blood pressure systolic and diastolic"),
        "subject": {"reference": f"urn:uuid:{patient_ref}"},
        "effectiveDateTime": date,
        "component": [
            {
                "code": _make_loinc_coding("8480-6", "Systolic blood pressure"),
                "valueQuantity": {"value": sbp, "unit": "mmHg",
                                  "system": "http://unitsofmeasure.org", "code": "mm[Hg]"},
            },
            {
                "code": _make_loinc_coding("8462-4", "Diastolic blood pressure"),
                "valueQuantity": {"value": dbp, "unit": "mmHg",
                                  "system": "http://unitsofmeasure.org", "code": "mm[Hg]"},
            },
        ],
    }


def _make_condition(patient_ref: str, snomed_code: str, display: str) -> dict:
    return {
        "resourceType": "Condition",
        "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                                        "code": "active"}]},
        "code": {"coding": [{"system": "http://snomed.info/sct",
                              "code": snomed_code, "display": display}],
                 "text": display},
        "subject": {"reference": f"urn:uuid:{patient_ref}"},
    }


def generate_synthea_fhir() -> None:
    """Write one FHIR R4 Bundle JSON per Synthea demo patient."""
    import json as _json
    import uuid

    adapter = SyntheaAdapter()
    DEMO_SYNTHEA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─' * 60}")
    print(f"  Synthea patients  →  {DEMO_SYNTHEA_DIR.relative_to(PROJECT_ROOT)}/")
    print(f"{'─' * 60}")
    print(f"  {'Patient ID':<12} {'Risk':<9} {'Encounters':>11}  Conditions")

    # SNOMED codes for conditions added based on risk
    _CONDITIONS_BY_RISK = {
        "high":   [("44054006", "Diabetes mellitus type 2"), ("38341003", "Hypertension")],
        "medium": [("38341003", "Hypertension")],
        "low":    [],
    }

    for patient_id, risk_level in SYNTHEA_PATIENTS:
        pts = adapter.load_demo_data(patient_id, risk_level)
        ts  = pts.time_series
        sf  = pts.static_features

        fhir_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, patient_id))
        birth_year = datetime.now().year - sf.get("age", 50)
        gender = "male" if sf.get("sex") == "M" else "female"

        entries: list[dict] = []

        # Patient resource
        entries.append({"fullUrl": f"urn:uuid:{fhir_id}", "resource": {
            "resourceType": "Patient",
            "id": fhir_id,
            "name": [{"use": "official", "family": f"Demo-{patient_id}",
                      "given": [risk_level.capitalize()]}],
            "gender": gender,
            "birthDate": f"{birth_year}-06-15",
        }})

        # Observations — one encounter row per date
        for obs_date, row in ts.iterrows():
            date_str = pd.Timestamp(obs_date).strftime("%Y-%m-%dT00:00:00+00:00")

            entries.append({"resource": _make_bp_observation(
                fhir_id, row["systolic_bp"], row["diastolic_bp"], date_str)})

            obs_map = [
                ("8867-4",  "Heart rate",          "heart_rate",             "bpm",    "/min"),
                ("9279-1",  "Respiratory rate",    "respiratory_rate",       "br/min", "/min"),
                ("8310-5",  "Body temperature",    "body_temperature",       "°C",     "Cel"),
                ("29463-7", "Body weight",         "body_weight_kg",         "kg",     "kg"),
                ("39156-5", "Body mass index",     "bmi",                    "kg/m2",  "kg/m2"),
                ("2339-0",  "Glucose",             "glucose_mgdl",           "mg/dL",  "mg/dL"),
                ("4548-4",  "Hemoglobin A1c",      "hba1c_pct",             "%",      "%"),
                ("2093-3",  "Total cholesterol",   "total_cholesterol_mgdl", "mg/dL",  "mg/dL"),
                ("18262-6", "LDL cholesterol",     "ldl_cholesterol_mgdl",   "mg/dL",  "mg/dL"),
            ]
            for loinc, display, feat, unit, unit_code in obs_map:
                if feat in row and not (isinstance(row[feat], float) and pd.isna(row[feat])):
                    entries.append({"resource": _make_observation(
                        fhir_id, loinc, display, round(float(row[feat]), 2),
                        unit, unit_code, date_str)})

        # Condition resources
        conditions = _CONDITIONS_BY_RISK.get(risk_level, [])
        for snomed, display in conditions:
            entries.append({"resource": _make_condition(fhir_id, snomed, display)})

        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": entries,
        }

        out_path = DEMO_SYNTHEA_DIR / f"{patient_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            _json.dump(bundle, f, indent=2)

        condition_names = ", ".join(d for _, d in conditions) or "None"
        print(f"  {patient_id:<12} {risk_level:<9} {len(ts):>11}  {condition_names}")

    print(f"\n  {len(SYNTHEA_PATIENTS)} FHIR bundles written")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  oura-clinical-workbench  —  demo data generator")
    print("=" * 60)

    generate_oura_excel()
    generate_synthea_fhir()

    print(f"\n{'─' * 60}")
    print("  All files written. Load them with:")
    print("    OuraAdapter().load_from_excel('demo_data/demo_oura.xlsx', 'PT-1001')")
    print("    SyntheaAdapter().load_from_fhir('demo_data/demo_synthea/PT-3001.json', 'PT-3001')")
    print(f"{'─' * 60}\n")


if __name__ == "__main__":
    main()
