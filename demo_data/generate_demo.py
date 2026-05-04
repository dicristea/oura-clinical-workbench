#!/usr/bin/env python3
"""Generate demo data files for development and testing.

Run from the project root:
    python demo_data/generate_demo.py

Outputs
-------
demo_data/demo_synthea/
    One FHIR R4 Bundle JSON per Synthea demo patient.

demo_data/demo_omh_ieee/
    OMH/IEEE wearable records generated from the Synthea FHIR patient
    profiles above.

No real patient data is used or required.
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import date
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — allow running from the project root or the demo_data/ dir
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.synthea_adapter import SyntheaAdapter   # noqa: E402
from demo_data.omh_ieee_generator.pipeline import generate_dataset  # noqa: E402

# ---------------------------------------------------------------------------
# Output locations
# ---------------------------------------------------------------------------
DEMO_DATA_DIR    = PROJECT_ROOT / "demo_data"
DEMO_SYNTHEA_DIR = DEMO_DATA_DIR / "demo_synthea"
DEMO_OMH_IEEE_DIR = DEMO_DATA_DIR / "demo_omh_ieee"
DEMO_REFERENCE_DATE = date(2026, 4, 19)

# ---------------------------------------------------------------------------
# Patient rosters
# ---------------------------------------------------------------------------
# (patient_id, risk_level)  — 2 high / 4 medium / 2 low
SYNTHEA_PATIENTS: list[tuple[str, str]] = [
    ("PT-3001", "high"),
    ("PT-3002", "high"),
    ("PT-3003", "medium"),
    ("PT-3004", "medium"),
    ("PT-3005", "low"),
    ("PT-3006", "medium"),
    ("PT-3007", "low"),
    ("PT-3008", "medium"),
]

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


def generate_synthea_fhir(seed: int) -> None:
    """Write one FHIR R4 Bundle JSON per Synthea demo patient."""
    import json as _json
    import uuid

    adapter = SyntheaAdapter()
    DEMO_SYNTHEA_DIR.mkdir(parents=True, exist_ok=True)
    for stale_bundle in DEMO_SYNTHEA_DIR.glob("*.json"):
        stale_bundle.unlink()

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
        pts = adapter.load_demo_data(patient_id, risk_level, seed=seed)
        ts  = pts.time_series
        sf  = pts.static_features

        fhir_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, patient_id))
        birth_year = DEMO_REFERENCE_DATE.year - sf.get("age", 50)
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
# OMH/IEEE wearable JSON generator
# ---------------------------------------------------------------------------

def generate_omh_ieee_wearable_records(seed: int) -> None:
    """Write OMH/IEEE Oura-like wearable records from demo Synthea FHIR."""
    print(f"\n{'─' * 60}")
    print(f"  OMH/IEEE wearable records  →  {DEMO_OMH_IEEE_DIR.relative_to(PROJECT_ROOT)}/")
    print(f"{'─' * 60}")

    manifest = generate_dataset(
        DEMO_SYNTHEA_DIR,
        DEMO_OMH_IEEE_DIR,
        days=30,
        end_date=DEMO_REFERENCE_DATE,
        seed=seed,
        source_name="demo-synthea-omh-ieee-generator",
    )

    print(f"  {manifest['patient_count']} patients")
    print(f"  {sum(manifest['counts_by_schema'].values())} total records")
    print("  Schemas:")
    for schema_id, count in manifest["counts_by_schema"].items():
        print(f"    {schema_id:<34} {count:>4}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate dashboard demo Synthea FHIR and OMH/IEEE wearable data."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional integer seed. Omit it to generate a new random demo dataset.",
    )
    return parser.parse_args()


def _resolve_seed(seed: int | None) -> int:
    if seed is not None:
        return seed
    return random.SystemRandom().randint(1, 2_147_483_647)


def main() -> None:
    args = _parse_args()
    seed = _resolve_seed(args.seed)

    print("=" * 60)
    print("  oura-clinical-workbench  —  demo data generator")
    print("=" * 60)
    print(f"  Seed: {seed}" + (" (provided)" if args.seed is not None else " (random)"))

    generate_synthea_fhir(seed)
    generate_omh_ieee_wearable_records(seed)

    print(f"\n{'─' * 60}")
    print("  All files written. Load them with:")
    print("    SyntheaAdapter().load_from_fhir('demo_data/demo_synthea/PT-3001.json', 'PT-3001')")
    print("    StandardWearableAdapter().load_all_from_dir('demo_data/demo_omh_ieee')")
    print(f"{'─' * 60}\n")


if __name__ == "__main__":
    main()
