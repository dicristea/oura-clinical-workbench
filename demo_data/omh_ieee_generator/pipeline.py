from __future__ import annotations

import json
import shutil
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from demo_data.omh_ieee_generator.fhir import SyntheaPatientProfile, load_synthea_profiles
from demo_data.omh_ieee_generator.records import build_patient_records
from demo_data.omh_ieee_generator.simulate import generate_patient_days
from demo_data.omh_ieee_generator.validate import validate_records


def generate_dataset(
    synthea_input: str | Path,
    output_dir: str | Path,
    *,
    days: int = 30,
    end_date: date | None = None,
    seed: int = 42,
    patient_limit: int | None = None,
    source_name: str = "synthea-python-generator",
) -> dict[str, Any]:
    end_date = end_date or date.today()
    profiles = load_synthea_profiles(synthea_input, reference_date=end_date)
    if patient_limit is not None:
        profiles = profiles[:patient_limit]

    output_dir = Path(output_dir)
    patients_dir = output_dir / "patients"
    if patients_dir.exists():
        shutil.rmtree(patients_dir)
    patients_dir.mkdir(parents=True, exist_ok=True)

    all_schema_counts: dict[str, int] = defaultdict(int)
    patient_entries: list[dict[str, Any]] = []

    for profile in profiles:
        patient_days = generate_patient_days(profile, days=days, end_date=end_date, seed=seed)
        records_by_schema = build_patient_records(profile, patient_days, source_name=source_name)
        validation_warnings = validate_records(records_by_schema)
        counts_by_schema = {
            schema_id: len(records)
            for schema_id, records in sorted(records_by_schema.items())
        }
        for schema_id, count in counts_by_schema.items():
            all_schema_counts[schema_id] += count

        patient_payload = {
            "patient": _profile_to_json(profile),
            "generation_window": {
                "days": days,
                "end_date": end_date.isoformat(),
            },
            "records_by_schema": records_by_schema,
            "counts_by_schema": counts_by_schema,
            "validation_warnings": validation_warnings,
        }

        patient_dir = patients_dir / profile.patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)
        records_path = patient_dir / "records.json"
        records_path.write_text(
            json.dumps(patient_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        patient_entries.append(
            {
                "patient_id": profile.patient_id,
                "display_name": profile.display_name,
                "gender": profile.gender,
                "age": profile.age,
                "conditions": list(profile.conditions),
                "record_count": sum(counts_by_schema.values()),
                "warning_count": len(validation_warnings),
                "output_file": str(records_path.relative_to(output_dir)),
            }
        )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "generator": {
            "source_name": source_name,
            "synthea_input": str(Path(synthea_input)),
            "days": days,
            "end_date": end_date.isoformat(),
            "seed": seed,
            "patient_limit": patient_limit,
        },
        "patient_count": len(patient_entries),
        "counts_by_schema": dict(sorted(all_schema_counts.items())),
        "patients": patient_entries,
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def _profile_to_json(profile: SyntheaPatientProfile) -> dict[str, Any]:
    return {
        "patient_id": profile.patient_id,
        "source_fhir_patient_id": profile.source_fhir_patient_id,
        "given_name": profile.given_name,
        "family_name": profile.family_name,
        "display_name": profile.display_name,
        "gender": profile.gender,
        "birth_date": profile.birth_date.isoformat() if profile.birth_date else None,
        "age": profile.age,
        "conditions": list(profile.conditions),
        "source_path": str(profile.source_path),
    }
