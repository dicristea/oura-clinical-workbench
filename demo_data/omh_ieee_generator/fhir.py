from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SyntheaPatientProfile:
    patient_id: str
    source_fhir_patient_id: str
    given_name: str
    family_name: str
    gender: str
    birth_date: date | None
    age: int | None
    conditions: tuple[str, ...]
    source_path: Path

    @property
    def display_name(self) -> str:
        name = f"{self.given_name} {self.family_name}".strip()
        return name or self.patient_id


def load_synthea_profiles(
    input_path: str | Path,
    *,
    reference_date: date,
) -> list[SyntheaPatientProfile]:
    """Load Synthea Patient profiles from per-patient FHIR Bundle JSON files."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Synthea input path not found: {input_path}")

    if input_path.is_file():
        candidates = [input_path]
    else:
        candidates = sorted(path for path in input_path.rglob("*.json") if path.is_file())

    profiles: list[SyntheaPatientProfile] = []
    for path in candidates:
        payload = _read_json(path)
        profile = _extract_profile(payload, source_path=path, reference_date=reference_date)
        if profile is not None:
            profiles.append(profile)

    if not profiles:
        raise ValueError(
            f"No usable Synthea FHIR patient bundles were found under {input_path}."
        )

    return profiles


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_profile(
    payload: Any,
    *,
    source_path: Path,
    reference_date: date,
) -> SyntheaPatientProfile | None:
    if not isinstance(payload, dict):
        return None

    resources = list(_iter_resources(payload))
    patient_resource = next(
        (resource for resource in resources if resource.get("resourceType") == "Patient"),
        None,
    )
    if patient_resource is None:
        return None

    source_fhir_patient_id = str(patient_resource.get("id") or "")
    patient_id = source_path.stem
    given_name, family_name = _extract_name(patient_resource)
    birth_date = _parse_birth_date(patient_resource.get("birthDate"))
    age = _compute_age(birth_date, reference_date)
    gender = str(patient_resource.get("gender") or "unknown")
    conditions = tuple(sorted({name for name in _extract_conditions(resources) if name}))

    return SyntheaPatientProfile(
        patient_id=patient_id,
        source_fhir_patient_id=source_fhir_patient_id,
        given_name=given_name,
        family_name=family_name,
        gender=gender,
        birth_date=birth_date,
        age=age,
        conditions=conditions,
        source_path=source_path,
    )


def _iter_resources(payload: dict[str, Any]) -> list[dict[str, Any]]:
    resource_type = payload.get("resourceType")
    if resource_type == "Bundle":
        resources: list[dict[str, Any]] = []
        for entry in payload.get("entry", []):
            resource = entry.get("resource") if isinstance(entry, dict) else None
            if isinstance(resource, dict):
                resources.append(resource)
        return resources

    if resource_type:
        return [payload]

    return []


def _extract_name(patient_resource: dict[str, Any]) -> tuple[str, str]:
    names = patient_resource.get("name")
    if not isinstance(names, list):
        return "", ""

    for name in names:
        if not isinstance(name, dict):
            continue
        given = name.get("given") or []
        family = name.get("family") or ""
        if isinstance(given, list):
            given_name = " ".join(str(part) for part in given if part)
        else:
            given_name = str(given)
        return given_name, str(family)

    return "", ""


def _parse_birth_date(raw: Any) -> date | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


def _compute_age(birth_date: date | None, reference_date: date) -> int | None:
    if birth_date is None:
        return None
    years = reference_date.year - birth_date.year
    if (reference_date.month, reference_date.day) < (birth_date.month, birth_date.day):
        years -= 1
    return max(0, years)


def _extract_conditions(resources: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for resource in resources:
        if resource.get("resourceType") != "Condition":
            continue
        code = resource.get("code")
        if not isinstance(code, dict):
            continue
        text = code.get("text")
        if isinstance(text, str) and text.strip():
            names.append(text.strip())
            continue
        for coding in code.get("coding", []):
            if not isinstance(coding, dict):
                continue
            display = coding.get("display")
            if isinstance(display, str) and display.strip():
                names.append(display.strip())
                break
    return names
