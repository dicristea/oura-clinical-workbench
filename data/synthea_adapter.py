from __future__ import annotations

import json
import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from data.base import DataSource, PatientTimeSeries
from data.feature_registry import get_feature_groups_for_source

# ---------------------------------------------------------------------------
# LOINC code → registry feature name
# Synthea uses LOINC codes for all Observation resources.
# ---------------------------------------------------------------------------
_LOINC_TO_FEATURE: dict[str, str] = {
    "8867-4":  "heart_rate",              # Heart rate
    "8480-6":  "systolic_bp",             # Systolic blood pressure
    "8462-4":  "diastolic_bp",            # Diastolic blood pressure
    "9279-1":  "respiratory_rate",        # Respiratory rate
    "8310-5":  "body_temperature",        # Body temperature
    "29463-7": "body_weight_kg",          # Body weight
    "39156-5": "bmi",                     # Body mass index
    "2339-0":  "glucose_mgdl",            # Glucose (blood)
    "4548-4":  "hba1c_pct",              # Hemoglobin A1c
    "2093-3":  "total_cholesterol_mgdl",  # Total cholesterol
    "18262-6": "ldl_cholesterol_mgdl",    # LDL cholesterol
}

# SNOMED condition codes we extract as boolean static_features
_SNOMED_TO_CONDITION: dict[str, str] = {
    "44054006":  "diabetes_type2",
    "38341003":  "hypertension",
    "414545008": "ischemic_heart_disease",
    "40930008":  "hypothyroidism",
    "59621000":  "essential_hypertension",
    "73211009":  "diabetes_mellitus",
}


def _parse_fhir_bundle(path: str) -> dict[str, Any]:
    """Load a FHIR Bundle JSON file and return it as a dict."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _extract_patient_meta(bundle: dict) -> dict:
    """Pull demographics from the first Patient resource in the bundle."""
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Patient":
            meta: dict = {}
            meta["fhir_id"] = resource.get("id", "")
            gender = resource.get("gender", "")
            meta["sex"] = "M" if gender == "male" else "F"
            birth_date = resource.get("birthDate", "")
            if birth_date:
                try:
                    dob = datetime.strptime(birth_date, "%Y-%m-%d")
                    meta["age"] = (datetime(2024, 12, 31) - dob).days // 365
                except ValueError:
                    pass
            name_list = resource.get("name", [])
            if name_list:
                name = name_list[0]
                family = name.get("family", "")
                given = " ".join(name.get("given", []))
                meta["name"] = f"{given} {family}".strip()
            return meta
    return {}


def _extract_observations(bundle: dict) -> pd.DataFrame:
    """Parse Observation resources and return a date-indexed DataFrame."""
    rows: list[dict] = []
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") != "Observation":
            continue

        # Date — prefer effectiveDateTime, fall back to effectivePeriod.start
        date_str = resource.get("effectiveDateTime") or (
            resource.get("effectivePeriod", {}).get("start")
        )
        if not date_str:
            continue
        try:
            obs_date = pd.to_datetime(date_str).normalize()
        except Exception:
            continue

        # LOINC code
        loinc_code = None
        for coding in resource.get("code", {}).get("coding", []):
            if coding.get("system", "").endswith("loinc.org"):
                loinc_code = coding.get("code")
                break
        if loinc_code is None or loinc_code not in _LOINC_TO_FEATURE:
            continue

        feature_name = _LOINC_TO_FEATURE[loinc_code]

        # Value — valueQuantity is the most common form
        value = None
        if "valueQuantity" in resource:
            value = resource["valueQuantity"].get("value")
        elif "valueCodeableConcept" in resource:
            value = resource["valueCodeableConcept"].get("text")
        elif "component" in resource:
            # Blood pressure is stored as a multi-component Observation
            for comp in resource["component"]:
                comp_loinc = None
                for coding in comp.get("code", {}).get("coding", []):
                    if coding.get("system", "").endswith("loinc.org"):
                        comp_loinc = coding.get("code")
                        break
                if comp_loinc in _LOINC_TO_FEATURE:
                    comp_feat = _LOINC_TO_FEATURE[comp_loinc]
                    comp_value = comp.get("valueQuantity", {}).get("value")
                    if comp_value is not None:
                        rows.append({"date": obs_date, comp_feat: comp_value})
            continue

        if value is not None:
            try:
                rows.append({"date": obs_date, feature_name: float(value)})
            except (TypeError, ValueError):
                pass

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Pivot: one row per date, one column per feature (mean across same-day dups)
    df = df.groupby("date").mean(numeric_only=True)
    df.index = pd.DatetimeIndex(df.index)
    return df.sort_index()


def _extract_conditions(bundle: dict) -> dict[str, bool]:
    """Extract active Condition resources as boolean flags."""
    conditions: dict[str, bool] = {}
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") != "Condition":
            continue
        for coding in resource.get("code", {}).get("coding", []):
            code = coding.get("code", "")
            if code in _SNOMED_TO_CONDITION:
                conditions[_SNOMED_TO_CONDITION[code]] = True
    return conditions


def _compute_risk_level(ts: pd.DataFrame) -> str:
    """Derive a risk level from glucose and systolic BP trends."""
    glucose_mean = ts["glucose_mgdl"].mean() if "glucose_mgdl" in ts.columns else None
    sbp_mean = ts["systolic_bp"].mean() if "systolic_bp" in ts.columns else None
    hba1c_mean = ts["hba1c_pct"].mean() if "hba1c_pct" in ts.columns else None

    high_flags: list[bool] = []
    medium_flags: list[bool] = []

    if glucose_mean and not pd.isna(glucose_mean):
        if glucose_mean > 200:
            high_flags.append(True)
        elif glucose_mean > 126:
            medium_flags.append(True)

    if sbp_mean and not pd.isna(sbp_mean):
        if sbp_mean > 160:
            high_flags.append(True)
        elif sbp_mean > 140:
            medium_flags.append(True)

    if hba1c_mean and not pd.isna(hba1c_mean):
        if hba1c_mean > 9.0:
            high_flags.append(True)
        elif hba1c_mean > 7.0:
            medium_flags.append(True)

    if high_flags:
        return "high"
    if medium_flags:
        return "medium"
    return "low"


def _build_feature_groups() -> dict[str, list[str]]:
    return {
        group: [fc.name for fc in fcs]
        for group, fcs in get_feature_groups_for_source(DataSource.SYNTHEA).items()
    }


# ---------------------------------------------------------------------------
# SyntheaAdapter
# ---------------------------------------------------------------------------

class SyntheaAdapter:
    """Loads patient data from Synthea FHIR R4 JSON bundles or generates demo data.

    Synthea (https://github.com/synthetichealth/synthea) produces one JSON
    file per patient containing a FHIR Bundle with Patient, Observation,
    Condition, and other resources.  This adapter extracts the subset of
    features registered in ``data.feature_registry.SYNTHEA_FEATURES``.

    Usage
    -----
    Load a real (or Synthea-generated) bundle::

        adapter = SyntheaAdapter()
        pts = adapter.load_from_fhir("demo_data/demo_synthea/PT-3001.json", "PT-3001")

    Generate reproducible demo data (no file required)::

        pts = adapter.load_demo_data("PT-3001", risk_level="medium")
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def load_from_fhir(self, filepath: str, patient_id: str) -> PatientTimeSeries:
        """Build a PatientTimeSeries from a Synthea FHIR Bundle JSON file.

        Args:
            filepath:   Path to the ``.json`` FHIR bundle file.
            patient_id: Identifier to assign (overrides the FHIR Patient.id).

        Returns:
            A PatientTimeSeries with a DatetimeIndex time_series.

        Raises:
            FileNotFoundError: If *filepath* does not exist.
            ValueError: If the bundle contains no usable Observation records.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"FHIR bundle not found: {filepath}")

        bundle = _parse_fhir_bundle(filepath)

        patient_meta = _extract_patient_meta(bundle)
        time_series = _extract_observations(bundle)
        conditions = _extract_conditions(bundle)

        if time_series.empty:
            raise ValueError(
                f"No recognised Observation resources found in {filepath}. "
                "Ensure the bundle contains LOINC-coded vitals/labs."
            )

        static_features: dict = {**patient_meta, **conditions}
        risk_level = _compute_risk_level(time_series)

        metadata = {
            "data_source_label": "Synthea FHIR",
            "data_points_count": len(time_series),
            "risk_level": risk_level,
            "loaded_from": filepath,
        }

        return PatientTimeSeries(
            patient_id=str(patient_id),
            data_source=DataSource.SYNTHEA,
            static_features=static_features,
            time_series=time_series,
            metadata=metadata,
            feature_groups=_build_feature_groups(),
        )

    def load_demo_data(
        self,
        patient_id: str,
        risk_level: str = "medium",
    ) -> PatientTimeSeries:
        """Generate synthetic Synthea-style encounter data for demo/testing.

        Produces 12 monthly encounter rows (one year) with clinically
        plausible vitals and metabolic labs, scaled to the requested risk.

        Risk thresholds:
            high   – glucose > 200, HbA1c > 9, SBP > 160
            medium – glucose 130–200, HbA1c 7–9, SBP 140–160
            low    – glucose < 100, HbA1c < 6, SBP < 130

        Args:
            patient_id: Arbitrary identifier string.
            risk_level: ``"high"``, ``"medium"``, or ``"low"``.

        Returns:
            A PatientTimeSeries with 12 monthly encounter rows.
        """
        rng = random.Random(hash((patient_id, risk_level)))

        end_date = datetime(2024, 12, 1)
        dates = [end_date - timedelta(days=30 * i) for i in range(11, -1, -1)]
        index = pd.DatetimeIndex(dates)
        n = len(dates)

        def _walk(start: float, end: float, noise: float = 0.04) -> list[float]:
            vals = [start]
            step = (end - start) / (n - 1)
            for i in range(1, n):
                vals.append(vals[-1] + step + rng.gauss(0, abs(end - start) * noise))
            return [round(v, 1) for v in vals]

        if risk_level == "high":
            glucose     = _walk(rng.uniform(195, 230), rng.uniform(200, 250))
            hba1c       = _walk(rng.uniform(9.0, 10.5), rng.uniform(9.5, 11.0))
            sbp         = _walk(rng.uniform(155, 175), rng.uniform(160, 180))
            dbp         = _walk(rng.uniform(95, 105), rng.uniform(95, 110))
            weight      = _walk(rng.uniform(95, 120), rng.uniform(95, 125))
            bmi_vals    = _walk(rng.uniform(33, 40), rng.uniform(33, 42))
            cholesterol = _walk(rng.uniform(220, 260), rng.uniform(220, 270))
            ldl         = _walk(rng.uniform(140, 180), rng.uniform(140, 185))
        elif risk_level == "medium":
            glucose     = _walk(rng.uniform(130, 170), rng.uniform(125, 180))
            hba1c       = _walk(rng.uniform(7.0, 8.5), rng.uniform(7.0, 9.0))
            sbp         = _walk(rng.uniform(138, 155), rng.uniform(135, 160))
            dbp         = _walk(rng.uniform(85, 95), rng.uniform(85, 98))
            weight      = _walk(rng.uniform(78, 95), rng.uniform(78, 98))
            bmi_vals    = _walk(rng.uniform(27, 33), rng.uniform(27, 35))
            cholesterol = _walk(rng.uniform(190, 220), rng.uniform(190, 225))
            ldl         = _walk(rng.uniform(110, 140), rng.uniform(110, 145))
        else:  # low
            glucose     = _walk(rng.uniform(75, 100), rng.uniform(75, 105))
            hba1c       = _walk(rng.uniform(4.8, 5.7), rng.uniform(4.8, 5.9))
            sbp         = _walk(rng.uniform(108, 128), rng.uniform(108, 130))
            dbp         = _walk(rng.uniform(65, 80), rng.uniform(65, 82))
            weight      = _walk(rng.uniform(60, 80), rng.uniform(60, 82))
            bmi_vals    = _walk(rng.uniform(20, 26), rng.uniform(20, 27))
            cholesterol = _walk(rng.uniform(150, 185), rng.uniform(150, 188))
            ldl         = _walk(rng.uniform(80, 110), rng.uniform(80, 112))

        hr   = [round(rng.uniform(58, 88), 0) for _ in range(n)]
        rr   = [round(rng.uniform(13, 18), 0) for _ in range(n)]
        temp = [round(rng.uniform(36.4, 37.2), 1) for _ in range(n)]

        time_series = pd.DataFrame(
            {
                "heart_rate":             hr,
                "systolic_bp":            sbp,
                "diastolic_bp":           dbp,
                "respiratory_rate":       rr,
                "body_temperature":       temp,
                "body_weight_kg":         weight,
                "bmi":                    bmi_vals,
                "glucose_mgdl":           glucose,
                "hba1c_pct":              hba1c,
                "total_cholesterol_mgdl": cholesterol,
                "ldl_cholesterol_mgdl":   ldl,
            },
            index=index,
        )

        static_features: dict = {
            "age":  rng.randint(35, 75),
            "sex":  rng.choice(["M", "F"]),
            "diabetes_type2":  risk_level in ("high", "medium"),
            "hypertension":    risk_level in ("high", "medium"),
        }

        metadata = {
            "data_source_label": "Synthea FHIR",
            "data_points_count": n,
            "risk_level": risk_level,
            "cohort": "demo",
            "encounter_type": "ambulatory",
            "enrollment_date": dates[0].strftime("%Y-%m-%d"),
        }

        return PatientTimeSeries(
            patient_id=patient_id,
            data_source=DataSource.SYNTHEA,
            static_features=static_features,
            time_series=time_series,
            metadata=metadata,
            feature_groups=_build_feature_groups(),
        )

    def load_all_from_dir(self, data_dir: str) -> list[PatientTimeSeries]:
        """Load every ``*.json`` FHIR bundle found in *data_dir*.

        Assigns patient IDs from the filename stem (e.g. ``PT-3001.json``
        → patient_id ``"PT-3001"``).  Files that fail to parse are skipped
        with a warning.

        Args:
            data_dir: Path to a directory containing Synthea bundle files.

        Returns:
            A list of PatientTimeSeries (one per successfully loaded file).
        """
        results: list[PatientTimeSeries] = []
        for path in sorted(Path(data_dir).glob("*.json")):
            patient_id = path.stem
            try:
                pts = self.load_from_fhir(str(path), patient_id)
                results.append(pts)
            except Exception as exc:
                print(f"[SyntheaAdapter] Skipping {path.name}: {exc}")
        return results
