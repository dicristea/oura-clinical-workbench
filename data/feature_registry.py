from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict

from data.base import DataSource


@dataclass(frozen=True)
class FeatureConfig:
    """Metadata for a single model/UI feature.

    Attributes:
        name:             Internal snake_case key — must match the column name in
                          PatientTimeSeries.time_series (or key in static_features
                          for non-temporal features like genetic status).
        display_name:     Human-readable label shown in the UI (checkboxes,
                          chart axes, feature importance bars).
        group:            Feature group label — controls checkbox section headers
                          in Model Lab and the Data Explorer sidebar.
        unit:             Unit string appended to values in tooltips/axes.
                          Use "" for dimensionless or categorical features.
        description:      One-sentence tooltip shown on hover in the UI.
        default_selected: Whether the checkbox is pre-checked when a patient's
                          Model Lab page loads for the first time.
    """

    name: str
    display_name: str
    group: str
    unit: str
    description: str
    default_selected: bool = False


# ── Oura Ring features ────────────────────────────────────────────────────────

OURA_FEATURES: list[FeatureConfig] = [
    # Sleep Architecture
    FeatureConfig(
        name="rem_sleep_pct",
        display_name="REM Sleep %",
        group="Sleep Architecture",
        unit="%",
        description="Percentage of total sleep time spent in REM stage; "
        "reductions may indicate early hepatic encephalopathy.",
        default_selected=True,
    ),
    FeatureConfig(
        name="deep_sleep_pct",
        display_name="Deep Sleep %",
        group="Sleep Architecture",
        unit="%",
        description="Percentage of total sleep time in slow-wave (N3) sleep; "
        "disruptions correlate with toxin accumulation in cirrhosis.",
        default_selected=True,
    ),
    FeatureConfig(
        name="sleep_latency",
        display_name="Sleep Latency",
        group="Sleep Architecture",
        unit="min",
        description="Minutes from lights-out to first sustained sleep epoch; "
        "elevated latency can reflect nocturnal confusion or discomfort.",
        default_selected=False,
    ),
    # Physiological
    FeatureConfig(
        name="hrv_balance",
        display_name="HRV Balance",
        group="Physiological",
        unit="ms",
        description="Heart rate variability (rMSSD-based balance score) reflecting "
        "autonomic nervous system function; reduced in advanced liver disease.",
        default_selected=True,
    ),
    FeatureConfig(
        name="body_temp_deviation",
        display_name="Body Temp Deviation",
        group="Physiological",
        unit="°C",
        description="Nightly skin temperature deviation from the patient's personal "
        "baseline; fever and hypothermia are common HE confounders.",
        default_selected=True,
    ),
    FeatureConfig(
        name="resting_hr",
        display_name="Resting HR",
        group="Physiological",
        unit="bpm",
        description="Overnight resting heart rate; elevated values can indicate "
        "infection, decompensation, or medication effects.",
        default_selected=False,
    ),
    # Activity
    FeatureConfig(
        name="step_count",
        display_name="Step Count",
        group="Activity",
        unit="steps",
        description="Total daily step count; declining activity is an early marker "
        "of functional deterioration in cirrhosis patients.",
        default_selected=False,
    ),
    FeatureConfig(
        name="inactivity_alerts",
        display_name="Inactivity Alerts",
        group="Activity",
        unit="alerts/day",
        description="Number of inactivity alert events triggered during the day; "
        "high counts suggest prolonged sedentary periods.",
        default_selected=False,
    ),
]


# ── Synthea FHIR features ─────────────────────────────────────────────────────

SYNTHEA_FEATURES: list[FeatureConfig] = [
    # Vital Signs
    FeatureConfig(
        name="heart_rate",
        display_name="Heart Rate",
        group="Vital Signs",
        unit="bpm",
        description="Resting heart rate measured at each encounter.",
        default_selected=True,
    ),
    FeatureConfig(
        name="systolic_bp",
        display_name="Systolic BP",
        group="Vital Signs",
        unit="mmHg",
        description="Systolic blood pressure; elevated values indicate hypertension risk.",
        default_selected=True,
    ),
    FeatureConfig(
        name="diastolic_bp",
        display_name="Diastolic BP",
        group="Vital Signs",
        unit="mmHg",
        description="Diastolic blood pressure; key marker of cardiovascular load.",
        default_selected=True,
    ),
    FeatureConfig(
        name="respiratory_rate",
        display_name="Respiratory Rate",
        group="Vital Signs",
        unit="br/min",
        description="Breaths per minute; elevated values can signal respiratory distress.",
        default_selected=False,
    ),
    FeatureConfig(
        name="body_temperature",
        display_name="Body Temperature",
        group="Vital Signs",
        unit="°C",
        description="Core body temperature; deviations may indicate infection or inflammation.",
        default_selected=False,
    ),
    # Body Composition
    FeatureConfig(
        name="body_weight_kg",
        display_name="Body Weight",
        group="Body Composition",
        unit="kg",
        description="Patient weight recorded at each encounter.",
        default_selected=True,
    ),
    FeatureConfig(
        name="bmi",
        display_name="BMI",
        group="Body Composition",
        unit="kg/m²",
        description="Body mass index; key metabolic risk stratifier.",
        default_selected=True,
    ),
    # Metabolic
    FeatureConfig(
        name="glucose_mgdl",
        display_name="Blood Glucose",
        group="Metabolic",
        unit="mg/dL",
        description="Fasting blood glucose level; primary diabetes screening marker.",
        default_selected=True,
    ),
    FeatureConfig(
        name="hba1c_pct",
        display_name="HbA1c",
        group="Metabolic",
        unit="%",
        description="Glycated haemoglobin; reflects average blood glucose over 2–3 months.",
        default_selected=True,
    ),
    FeatureConfig(
        name="total_cholesterol_mgdl",
        display_name="Total Cholesterol",
        group="Metabolic",
        unit="mg/dL",
        description="Total serum cholesterol; cardiovascular disease risk factor.",
        default_selected=False,
    ),
    FeatureConfig(
        name="ldl_cholesterol_mgdl",
        display_name="LDL Cholesterol",
        group="Metabolic",
        unit="mg/dL",
        description="Low-density lipoprotein; primary target for cardiovascular prevention.",
        default_selected=False,
    ),
]


# ── Registry lookup ───────────────────────────────────────────────────────────

_REGISTRY: dict[DataSource, list[FeatureConfig]] = {
    DataSource.OURA: OURA_FEATURES,
    DataSource.SYNTHEA: SYNTHEA_FEATURES,
}


def get_features_for_source(data_source: DataSource) -> list[FeatureConfig]:
    """Return the ordered list of FeatureConfig objects for *data_source*.

    For DataSource.SYNTHEA the list is empty by default; the Synthea adapter
    is expected to register features at load time.
    """
    return list(_REGISTRY.get(data_source, []))


def get_feature_groups_for_source(
    data_source: DataSource,
) -> dict[str, list[FeatureConfig]]:
    """Return features grouped by their group label, preserving declaration order.

    Returns:
        An ordered dict mapping group name → list of FeatureConfig, in the
        same order features were declared in OURA_FEATURES / SYNTHEA_FEATURES.
    """
    groups: dict[str, list[FeatureConfig]] = defaultdict(list)
    for fc in get_features_for_source(data_source):
        groups[fc.group].append(fc)
    return dict(groups)
