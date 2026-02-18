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


# ── PPMI features ─────────────────────────────────────────────────────────────

PPMI_FEATURES: list[FeatureConfig] = [
    # Genetic
    FeatureConfig(
        name="gba_mutation",
        display_name="GBA Mutation Status",
        group="Genetic",
        unit="",
        description="Glucocerebrosidase gene mutation carrier status (0 = negative, "
        "1 = heterozygous, 2 = homozygous); strongest known genetic risk "
        "factor for rapid PD motor progression.",
        default_selected=True,
    ),
    FeatureConfig(
        name="lrrk2_mutation",
        display_name="LRRK2 Mutation",
        group="Genetic",
        unit="",
        description="Leucine-rich repeat kinase 2 pathogenic variant status; "
        "associated with slower progression in some cohorts.",
        default_selected=True,
    ),
    FeatureConfig(
        name="apoe_status",
        display_name="APOE Status",
        group="Genetic",
        unit="",
        description="Apolipoprotein E allele status (e2/e3/e4); e4 carriers show "
        "accelerated cognitive decline alongside motor symptoms.",
        default_selected=False,
    ),
    # Biofluid
    FeatureConfig(
        name="csf_alpha_synuclein",
        display_name="CSF Alpha-synuclein",
        group="Biofluid",
        unit="pg/mL",
        description="Cerebrospinal fluid alpha-synuclein concentration; lower levels "
        "at baseline predict faster motor decline in de novo PD.",
        default_selected=True,
    ),
    FeatureConfig(
        name="amyloid_beta",
        display_name="Amyloid-beta (CSF)",
        group="Biofluid",
        unit="pg/mL",
        description="CSF amyloid-beta 1–42 level; co-pathology marker that may "
        "accelerate motor and cognitive progression.",
        default_selected=False,
    ),
    FeatureConfig(
        name="total_tau",
        display_name="Total Tau (CSF)",
        group="Biofluid",
        unit="pg/mL",
        description="Total tau protein in CSF; elevated levels suggest concurrent "
        "neurodegeneration beyond alpha-synuclein pathology.",
        default_selected=False,
    ),
    # Clinical
    FeatureConfig(
        name="baseline_updrs",
        display_name="Baseline UPDRS III",
        group="Clinical",
        unit="score",
        description="MDS-UPDRS Part III (Motor Examination) score at enrollment; "
        "primary target variable and the strongest predictor of future scores.",
        default_selected=True,
    ),
    FeatureConfig(
        name="epworth_sleep",
        display_name="Epworth Sleep Scale",
        group="Clinical",
        unit="score",
        description="Epworth Sleepiness Scale (0–24); excessive daytime sleepiness "
        "is a common non-motor symptom and HE confounder.",
        default_selected=False,
    ),
    FeatureConfig(
        name="schwab_england_adl",
        display_name="Schwab & England ADL",
        group="Clinical",
        unit="%",
        description="Schwab & England Activities of Daily Living scale (0–100%); "
        "captures functional independence independently of motor score.",
        default_selected=False,
    ),
    # Imaging
    FeatureConfig(
        name="datscan",
        display_name="DaTscan",
        group="Imaging",
        unit="SBR",
        description="Dopamine transporter SPECT striatal binding ratio; lower values "
        "confirm dopaminergic deficit and correlate with motor severity.",
        default_selected=True,
    ),
]


# ── Registry lookup ───────────────────────────────────────────────────────────

_REGISTRY: dict[DataSource, list[FeatureConfig]] = {
    DataSource.OURA: OURA_FEATURES,
    DataSource.PPMI: PPMI_FEATURES,
    DataSource.SYNTHEA: [],  # populated dynamically by the Synthea adapter
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
        same order features were declared in OURA_FEATURES / PPMI_FEATURES.
    """
    groups: dict[str, list[FeatureConfig]] = defaultdict(list)
    for fc in get_features_for_source(data_source):
        groups[fc.group].append(fc)
    return dict(groups)
