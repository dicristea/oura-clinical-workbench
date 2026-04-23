"""Open Wearables adapter — unified wearable API (https://www.openwearables.io).

STATUS: Partial — Oura Ring integration is planned for Q1 2026.
        The FHIR bundle reader is production-ready today.
        The live-API path is stubbed and documented for when it ships.

Architecture fit
----------------
Open Wearables normalises every supported device into the same FHIR R4 Observation
schema it already produces for Synthea bundles.  That means ``_extract_observations``
from ``synthea_adapter.py`` works without modification — we only need a different
LOINC→feature mapping (Open Wearables uses the same LOINC codes but also exposes
its own vendor-normalised JSON keys).

Wire-up checklist (when Oura integration ships)
------------------------------------------------
1.  Set ``OPEN_WEARABLES_API_KEY`` in the environment / Render config.
2.  Set ``OPEN_WEARABLES_BASE_URL`` (default: https://api.openwearables.io/v1).
3.  Replace the ``load_from_oura_adapter()`` call in ``app.py`` with
    ``OpenWearablesAdapter().load_from_api(patient_id, device="oura")``.
4.  OAuth consent redirect is handled by Open Wearables — no per-patient token
    storage needed on our side.

Field-name mapping  (Open Wearables → our internal names)
----------------------------------------------------------
Open Wearables JSON key      Our feature_registry name
---------------------------  -------------------------
hrv_rmssd                    hrv_rmssd          (was hrv_balance for Oura V2)
resting_heart_rate           resting_heart_rate (was resting_hr)
steps                        steps              (was step_count)
rem_sleep_percentage         rem_sleep_pct
deep_sleep_percentage        deep_sleep_pct
sleep_latency_minutes        sleep_latency
skin_temperature_deviation   body_temp_deviation
respiratory_rate             respiratory_rate

NOTE: hrv_rmssd and resting_heart_rate use the Open Wearables names directly
because they are more precise than the Oura V2 API naming.  The feature_registry
for DataSource.OPEN_WEARABLES already uses these names (see data/base.py).
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from data.base import DataSource, PatientTimeSeries
from data.synthea_adapter import (
    _parse_fhir_bundle,
    _extract_patient_meta,
    _extract_conditions,
    _compute_risk_level,
)


# ---------------------------------------------------------------------------
# Open Wearables normalised JSON key → our internal feature name
# ---------------------------------------------------------------------------
_OW_KEY_TO_FEATURE: dict[str, str] = {
    "hrv_rmssd":                 "hrv_rmssd",
    "resting_heart_rate":        "resting_heart_rate",
    "steps":                     "steps",
    "rem_sleep_percentage":      "rem_sleep_pct",
    "deep_sleep_percentage":     "deep_sleep_pct",
    "sleep_latency_minutes":     "sleep_latency",
    "skin_temperature_deviation":"body_temp_deviation",
    "respiratory_rate":          "respiratory_rate",
    # Readiness / recovery score (Open Wearables unified score)
    "readiness_score":           "readiness_score",
    "sleep_score":               "sleep_score",
}

# LOINC codes Open Wearables uses in its FHIR Observation output
# (same as Oura V2 FHIR export — identical to synthea_adapter._LOINC_TO_FEATURE
#  for the subset that Oura tracks)
_OW_LOINC_TO_FEATURE: dict[str, str] = {
    "8867-4":  "resting_heart_rate",   # Heart rate
    "80404-7": "hrv_rmssd",            # HRV rMSSD
    "9279-1":  "respiratory_rate",     # Respiratory rate
    "55284-4": "body_temp_deviation",  # Body temperature deviation
}


# ---------------------------------------------------------------------------
# FHIR bundle reader (re-uses synthea_adapter helpers)
# ---------------------------------------------------------------------------

def _extract_ow_observations(bundle: dict) -> pd.DataFrame:
    """Parse an Open Wearables FHIR Bundle into a DatetimeIndex DataFrame.

    Handles both LOINC-coded Observations (identical path to SyntheaAdapter)
    and Open Wearables' proprietary ``valueExtension`` blocks that carry the
    normalised JSON keys listed in ``_OW_KEY_TO_FEATURE``.
    """
    rows: list[dict] = []
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") != "Observation":
            continue

        date_str = resource.get("effectiveDateTime") or (
            resource.get("effectivePeriod", {}).get("start")
        )
        if not date_str:
            continue
        try:
            obs_date = pd.to_datetime(date_str).normalize()
        except Exception:
            continue

        # Path 1: standard LOINC code
        loinc_code = None
        for coding in resource.get("code", {}).get("coding", []):
            if coding.get("system", "").endswith("loinc.org"):
                loinc_code = coding.get("code")
                break
        if loinc_code and loinc_code in _OW_LOINC_TO_FEATURE:
            value = resource.get("valueQuantity", {}).get("value")
            if value is not None:
                try:
                    rows.append({"date": obs_date, _OW_LOINC_TO_FEATURE[loinc_code]: float(value)})
                except (TypeError, ValueError):
                    pass
            continue

        # Path 2: Open Wearables normalised extension keys
        for ext in resource.get("extension", []):
            url = ext.get("url", "")
            ow_key = url.split("/")[-1]   # e.g. ".../hrv_rmssd" → "hrv_rmssd"
            if ow_key in _OW_KEY_TO_FEATURE:
                value = ext.get("valueDecimal") or ext.get("valueQuantity", {}).get("value")
                if value is not None:
                    try:
                        rows.append({"date": obs_date, _OW_KEY_TO_FEATURE[ow_key]: float(value)})
                    except (TypeError, ValueError):
                        pass

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.groupby("date").mean(numeric_only=True)
    df.index = pd.DatetimeIndex(df.index)
    return df.sort_index()


# ---------------------------------------------------------------------------
# OpenWearablesAdapter
# ---------------------------------------------------------------------------

class OpenWearablesAdapter:
    """Loads patient data from Open Wearables (https://www.openwearables.io).

    Two paths are available:

    1.  ``load_from_fhir(filepath, patient_id)``  — reads a local FHIR bundle
        exported from the Open Wearables dashboard.  Works today for any device
        Open Wearables already supports (Apple Health, Garmin, Whoop, etc.).

    2.  ``load_from_api(patient_id, device)``  — calls the live Open Wearables
        REST API.  Requires ``OPEN_WEARABLES_API_KEY`` in the environment.
        Oura support ships Q1 2026; other devices are available sooner.

    Environment variables
    ---------------------
    OPEN_WEARABLES_API_KEY   Required for load_from_api.
    OPEN_WEARABLES_BASE_URL  Optional; defaults to https://api.openwearables.io/v1.
    """

    BASE_URL = os.getenv("OPEN_WEARABLES_BASE_URL", "https://api.openwearables.io/v1")

    # ── FHIR bundle path (works today) ────────────────────────────────────────

    def load_from_fhir(self, filepath: str, patient_id: str) -> PatientTimeSeries:
        """Build a PatientTimeSeries from an Open Wearables FHIR export bundle.

        Args:
            filepath:   Path to the ``.json`` FHIR bundle exported from the
                        Open Wearables dashboard or downloaded via their API.
            patient_id: Identifier to assign to this record.

        Returns:
            PatientTimeSeries with DataSource.OPEN_WEARABLES.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Open Wearables FHIR bundle not found: {filepath}")

        bundle       = _parse_fhir_bundle(filepath)
        patient_meta = _extract_patient_meta(bundle)
        time_series  = _extract_ow_observations(bundle)
        conditions   = _extract_conditions(bundle)

        if time_series.empty:
            raise ValueError(
                f"No recognised Observation resources in {filepath}. "
                "Export the bundle from the Open Wearables dashboard with "
                "'Include FHIR Observations' enabled."
            )

        static_features = {**patient_meta, **conditions}
        risk_level      = _compute_risk_level(time_series)

        return PatientTimeSeries(
            patient_id=str(patient_id),
            data_source=DataSource.OPEN_WEARABLES,
            static_features=static_features,
            time_series=time_series,
            metadata={
                "data_source_label": "Open Wearables",
                "data_points_count": len(time_series),
                "risk_level":        risk_level,
                "loaded_from":       filepath,
            },
        )

    # ── Live API path (stub — wire up when Oura integration ships) ────────────

    def load_from_api(
        self,
        patient_id: str,
        device: str = "oura",
        days: int = 90,
    ) -> PatientTimeSeries:
        """Fetch wearable data from the Open Wearables REST API.

        STATUS: Stub.  Oura device support launches Q1 2026.
                Other devices (Apple Health, Garmin, Whoop) can be tested
                sooner — set ``device`` to the appropriate slug.

        To activate:
            1. Set OPEN_WEARABLES_API_KEY in your environment.
            2. Direct patients through the Open Wearables OAuth consent URL
               (handled by their hosted flow — no token storage needed here).
            3. Call this method; it fetches the last ``days`` days of data,
               parses the FHIR bundle returned by the /fhir endpoint, and
               returns a PatientTimeSeries ready for the rest of the pipeline.

        Args:
            patient_id: Open Wearables patient/user identifier.
            device:     Device slug (``"oura"``, ``"garmin"``, ``"whoop"``…).
            days:       How many days of history to fetch (default 90).

        Raises:
            EnvironmentError: If OPEN_WEARABLES_API_KEY is not set.
            NotImplementedError: Until this stub is fully wired.
        """
        api_key = os.getenv("OPEN_WEARABLES_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPEN_WEARABLES_API_KEY environment variable is not set. "
                "Obtain a key from https://www.openwearables.io."
            )

        # ── When wiring this up, replace the block below with: ────────────────
        #
        #   import requests
        #   end   = datetime.utcnow()
        #   start = end - timedelta(days=days)
        #   resp  = requests.get(
        #       f"{self.BASE_URL}/patients/{patient_id}/fhir/bundle",
        #       params={"device": device, "start": start.date(), "end": end.date()},
        #       headers={"Authorization": f"Bearer {api_key}"},
        #       timeout=30,
        #   )
        #   resp.raise_for_status()
        #   bundle = resp.json()
        #   time_series = _extract_ow_observations(bundle)
        #   ... build and return PatientTimeSeries ...
        #
        # ─────────────────────────────────────────────────────────────────────

        raise NotImplementedError(
            f"Open Wearables live API is stubbed. "
            f"Oura device integration ships Q1 2026. "
            f"See data/open_wearables_adapter.py for the wire-up checklist."
        )
