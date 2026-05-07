"""
Clinical Coordinator Dashboard - Flask App
Exact mockup implementation
"""

from flask import Flask, render_template, jsonify, request
import pandas as pd
from datetime import datetime, timedelta
from functools import lru_cache
import json
import random
import re
import os
import uuid
import numpy as np

app = Flask(__name__)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


app.json_encoder = _NumpyEncoder

# Ordered color palette assigned to features by their declaration index.
STUDY_CONFIG_PATH             = 'demo_data/study_config.json'
DEMO_HE_PATIENT_ID            = 'PT-1042'   # Featured patient for demo/presentation
CONDITIONS_DICT_PATH          = 'data/conditions_dict.json'
STANDARD_WEARABLE_DATASET_DIR = 'demo_data/demo_omh_ieee'
NOTEBOOKS_DIR                 = 'notebooks'
JUPYTER_BASE_URL              = os.environ.get('JUPYTER_BASE_URL', 'http://localhost:8888')
CONFOUNDERS_PATH              = 'demo_data/confounders.json'

CONFOUNDER_TYPES = [
    'Narcotics / Sedatives',
    'Alcohol',
    'Illness / Infection',
    'Travel / Time Zone Change',
    'Sleep Apnea Exacerbation',
    'Other',
]

FEATURE_COLORS = [
    '#3b82f6', '#06b6d4', '#10b981', '#f59e0b',
    '#ef4444', '#8b5cf6', '#f97316', '#ec4899',
    '#84cc16', '#14b8a6', '#6366f1', '#a78bfa',
]


def load_synthea_patients():
    """Return dashboard patient dicts for every Synthea FHIR bundle found."""
    synthea_dir = 'demo_data/demo_synthea'
    if not os.path.isdir(synthea_dir):
        return []

    try:
        from data.synthea_adapter import SyntheaAdapter
        adapter = SyntheaAdapter()
    except Exception as e:
        print(f"[Synthea] Could not import adapter: {e}")
        return []

    patients = []
    for fname in sorted(os.listdir(synthea_dir)):
        if not fname.endswith('.json'):
            continue
        patient_id = fname[:-5]          # strip .json
        fpath = os.path.join(synthea_dir, fname)
        try:
            pts = adapter.load_from_fhir(fpath, patient_id)
        except Exception as e:
            print(f"[Synthea] Skipping {fname}: {e}")
            continue

        ts   = pts.time_series
        meta = pts.metadata
        sf   = pts.static_features

        last_date = ts.index.max() if not ts.empty else None
        last_dt       = last_date.to_pydatetime().replace(tzinfo=None) if last_date else None
        last_sync_str = format_last_sync(last_dt) if last_dt else "Never"
        days_since    = (datetime.now() - last_dt).days if last_dt else 999

        if days_since <= 4:
            status = "active"
        elif days_since <= 30:
            status = "follow-up"
        else:
            status = "outreach"

        condition_str = ", ".join(
            k.replace("_", " ").title() for k, v in sf.items()
            if isinstance(v, bool) and v
        ) or "None"

        # Last visit: second-to-last encounter date (simulates prior clinic visit)
        _visit_rng = random.Random(hash(patient_id))
        if len(ts) >= 2:
            _vdt = ts.index[-2].to_pydatetime()
            _vtype = _visit_rng.choice(['Virtual', 'In-Person'])
            last_visit_str = f"{_vdt.strftime('%b %d, %Y')} ({_vtype})"
        elif not ts.empty:
            last_visit_str = ts.index[-1].to_pydatetime().strftime('%b %d, %Y')
        else:
            last_visit_str = None

        # Model status derived from risk level
        risk = meta.get('risk_level', '')
        if risk == 'high':
            model_status_level = 'alert'
            model_status_label = 'Risk Signal Detected'
        elif risk == 'medium':
            model_status_level = 'warn'
            model_status_label = 'Borderline Range'
        elif risk == 'low':
            model_status_level = 'ok'
            model_status_label = 'Within Normal Range'
        else:
            model_status_level = 'pending'
            model_status_label = 'Insufficient Data'

        patients.append({
            'id':                  patient_id,
            'mrn':                 patient_id,
            'name':                sf.get('name', patient_id),
            'inpatient':           0,
            'outpatient':          len(ts),
            'last_sync':           last_sync_str,
            'last_visit':          last_visit_str,
            'model_status_level':  model_status_level,
            'model_status_label':  model_status_label,
            'has_oura':            False,
            'has_ehr':             False,
            'has_synthea':         True,
            'data_source':         'synthea',
            'status':              status,
            'hospital_start':      None,
            'hospital_end':        None,
            'risk_level':          risk,
            'conditions':          condition_str,
            'sleep_score':         [],
            'hrv_average':         [],
            'activity_score':      [],
        })

    return patients


@lru_cache(maxsize=1)
def _standard_wearable_series_by_patient() -> dict:
    """Load OMH/IEEE wearable records and cache PatientTimeSeries by patient id."""
    try:
        from data.standard_wearable_adapter import StandardWearableAdapter
    except Exception as e:
        print(f"[StandardWearable] Could not import adapter: {e}")
        return {}

    by_patient = {}
    adapter = StandardWearableAdapter()
    if not os.path.isdir(STANDARD_WEARABLE_DATASET_DIR):
        return {}
    try:
        for pts in adapter.load_all_from_dir(STANDARD_WEARABLE_DATASET_DIR):
            by_patient[pts.patient_id] = pts
    except Exception as e:
        print(f"[StandardWearable] Could not load {STANDARD_WEARABLE_DATASET_DIR}: {e}")
    return by_patient


def get_standard_wearable_series(patient_id: str):
    """Return standard OMH/IEEE wearable time series for one patient, if present."""
    return _standard_wearable_series_by_patient().get(str(patient_id))


def load_standard_wearable_patients():
    """Return dashboard patient dicts backed by OMH/IEEE wearable records."""
    patients = []
    for patient_id, pts in _standard_wearable_series_by_patient().items():
        ts = pts.time_series
        meta = pts.metadata
        sf = pts.static_features

        last_date = ts.index.max() if not ts.empty else None
        last_dt = last_date.to_pydatetime().replace(tzinfo=None) if last_date else None
        last_sync_str = format_last_sync(last_dt) if last_dt else "Never"
        days_since = (datetime.now() - last_dt).days if last_dt else 999

        if days_since <= 4:
            status = "active"
        elif days_since <= 30:
            status = "follow-up"
        else:
            status = "outreach"

        risk = meta.get('risk_level', '')
        if risk == 'high':
            model_status_level = 'alert'
            model_status_label = 'Sleep Risk Signal'
        elif risk == 'medium':
            model_status_level = 'warn'
            model_status_label = 'Borderline Sleep Range'
        elif risk == 'low':
            model_status_level = 'ok'
            model_status_label = 'Wearable Signals Stable'
        else:
            model_status_level = 'pending'
            model_status_label = 'Insufficient Data'

        conditions = sf.get('conditions') or 'None'
        patients.append({
            'id':                  patient_id,
            'mrn':                 patient_id,
            'name':                sf.get('name') or patient_id,
            'inpatient':           0,
            'outpatient':          len(ts),
            'last_sync':           last_sync_str,
            'last_visit':          last_dt.strftime('%b %d, %Y') if last_dt else None,
            'model_status_level':  model_status_level,
            'model_status_label':  model_status_label,
            'has_oura':            True,
            'has_ehr':             False,
            'has_synthea':         False,
            'has_omh_ieee':        True,
            'data_source':         'omh_ieee',
            'status':              status,
            'hospital_start':      None,
            'hospital_end':        None,
            'risk_level':          risk,
            'conditions':          conditions,
            'sleep_score':         [],
            'hrv_average':         [],
            'activity_score':      [],
        })
    return patients


def merge_standard_wearable_patients(patients: list[dict]) -> list[dict]:
    """Attach OMH/IEEE wearable availability to matching dashboard patients."""
    wearable_patients = load_standard_wearable_patients()
    by_id = {p['id']: p for p in patients}

    for wearable_patient in wearable_patients:
        patient_id = wearable_patient['id']
        existing = by_id.get(patient_id)
        if existing is None:
            patients.append(wearable_patient)
            by_id[patient_id] = wearable_patient
            continue

        existing['has_oura'] = True
        existing['has_omh_ieee'] = True
        existing['data_source'] = (
            'synthea_omh_ieee' if existing.get('has_synthea') else 'omh_ieee'
        )
        existing['last_sync'] = wearable_patient.get('last_sync') or existing.get('last_sync')
        existing['status'] = wearable_patient.get('status') or existing.get('status')
        existing['outpatient'] = max(
            int(existing.get('outpatient') or 0),
            int(wearable_patient.get('outpatient') or 0),
        )
        existing['model_status_level'] = wearable_patient.get(
            'model_status_level',
            existing.get('model_status_level'),
        )
        existing['model_status_label'] = wearable_patient.get(
            'model_status_label',
            existing.get('model_status_label'),
        )

    return patients


def _make_demo_he_patient() -> dict:
    """
    Returns a hardcoded demo patient for the HE sleep biomarker study.
    Used during presentations — tells a clear clinical story without real PHI.
    """
    return {
        'id':                  DEMO_HE_PATIENT_ID,
        'mrn':                 DEMO_HE_PATIENT_ID,
        'name':                'Synthea Demo',
        'inpatient':           14,
        'outpatient':          14,
        'last_sync':           '2 hours ago',
        'last_visit':          'Dec 08, 2024 (In-Person)',
        'model_status_level':  'alert',
        'model_status_label':  'Sleep Risk Signal Detected',
        'has_oura':            True,
        'has_ehr':             True,
        'has_synthea':         False,
        'data_source':         'oura',
        'status':              'active',
        'hospital_start':      None,
        'hospital_end':        None,
        'participation_start': 'Nov 15, 2022',
        'risk_level':          'high',
        'conditions':          'Covert Hepatic Encephalopathy',
        # Sparklines shown on dashboard (14-day window matching the biometrics story)
        'sleep_score':    [72, 68, 75, 65, 70, 67, 71, 55, 73, 63, 61, 58, 60, 53],
        'hrv_average':    [28, 24, 31, 19, 22, 26, 18, 25, 21, 17, 20, 23, 19, 16],
        'activity_score': [72, 68, 71, 59, 64, 61, 58, 42, 56, 48, 41, 37, 29, 21],
    }


def _demo_he_overrides() -> dict:
    """
    Hand-crafted 14-day time series for PT-1042.

    Clinical narrative:
      Week 1 — borderline: REM 18-21 %, HRV low but stable, efficiency >80 %.
      Dec 8  — anomalous night: SpO₂ dips to 91.8 %, temp deviation spikes to +0.93 °C,
               sleep latency 48 min + WASO 68 min (all three auto-flagged by the system).
      Week 2 — clear deterioration: REM drops to 11-12 %, step count halves,
               efficiency falls below 70 % by Dec 14.

    This lets the viewer point to auto-detected confounders in the Risk Analysis Lab
    and a visually clear downward trend across every biometric chart.
    """
    sleep = [7.2, 6.8, 7.5, 6.1, 7.0, 6.4, 6.9, 5.8, 7.1, 6.2, 6.7, 5.9, 6.3, 5.7]
    rem   = [1.5, 1.4, 1.4, 1.1, 1.2, 1.1, 1.1, 0.9, 1.0, 0.8, 0.9, 0.7, 0.8, 0.6]
    deep  = [1.4, 1.2, 1.3, 1.0, 1.1, 1.0, 1.0, 0.8, 1.0, 0.9, 1.0, 0.8, 0.9, 0.8]
    awake = [0.4, 0.3, 0.4, 0.4, 0.5, 0.5, 0.4, 0.8, 0.4, 0.6, 0.7, 0.7, 0.6, 0.8]
    light = [round(sleep[i] - rem[i] - deep[i] - awake[i], 1) for i in range(14)]
    lat   = [14, 18, 12, 22, 16, 28, 20, 48, 15, 31, 35, 42, 38, 52]
    waso  = [22, 18, 25, 31, 28, 35, 30, 68, 25, 42, 48, 55, 52, 71]
    tib   = [round(sleep[i] + awake[i] + lat[i]/60 + waso[i]/60, 1) for i in range(14)]
    eff   = [round(sleep[i] / tib[i] * 100, 1) for i in range(14)]

    return {
        'heart_rate_data':     [58, 61, 59, 63, 60, 62, 64, 61, 65, 63, 62, 67, 64, 66],
        'respiratory_data':    [15.2, 15.8, 14.9, 16.1, 15.5, 16.4, 15.7, 17.8, 15.3, 16.2, 16.8, 17.1, 16.9, 17.4],
        'hrv_data':            [28, 24, 31, 19, 22, 26, 18, 25, 21, 17, 20, 23, 19, 16],
        'sleep_duration_data': sleep,
        'steps_data':          [7200, 6800, 7100, 5900, 6400, 6100, 5800, 4200, 5600, 4800, 4100, 3700, 2900, 2100],
        'temperature_data':    [0.12, -0.08, 0.21, 0.05, 0.18, -0.14, 0.31, 0.93, -0.07, 0.24, 0.15, -0.12, 0.28, 0.19],
        'rem_data':            rem,
        'deep_data':           deep,
        'light_data':          light,
        'awake_data':          awake,
        'time_in_bed_data':    tib,
        'sleep_latency_data':  lat,
        'waso_data':           waso,
        'sleep_efficiency_data': eff,
        'spo2_data':           [96.2, 95.8, 96.5, 95.4, 96.1, 95.9, 95.3, 91.8, 96.4, 95.7, 94.9, 95.5, 94.8, 95.1],
        'sleep_start_data':    ['10:30 PM','10:45 PM','10:15 PM','11:00 PM','10:30 PM',
                                '11:15 PM','10:45 PM','11:00 PM','10:30 PM','11:00 PM',
                                '10:45 PM','11:30 PM','11:00 PM','11:15 PM'],
        'sleep_end_data':      ['6:42 AM','6:27 AM','6:45 AM','6:24 AM','6:42 AM',
                                '7:15 AM','6:51 AM','7:30 AM','6:42 AM','7:00 AM',
                                '7:33 AM','7:42 AM','7:24 AM','7:51 AM'],
    }


def load_patient_data():
    """Load and process patient data from Excel file."""
    try:
        # Try real data first, fall back to demo data
        if os.path.exists('data.xlsx'):
            df = pd.read_excel('data.xlsx')
        else:
            df = pd.read_excel('demo_data.xlsx')
        patients = []
        
        for mrn in df['mrn'].dropna().unique():
            patient_data = df[df['mrn'] == mrn]
            first_row = patient_data.iloc[0]
            
            # Calculate data counts
            inpatient_count = len(patient_data[patient_data['flowsheet_record_date'].notna()])
            outpatient_count = random.randint(10, 70)  # Simulated Oura data count
            
            # Get dates
            admit_date = first_row.get('clarity_admit_date') or first_row.get('admit_date')
            discharge_date = first_row.get('clarity_discharge_date') or first_row.get('inpatient_end_date')
            inpatient_start = first_row.get('inpatient_start_date')
            
            # Last sync
            last_sync = patient_data['flowsheet_entry_datetime'].max() if 'flowsheet_entry_datetime' in patient_data.columns else None
            
            # Check data sources
            has_oura = pd.notna(first_row.get('token'))
            has_ehr = inpatient_count > 0
            
            # Calculate status
            if pd.notna(last_sync):
                try:
                    last_sync_dt = pd.to_datetime(last_sync)
                    days_since = (datetime.now() - last_sync_dt).days
                    if days_since <= 4 and has_oura and has_ehr:
                        status = "active"
                    elif not has_oura or not has_ehr:
                        status = "follow-up"
                    elif days_since > 7:
                        status = "outreach"
                    else:
                        status = "follow-up"
                    last_sync_str = format_last_sync(last_sync_dt)
                except:
                    status = "outreach"
                    last_sync_str = "Unknown"
            else:
                status = "outreach"
                last_sync_str = "Never"
            
            # Generate sparkline data for 3 metrics
            sleep_score = [random.randint(65, 95) for _ in range(12)]
            hrv_average = [random.randint(25, 55) for _ in range(12)]
            activity_score = [random.randint(50, 90) for _ in range(12)]
            
            # Last visit from discharge/admit dates
            _lv = format_date(discharge_date or admit_date)
            last_visit_str = f"{_lv} (In-Person)" if _lv else None

            # Model status: seeded per-patient for stable demo results
            _ms_rng = random.Random(hash(str(mrn)))
            if status == 'active' and has_oura:
                if _ms_rng.random() < 0.4:
                    model_status_level, model_status_label = 'alert', 'Risk Signal Detected'
                else:
                    model_status_level, model_status_label = 'ok',    'Within Normal Range'
            elif status == 'follow-up':
                model_status_level, model_status_label = 'warn',    'Borderline — Review Needed'
            else:
                model_status_level, model_status_label = 'pending', 'Insufficient Data'

            patients.append({
                'id':                  f"PT-{str(int(mrn))[-4:]}",
                'mrn':                 mrn,
                'name':                f"{first_row.get('first_name', '')} {first_row.get('last_name', '')}".strip(),
                'inpatient':           inpatient_count,
                'outpatient':          outpatient_count,
                'last_sync':           last_sync_str,
                'last_visit':          last_visit_str,
                'model_status_level':  model_status_level,
                'model_status_label':  model_status_label,
                'has_oura':            has_oura,
                'has_ehr':             has_ehr,
                'status':              status,
                'risk_level':          '',
                'conditions':          '',
                'participation_start': format_date(inpatient_start or admit_date),
                'hospital_start':      format_date(admit_date),
                'hospital_end':        format_date(discharge_date),
                'sleep_score':         sleep_score,
                'hrv_average':         hrv_average,
                'activity_score':      activity_score,
            })
        
        # Append Synthea clinical patients, then merge matching OMH/IEEE wearable records.
        patients.extend(load_synthea_patients())
        patients.insert(0, _make_demo_he_patient())
        return merge_standard_wearable_patients(patients)
    except Exception as e:
        print(f"Error loading data: {e}")
        # Still try to return adapter-backed patients even if Excel fails.
        patients = load_synthea_patients()
        patients.insert(0, _make_demo_he_patient())
        return merge_standard_wearable_patients(patients)


def format_date(date_val):
    """Format date for display."""
    if pd.isna(date_val):
        return None
    try:
        dt = pd.to_datetime(date_val)
        return dt.strftime("%b %d, %Y")
    except:
        return str(date_val)


def format_last_sync(dt):
    """Format last sync as relative time."""
    delta = datetime.now() - dt
    if delta.days > 0:
        return f"{delta.days} days ago"
    elif delta.seconds > 3600:
        return f"{delta.seconds // 3600} hours ago"
    else:
        return f"{delta.seconds // 60} minutes ago"


def load_study_config() -> dict:
    """Load study config from disk, creating it if missing."""
    if os.path.exists(STUDY_CONFIG_PATH):
        with open(STUDY_CONFIG_PATH) as f:
            return json.load(f)
    return {'conditions_of_interest': []}


def save_study_config(config: dict) -> None:
    config['last_updated'] = datetime.now().strftime('%Y-%m-%d')
    with open(STUDY_CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)


def _compute_cohort_stats(patients: list) -> dict:
    """Aggregate cohort-level statistics for the research dashboard."""
    by_risk: dict = {'high': 0, 'medium': 0, 'low': 0, 'unknown': 0}
    by_source: dict = {}
    condition_counts: dict = {}

    for p in patients:
        risk = (p.get('risk_level') or '').strip().lower()
        by_risk[risk if risk in by_risk else 'unknown'] += 1

        src = p.get('data_source', 'oura')
        by_source[src] = by_source.get(src, 0) + 1

        for cond in (p.get('conditions') or '').split(','):
            cond = cond.strip()
            if cond and cond.lower() != 'none':
                condition_counts[cond] = condition_counts.get(cond, 0) + 1

    total = len(patients)

    # Observed conditions (from patient data), sorted by frequency
    observed = sorted(condition_counts.items(), key=lambda x: -x[1])

    # Study-defined conditions of interest
    study_cfg = load_study_config()
    study_conds = study_cfg.get('conditions_of_interest', [])
    study_names_lower = {c['name'].lower() for c in study_conds}

    # Merge: observed list (with counts) + study-only (count=0, not yet in patient data)
    observed_names_lower = {name.lower() for name, _ in observed}
    all_conditions = [
        {'name': name, 'count': cnt, 'study_tracked': name.lower() in study_names_lower}
        for name, cnt in observed
    ] + [
        {'name': c['name'], 'count': 0, 'study_tracked': True, 'category': c.get('category', '')}
        for c in study_conds
        if c['name'].lower() not in observed_names_lower
    ]

    return {
        'total': total,
        'by_risk': by_risk,
        'by_source': by_source,
        'conditions': observed[:8],          # legacy: list of (name, count) tuples
        'all_conditions': all_conditions,    # merged: list of dicts
        'study_conditions': study_conds,
        'active': sum(1 for p in patients if p.get('status') == 'active'),
    }


def _compute_sleep_summaries(patients: list) -> list:
    """Compute sleep summaries from OMH/IEEE records when available."""
    summaries = []
    for p in patients:
        standard_ts = get_standard_wearable_series(p['id'])
        if standard_ts is not None and not standard_ts.time_series.empty:
            ts = standard_ts.get_analysis_window(14)
            tst = ts['total_sleep_time_hours'] if 'total_sleep_time_hours' in ts else ts.get('sleep_duration_hours')

            # Week-over-week efficiency trend
            if 'sleep_efficiency_pct' in ts and len(ts) >= 8:
                w1 = float(ts['sleep_efficiency_pct'].iloc[:7].mean())
                w2 = float(ts['sleep_efficiency_pct'].iloc[7:].mean())
                eff_delta = round(w2 - w1, 1)
            elif 'sleep_efficiency_pct' in ts:
                eff_delta = 0.0
            else:
                eff_delta = 0.0

            summaries.append({
                'id':               p['id'],
                'risk':             p.get('risk_level', ''),
                'avg_tst':          round(float(tst.mean()), 1) if tst is not None else 0,
                'avg_rem_pct':      round(float(ts['rem_sleep_pct'].mean()), 1) if 'rem_sleep_pct' in ts else 0,
                'avg_deep_pct':     round(float(ts['deep_sleep_pct'].mean()), 1) if 'deep_sleep_pct' in ts else 0,
                'avg_efficiency':   round(float(ts['sleep_efficiency_pct'].mean()), 1) if 'sleep_efficiency_pct' in ts else 0,
                'avg_latency':      int(round(float(ts['sleep_latency'].mean()))) if 'sleep_latency' in ts else 0,
                'avg_waso':         int(round(float(ts['waso_minutes'].mean()))) if 'waso_minutes' in ts else 0,
                'avg_spo2':         round(float(ts['spo2_pct'].mean()), 1) if 'spo2_pct' in ts else 0,
                'efficiency_trend': eff_delta,
                'trend_dir':        'up' if eff_delta > 1.5 else 'down' if eff_delta < -1.5 else 'flat',
            })
            continue

        rng  = random.Random(abs(hash(p['id']))              % (2 ** 31))
        rng2 = random.Random(abs(hash(p['id'] + '_wktrend')) % (2 ** 31))
        avg_tst  = round(rng.uniform(5.2, 8.6), 1)
        avg_rem  = round(rng.uniform(12, 26), 1)
        avg_deep = round(rng.uniform(10, 23), 1)
        avg_eff  = round(rng.uniform(72, 94), 1)
        avg_lat  = int(round(rng.uniform(7, 40)))
        avg_waso = int(round(rng.uniform(12, 58)))
        avg_spo2 = round(rng.uniform(93.2, 98.1), 1)
        eff_delta = round(rng2.uniform(-9.5, 6.0), 1)
        summaries.append({
            'id':               p['id'],
            'risk':             p.get('risk_level', ''),
            'avg_tst':          avg_tst,
            'avg_rem_pct':      avg_rem,
            'avg_deep_pct':     avg_deep,
            'avg_efficiency':   avg_eff,
            'avg_latency':      avg_lat,
            'avg_waso':         avg_waso,
            'avg_spo2':         avg_spo2,
            'efficiency_trend': eff_delta,
            'trend_dir':        'up' if eff_delta > 1.5 else 'down' if eff_delta < -1.5 else 'flat',
        })
    return summaries


def _load_confounders() -> dict:
    if os.path.exists(CONFOUNDERS_PATH):
        with open(CONFOUNDERS_PATH) as f:
            return json.load(f)
    return {}


def _save_confounders(data: dict) -> None:
    os.makedirs(os.path.dirname(CONFOUNDERS_PATH), exist_ok=True)
    with open(CONFOUNDERS_PATH, 'w') as f:
        json.dump(data, f, indent=2)


def _auto_detect_confounders(patient: dict) -> dict:
    """
    Automatically detect nights with physiological patterns suggesting a confounder.
    Thresholds are clinically grounded for the HE sleep biomarker study.
    Returns {iso_date: [reason_strings]}.
    """
    flags: dict[str, list[str]] = {}
    iso_dates   = patient.get('iso_dates', [])
    spo2        = patient.get('spo2_data', [])
    temperature = patient.get('temperature_data', [])
    hrv         = patient.get('hrv_data', [])
    latency     = patient.get('sleep_latency_data', [])
    waso        = patient.get('waso_data', [])

    valid_hrv  = [v for v in hrv if v is not None and v > 0]
    hrv_median = sorted(valid_hrv)[len(valid_hrv) // 2] if valid_hrv else None

    for i, iso in enumerate(iso_dates):
        reasons: list[str] = []

        if i < len(spo2) and spo2[i] is not None and spo2[i] < 93.0:
            reasons.append('Low SpO₂ (possible apnea)')

        if i < len(temperature) and temperature[i] is not None:
            if temperature[i] > 0.8:
                reasons.append('Elevated temp deviation (fever / alcohol)')
            elif temperature[i] < -0.8:
                reasons.append('Low temp deviation')

        if hrv_median and i < len(hrv) and hrv[i] is not None and hrv[i] < hrv_median * 0.5:
            reasons.append('Low HRV (acute stressor)')

        lat_hi  = i < len(latency) and latency[i]  is not None and latency[i]  > 45
        waso_hi = i < len(waso)    and waso[i]     is not None and waso[i]     > 60
        if lat_hi and waso_hi:
            reasons.append('Fragmented sleep (behavioural disruption)')

        if reasons:
            flags[iso] = reasons

    return flags


def _chat_respond(message: str, context: dict, history: list) -> tuple:
    """Rule-based research assistant — swap body for LLM call in Phase 2."""
    msg = message.lower().strip()
    patient_id = context.get('patient_id', '')

    all_patients = load_patient_data()
    patient = next((p for p in all_patients if p['id'] == patient_id), None) if patient_id else None
    stats = _compute_cohort_stats(all_patients)

    def pct(n):
        return round(100 * n / max(stats['total'], 1))

    # ── Cohort size / overview ───────────────────────────────────────────────
    if any(w in msg for w in ['how many', 'total patients', 'cohort size', 'n=', 'participants', 'enrolled']):
        br = stats['by_risk']
        src = ', '.join(f"{v} {k}" for k, v in stats['by_source'].items())
        resp = (f"The cohort has **{stats['total']} participants** ({src}). "
                f"{br['high']} ({pct(br['high'])}%) are high risk, "
                f"{br['medium']} ({pct(br['medium'])}%) medium, "
                f"{br['low']} ({pct(br['low'])}%) low.")
        return resp, ['Show risk breakdown', 'What conditions are most common?', 'Compare data sources']

    # ── Risk breakdown ───────────────────────────────────────────────────────
    if any(w in msg for w in ['risk breakdown', 'risk distribution', 'risk split', 'risk level']):
        br = stats['by_risk']
        resp = (f"Risk distribution across **{stats['total']} participants**: "
                f"**{br['high']} high** ({pct(br['high'])}%), "
                f"**{br['medium']} medium** ({pct(br['medium'])}%), "
                f"**{br['low']} low** ({pct(br['low'])}%). "
                f"High risk is flagged by glucose > 200 mg/dL, HbA1c > 9%, or systolic BP > 160 mmHg.")
        return resp, ['What drives high risk?', 'Show high-risk patients', 'What is HbA1c?']

    # ── Conditions ───────────────────────────────────────────────────────────
    if any(w in msg for w in ['condition', 'diagnosis', 'diagnoses', 'disease', 'prevalent']):
        if stats['conditions']:
            top = ', '.join(f"{n} with {c}" for c, n in stats['conditions'][:5])
            resp = f"Most prevalent conditions in the cohort: {top}."
        else:
            resp = "Condition data is sourced from Synthea FHIR Condition resources and Oura patient metadata."
        return resp, ['What is the risk for diabetes patients?', 'What is hepatic encephalopathy?']

    # ── Data sources ─────────────────────────────────────────────────────────
    if any(w in msg for w in ['data source', 'synthea', 'oura', 'fhir', 'wearable', 'compare source']):
        resp = ("This workbench fuses two data layers: **Synthea FHIR** (synthetic EHR — metabolic labs, "
                "diagnoses, clinical vitals) and **Oura Ring** (wearable — sleep architecture, HRV, SpO2, activity). "
                "Synthea is the clinical ground truth; Oura provides continuous longitudinal signals.")
        return resp, ['What features does Synthea provide?', 'What does Oura measure?', 'How many from each source?']

    # ── Current patient (context-aware) ─────────────────────────────────────
    if patient and any(w in msg for w in ['this patient', 'participant', 'their risk', 'vitals', 'profile']):
        risk = patient.get('risk_level', 'unknown')
        conds = patient.get('conditions', 'None')
        src = patient.get('data_source', 'unknown')
        resp = (f"**{patient_id}** — {risk} risk · {src} data source. "
                f"Conditions: {conds}. Last sync: {patient.get('last_sync', 'unknown')}.")
        return resp, ['What are their key risk factors?', 'What is HbA1c?', 'View data explorer']

    # ── Clinical definitions ─────────────────────────────────────────────────
    if 'hba1c' in msg or 'hemoglobin a1c' in msg or 'a1c' in msg:
        resp = ("**HbA1c** reflects average blood glucose over 2–3 months. "
                "Normal < 5.7% · Pre-diabetes 5.7–6.4% · Diabetes ≥ 6.5%. "
                "Values > 9% indicate poor glycaemic control and drive high-risk classification here.")
        return resp, ['What is blood glucose?', 'How is risk calculated?']

    if any(w in msg for w in ['glucose', 'blood sugar', 'blood glucose']):
        resp = ("**Blood glucose** (mg/dL): fasting normal 70–99 · pre-diabetes 100–125 · diabetes ≥ 126. "
                "In this cohort glucose > 200 mg/dL triggers a high-risk flag.")
        return resp, ['What is HbA1c?', 'Show metabolic trends']

    if any(w in msg for w in ['hrv', 'heart rate variability', 'rmssd']):
        resp = ("**HRV (Heart Rate Variability)** — milliseconds between heartbeats. Higher HRV = better "
                "autonomic function. In HE research, declining HRV during sleep is an early marker of "
                "covert cognitive impairment, preceding clinical detection by weeks.")
        return resp, ['What is hepatic encephalopathy?', 'What is REM sleep?']

    if any(w in msg for w in ['rem', 'deep sleep', 'sleep architecture', 'waso', 'sleep stage']):
        resp = ("**Sleep architecture** spans REM, deep (slow-wave), and light stages. "
                "Reduced REM % and elevated WASO (Wake After Sleep Onset) are the primary wearable biomarkers "
                "for covert hepatic encephalopathy in this study — they precede measurable ammonia elevation.")
        return resp, ['What is hepatic encephalopathy?', 'What is HRV?', 'How is sleep scored?']

    if any(w in msg for w in ['hepatic encephalopathy', 'liver', 'cirrhosis', 'meld', 'ammonia', 'he ']):
        resp = ("**Hepatic encephalopathy (HE)** — cognitive dysfunction from liver failure to clear toxins. "
                "Covert HE affects ~30% of cirrhosis patients and is routinely missed. "
                "This study tests whether Oura wearable signals (REM %, HRV, circadian rhythm) can detect "
                "it earlier than pen-and-paper cognitive tests. MELD-Na score tracks disease severity.")
        return resp, ['What biomarkers predict HE?', 'What is MELD?', 'Show HRV data']

    if any(w in msg for w in ['spo2', 'oxygen', 'saturation']):
        resp = ("**SpO2** (blood oxygen saturation) is measured nightly by the Oura ring. "
                "Normal ≥ 95%. Nocturnal desaturation can indicate sleep apnea — a key confounder "
                "in HE research that must be controlled for in model training.")
        return resp, ['What is WASO?', 'What confounders matter?']

    if any(w in msg for w in ['model', 'xgboost', 'random forest', 'lstm', 'prediction', 'auc', 'shap', 'feature importance']):
        resp = ("The Model Lab supports **XGBoost**, **Random Forest**, and **LSTM** classifiers, "
                "evaluated by AUC-ROC, precision, recall, and F1. SHAP values explain feature contributions. "
                "Best demo result: XGBoost AUC 0.91 on 5 sleep + HRV features.")
        return resp, ['What features matter most?', 'How does SHAP work?', 'Run an experiment']

    if any(w in msg for w in ['what can you', 'help', 'capabilities', 'what do you know']):
        resp = ("I'm your **Research Assistant**. I can help with:\n"
                "- Cohort statistics (size, risk, conditions, sources)\n"
                "- Clinical definitions (HbA1c, HRV, REM, WASO, MELD, SpO2)\n"
                "- Study background (hepatic encephalopathy, design, goals)\n"
                "- Model interpretation (SHAP, AUC, feature importance)\n"
                "- Patient-specific context when you're viewing a participant profile")
        return resp, ['How many patients are enrolled?', 'What is HbA1c?', 'Explain the study']

    if any(w in msg for w in ['study', 'research', 'about this', 'purpose', 'goal', 'what is this']):
        resp = ("This workbench supports a **Cornell Medicine clinical research study** — "
                "can Oura Ring wearable biomarkers (sleep architecture, HRV, circadian rhythm) "
                "detect covert hepatic encephalopathy earlier than current clinical methods? "
                "~140 cirrhosis patients, 6+ years of data. Synthea FHIR provides a parallel "
                "synthetic metabolic cohort for algorithm development.")
        return resp, ['What is hepatic encephalopathy?', 'How many patients?', 'What models are used?']

    # ── Fallback ─────────────────────────────────────────────────────────────
    resp = ("I can help you explore the cohort, understand clinical biomarkers, or interpret model results. "
            "Try asking about the cohort size, specific biomarkers (HbA1c, HRV, REM sleep), "
            "the study background, or a specific participant if you're viewing their profile.")
    return resp, ['How many patients are enrolled?', 'What is HbA1c?', 'Explain the study']


@app.route('/')
def dashboard():
    patients = load_patient_data()
    cohort_stats    = _compute_cohort_stats(patients)
    sleep_summaries = _compute_sleep_summaries(patients)
    sleep_summary_map = {s['id']: s for s in sleep_summaries}
    study_cfg = load_study_config()
    enrollment = study_cfg.get('enrollment', {
        'enrolled': 140, 'target': 150,
        'days_of_data': '20,000+', 'years_of_data': 6,
        'institutions': 'Cornell Medicine',
    })
    return render_template('dashboard.html', patients=patients, cohort_stats=cohort_stats,
                           sleep_summaries=sleep_summaries, sleep_summary_map=sleep_summary_map,
                           enrollment=enrollment)



@app.route('/api/patients')
def get_patients():
    patients = load_patient_data()
    return jsonify(patients)


@app.route('/api/conditions/search')
def conditions_search():
    """Search the bundled clinical conditions dictionary."""
    q = request.args.get('q', '').strip().lower()
    if len(q) < 2:
        return jsonify([])

    # Load dictionary
    try:
        with open(CONDITIONS_DICT_PATH) as f:
            dictionary = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return jsonify([])

    # Cohort condition counts for annotation
    patients = load_patient_data()
    stats = _compute_cohort_stats(patients)
    cohort_map = {name.lower(): cnt for name, cnt in stats['conditions']}

    # Study-tracked names for annotation
    study_cfg = load_study_config()
    tracked = {c['name'].lower() for c in study_cfg.get('conditions_of_interest', [])}

    # Filter dictionary
    matches = [c for c in dictionary if q in c['name'].lower()]
    for m in matches:
        m['cohort_count']   = cohort_map.get(m['name'].lower(), 0)
        m['study_tracked']  = m['name'].lower() in tracked

    # Sort: cohort matches first, then alphabetical
    matches.sort(key=lambda x: (-x['cohort_count'], x['name']))
    return jsonify(matches[:12])


@app.route('/api/study/conditions', methods=['POST'])
def add_study_condition():
    """Add a condition to the study conditions of interest."""
    data = request.get_json(silent=True) or {}
    name = (data.get('name') or '').strip()
    if not name:
        return jsonify({'error': 'name is required'}), 400

    config = load_study_config()
    existing = {c['name'].lower() for c in config.get('conditions_of_interest', [])}
    if name.lower() in existing:
        return jsonify({'error': 'already tracked'}), 409

    config.setdefault('conditions_of_interest', []).append({
        'name':     name,
        'code':     data.get('code', ''),
        'system':   data.get('system', 'custom'),
        'category': data.get('category', ''),
        'added_date': datetime.now().strftime('%Y-%m-%d'),
    })
    save_study_config(config)
    return jsonify({'ok': True, 'name': name})


@app.route('/api/study/conditions/<path:name>', methods=['DELETE'])
def remove_study_condition(name):
    """Remove a condition from the study conditions of interest."""
    config = load_study_config()
    before = len(config.get('conditions_of_interest', []))
    config['conditions_of_interest'] = [
        c for c in config.get('conditions_of_interest', [])
        if c['name'].lower() != name.lower()
    ]
    if len(config['conditions_of_interest']) == before:
        return jsonify({'error': 'not found'}), 404
    save_study_config(config)
    return jsonify({'ok': True})


@app.route('/api/run-experiment', methods=['POST'])
def api_run_experiment():
    """Train a model on synthetic patient data and return results as JSON."""
    # 1. Parse + validate body
    body = request.get_json(silent=True)
    if not body:
        return jsonify({'error': 'Request body must be JSON.'}), 400

    patient_id           = str(body.get('patient_id', '')).strip()
    model_type           = str(body.get('model_type', '')).strip()
    features             = body.get('features', [])
    hyperparameters      = body.get('hyperparameters', {})
    analysis_window_days = int(body.get('analysis_window_days', 30))
    excluded_dates       = body.get('excluded_dates', [])

    if not patient_id:
        return jsonify({'error': 'patient_id is required.'}), 400
    if not model_type:
        return jsonify({'error': 'model_type is required.'}), 400
    if not isinstance(features, list) or len(features) == 0:
        return jsonify({'error': 'Select at least one feature.'}), 400

    # 2. Verify patient exists
    patients = load_patient_data()
    patient  = next((p for p in patients if p['id'] == patient_id), None)
    if not patient:
        return jsonify({'error': f'Patient {patient_id!r} not found.'}), 404

    # 3. Build a synthetic PatientTimeSeries (real adapters wired in Phase 2)
    #    Seeded per-patient so results are stable across repeated calls.
    try:
        import numpy as np
        import pandas as pd
        from data.base import PatientTimeSeries, DataSource

        seed   = abs(hash(patient_id)) % (2 ** 31)
        rng    = np.random.default_rng(seed)
        n_rows = max(60, analysis_window_days if analysis_window_days > 0 else 60)
        idx    = pd.date_range(end='2024-12-31', periods=n_rows, freq='D')
        ds     = DataSource.SYNTHEA if patient_id.startswith('PT-3') else DataSource.OURA

        col_data = {
            # Oura Ring features
            'rem_sleep_pct':       rng.uniform(10, 30, n_rows),
            'deep_sleep_pct':      rng.uniform(8,  25, n_rows),
            'sleep_latency':       rng.uniform(5,  45, n_rows),
            'hrv_balance':         rng.uniform(20, 80, n_rows),
            'body_temp_deviation': rng.uniform(-1.0, 1.0, n_rows),
            'resting_hr':          rng.uniform(50, 75, n_rows),
            'step_count':          rng.uniform(2000, 15000, n_rows),
            'inactivity_alerts':   rng.uniform(0, 10, n_rows),
            # Synthea features
            'heart_rate':             rng.uniform(58, 88, n_rows),
            'systolic_bp':            rng.uniform(108, 175, n_rows),
            'diastolic_bp':           rng.uniform(65, 110, n_rows),
            'respiratory_rate':       rng.uniform(13, 18, n_rows),
            'body_temperature':       rng.uniform(36.4, 37.2, n_rows),
            'body_weight_kg':         rng.uniform(60, 120, n_rows),
            'bmi':                    rng.uniform(20, 40, n_rows),
            'glucose_mgdl':           rng.uniform(75, 250, n_rows),
            'hba1c_pct':              rng.uniform(4.8, 11.0, n_rows),
            'total_cholesterol_mgdl': rng.uniform(150, 270, n_rows),
            'ldl_cholesterol_mgdl':   rng.uniform(80, 185, n_rows),
        }

        patient_ts = PatientTimeSeries(
            patient_id=patient_id,
            data_source=ds,
            time_series=pd.DataFrame(col_data, index=idx),
        )

    except Exception as exc:
        return jsonify({'error': f'Failed to build patient data: {exc}'}), 500

    # 4. Run experiment
    try:
        from models.experiment import ExperimentConfig, run_experiment

        config = ExperimentConfig(
            model_type=model_type,
            features=features,
            hyperparameters=hyperparameters,
            analysis_window_days=analysis_window_days,
            patient_id=patient_id,
        )
        result = run_experiment(config, patient_ts)

    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except ImportError as exc:
        return jsonify({'error': str(exc)}), 500
    except Exception as exc:
        return jsonify({'error': f'Experiment failed: {exc}'}), 500

    return jsonify({
        'metrics':               result.metrics,
        'feature_importance':    result.feature_importance,
        'prediction_confidence': result.prediction_confidence,
        'trained_at':            result.trained_at.isoformat(),
        'excluded_nights':       len(excluded_dates),
    })


@app.route('/api/generate-patient', methods=['POST'])
def api_generate_patient():
    """Generate a new synthetic Synthea patient, save a FHIR bundle, return patient dict."""
    synthea_dir = 'demo_data/demo_synthea'
    os.makedirs(synthea_dir, exist_ok=True)

    # Next available PT-3xxx ID
    existing_nums = []
    for fname in os.listdir(synthea_dir):
        m = re.match(r'^PT-3(\d+)\.json$', fname)
        if m:
            existing_nums.append(int(m.group(1)))
    next_num = max(existing_nums, default=5) + 1
    patient_id = f'PT-3{next_num:03d}'

    risk_level = random.choices(['high', 'medium', 'low'], weights=[0.3, 0.4, 0.3])[0]

    try:
        from data.synthea_adapter import SyntheaAdapter
        pts = SyntheaAdapter().load_demo_data(patient_id, risk_level)
    except Exception as e:
        return jsonify({'error': f'Failed to generate patient data: {e}'}), 500

    ts = pts.time_series
    sf = pts.static_features

    # ── Build minimal FHIR R4 Bundle ─────────────────────────────────────────
    fhir_id    = str(uuid.uuid5(uuid.NAMESPACE_DNS, patient_id))
    birth_year = datetime.now().year - sf.get('age', 50)
    gender     = 'male' if sf.get('sex') == 'M' else 'female'

    def _loinc(code, display):
        return {'coding': [{'system': 'http://loinc.org', 'code': code, 'display': display}],
                'text': display}

    def _obs(pat_ref, code, display, value, unit, unit_code, date_str):
        return {'resourceType': 'Observation', 'status': 'final',
                'code': _loinc(code, display),
                'subject': {'reference': f'urn:uuid:{pat_ref}'},
                'effectiveDateTime': date_str,
                'valueQuantity': {'value': value, 'unit': unit,
                                  'system': 'http://unitsofmeasure.org', 'code': unit_code}}

    def _bp_obs(pat_ref, sbp, dbp, date_str):
        return {'resourceType': 'Observation', 'status': 'final',
                'code': _loinc('55284-4', 'Blood pressure systolic and diastolic'),
                'subject': {'reference': f'urn:uuid:{pat_ref}'},
                'effectiveDateTime': date_str,
                'component': [
                    {'code': _loinc('8480-6', 'Systolic blood pressure'),
                     'valueQuantity': {'value': sbp, 'unit': 'mmHg',
                                       'system': 'http://unitsofmeasure.org', 'code': 'mm[Hg]'}},
                    {'code': _loinc('8462-4', 'Diastolic blood pressure'),
                     'valueQuantity': {'value': dbp, 'unit': 'mmHg',
                                       'system': 'http://unitsofmeasure.org', 'code': 'mm[Hg]'}},
                ]}

    def _cond(pat_ref, snomed_code, display):
        return {'resourceType': 'Condition',
                'clinicalStatus': {'coding': [{'system': 'http://terminology.hl7.org/CodeSystem/condition-clinical',
                                               'code': 'active'}]},
                'code': {'coding': [{'system': 'http://snomed.info/sct',
                                     'code': snomed_code, 'display': display}], 'text': display},
                'subject': {'reference': f'urn:uuid:{pat_ref}'}}

    _CONDITIONS_BY_RISK = {
        'high':   [('44054006', 'Diabetes mellitus type 2'), ('38341003', 'Hypertension')],
        'medium': [('38341003', 'Hypertension')],
        'low':    [],
    }
    _OBS_MAP = [
        ('8867-4',  'Heart rate',        'heart_rate',             'bpm',    '/min'),
        ('9279-1',  'Respiratory rate',  'respiratory_rate',       'br/min', '/min'),
        ('8310-5',  'Body temperature',  'body_temperature',       '°C',     'Cel'),
        ('29463-7', 'Body weight',       'body_weight_kg',         'kg',     'kg'),
        ('39156-5', 'Body mass index',   'bmi',                    'kg/m2',  'kg/m2'),
        ('2339-0',  'Glucose',           'glucose_mgdl',           'mg/dL',  'mg/dL'),
        ('4548-4',  'Hemoglobin A1c',    'hba1c_pct',              '%',      '%'),
        ('2093-3',  'Total cholesterol', 'total_cholesterol_mgdl', 'mg/dL',  'mg/dL'),
        ('18262-6', 'LDL cholesterol',   'ldl_cholesterol_mgdl',   'mg/dL',  'mg/dL'),
    ]

    entries = [{'fullUrl': f'urn:uuid:{fhir_id}', 'resource': {
        'resourceType': 'Patient', 'id': fhir_id,
        'name': [{'use': 'official', 'family': f'Demo-{patient_id}',
                  'given': [risk_level.capitalize()]}],
        'gender': gender,
        'birthDate': f'{birth_year}-06-15',
    }}]

    for obs_date, row in ts.iterrows():
        date_str = pd.Timestamp(obs_date).strftime('%Y-%m-%dT00:00:00+00:00')
        entries.append({'resource': _bp_obs(
            fhir_id,
            round(float(row['systolic_bp']), 2),
            round(float(row['diastolic_bp']), 2),
            date_str,
        )})
        for loinc, display, feat, unit, unit_code in _OBS_MAP:
            if feat in row and not (isinstance(row[feat], float) and pd.isna(row[feat])):
                entries.append({'resource': _obs(
                    fhir_id, loinc, display,
                    round(float(row[feat]), 2), unit, unit_code, date_str,
                )})

    for snomed, display in _CONDITIONS_BY_RISK.get(risk_level, []):
        entries.append({'resource': _cond(fhir_id, snomed, display)})

    bundle = {'resourceType': 'Bundle', 'type': 'collection', 'entry': entries}

    out_path = os.path.join(synthea_dir, f'{patient_id}.json')
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(bundle, f, indent=2)
    except Exception as e:
        return jsonify({'error': f'Failed to save FHIR bundle: {e}'}), 500

    # ── Build response patient dict (mirrors load_synthea_patients logic) ────
    last_date     = ts.index.max() if not ts.empty else None
    last_dt       = last_date.to_pydatetime().replace(tzinfo=None) if last_date else None
    last_sync_str = format_last_sync(last_dt) if last_dt else 'Never'
    days_since    = (datetime.now() - last_dt).days if last_dt else 999

    if days_since <= 4:
        status = 'active'
    elif days_since <= 30:
        status = 'follow-up'
    else:
        status = 'outreach'

    _risk_map = {
        'high':   ('alert', 'Risk Signal Detected'),
        'medium': ('warn',  'Borderline Range'),
        'low':    ('ok',    'Within Normal Range'),
    }
    model_status_level, model_status_label = _risk_map.get(risk_level, ('pending', 'Insufficient Data'))

    condition_str = ', '.join(
        k.replace('_', ' ').title() for k, v in sf.items() if isinstance(v, bool) and v
    ) or 'None'

    return jsonify({
        'patient': {
            'id':                 patient_id,
            'name':               sf.get('name', patient_id),
            'risk_level':         risk_level,
            'conditions':         condition_str,
            'status':             status,
            'last_sync':          last_sync_str,
            'model_status_level': model_status_level,
            'model_status_label': model_status_label,
            'data_source':        'synthea',
        },
        'message': f'Patient {patient_id} generated ({risk_level} risk).',
    })


@app.route('/api/chat', methods=['POST'])
def api_chat():
    """Research assistant chatbot endpoint."""
    body = request.get_json(silent=True) or {}
    message = str(body.get('message', '')).strip()
    context = body.get('context', {})
    history = body.get('history', [])
    if not message:
        return jsonify({'error': 'message is required'}), 400
    try:
        response, suggestions = _chat_respond(message, context, history)
        return jsonify({'response': response, 'suggestions': suggestions})
    except Exception:
        return jsonify({'response': 'Sorry, I encountered an error. Please try again.', 'suggestions': []}), 200


def _compute_patient_view_data(patient_id: str, patients: list | None = None) -> dict | None:
    """
    Shared computation for patient_detail and patient_report.
    Returns a dict with patient (with time-series attached), clinical_history,
    sleep_rows, confounders, and summary averages. Returns None if not found.
    """
    if patients is None:
        patients = load_patient_data()
    patient = next((p for p in patients if p['id'] == patient_id), None)
    if patient is None:
        return None

    # Isolated per-patient RNG — never touches global random state
    rng = random.Random(abs(hash(patient_id)) % (2 ** 31))

    base_date = datetime(2024, 12, 1)
    # Store both display label and ISO key for each day
    dates      = [(base_date + timedelta(days=i)).strftime("%b %d")  for i in range(14)]
    iso_dates  = [(base_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(14)]

    heart_rate_data    = [rng.randint(48, 72)              for _ in range(14)]
    respiratory_data   = [round(rng.uniform(12, 18), 1)    for _ in range(14)]
    hrv_data           = [rng.randint(20, 80)              for _ in range(14)]
    sleep_duration_data= [round(rng.uniform(5, 9), 1)      for _ in range(14)]
    steps_data         = [rng.randint(2000, 15000)          for _ in range(14)]
    temperature_data   = [round(rng.uniform(-1.0, 1.0), 2) for _ in range(14)]

    rem_data, deep_data, light_data, awake_data      = [], [], [], []
    time_in_bed_data, sleep_latency_data, waso_data  = [], [], []
    sleep_efficiency_data, spo2_data                 = [], []
    sleep_start_data, sleep_end_data                 = [], []

    for i in range(14):
        total     = sleep_duration_data[i]
        rem       = round(max(0.5, rng.uniform(0.15, 0.25) * total + rng.uniform(-0.2, 0.2)), 1)
        deep      = round(max(0.3, rng.uniform(0.13, 0.22) * total + rng.uniform(-0.15, 0.15)), 1)
        awake     = round(rng.uniform(0.1, 0.6), 1)
        light     = round(max(0.4, total - rem - deep), 1)
        lat       = rng.randint(5, 38)
        w         = rng.randint(8, 55)
        tib       = round(total + awake + lat / 60 + w / 60, 1)
        eff       = round((total / tib * 100) if tib > 0 else 0, 1)
        sp        = round(rng.uniform(93.0, 98.5), 1)
        start_hr  = rng.randint(21, 23)
        start_min = rng.choice([0, 15, 30, 45])
        end_total = start_hr * 60 + start_min + int(tib * 60)
        end_hr, end_min = (end_total // 60) % 24, end_total % 60

        rem_data.append(rem);   deep_data.append(deep)
        light_data.append(light); awake_data.append(awake)
        time_in_bed_data.append(tib);  sleep_latency_data.append(lat)
        waso_data.append(w);    sleep_efficiency_data.append(eff)
        spo2_data.append(sp)
        sleep_start_data.append(f"{start_hr % 12 or 12}:{start_min:02d} {'PM' if start_hr >= 12 else 'AM'}")
        sleep_end_data.append(f"{end_hr % 12 or 12}:{end_min:02d} {'AM' if end_hr < 12 else 'PM'}")

    # Override with real OMH/IEEE data when available
    standard_wearable = get_standard_wearable_series(patient_id)
    if standard_wearable is not None and not standard_wearable.time_series.empty:
        ts = standard_wearable.get_analysis_window(14).copy()
        if not ts.empty:
            dates     = [pd.Timestamp(d).strftime("%b %d")   for d in ts.index]
            iso_dates = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in ts.index]

            def _v(col: str, default=None, digits: int | None = 1):
                if col not in ts:
                    return [default] * len(ts)
                return [
                    (int(round(float(x))) if digits is None else round(float(x), digits))
                    if not pd.isna(x) else default
                    for x in ts[col].tolist()
                ]

            heart_rate_data     = _v('resting_hr',          default=0, digits=None)
            respiratory_data    = _v('respiratory_rate',     default=0)
            sleep_duration_data = _v('sleep_duration_hours', default=0)
            steps_data          = _v('step_count',           default=0, digits=None)
            time_in_bed_data    = _v('time_in_bed_hours',    default=0)
            sleep_latency_data  = _v('sleep_latency',        default=0, digits=None)
            waso_data           = _v('waso_minutes',         default=0, digits=None)
            sleep_efficiency_data = _v('sleep_efficiency_pct', default=0)
            spo2_data           = _v('spo2_pct',             default=0)
            rem_data            = _v('rem_sleep_hours',      default=0)
            deep_data           = _v('deep_sleep_hours',     default=0)
            light_data          = _v('light_sleep_hours',    default=0)
            awake_data          = _v('awake_hours',          default=0)
            sleep_start_data    = (ts['sleep_start_label'].fillna('').tolist()
                                   if 'sleep_start_label' in ts else ['']*len(ts))
            sleep_end_data      = (ts['sleep_end_label'].fillna('').tolist()
                                   if 'sleep_end_label' in ts else ['']*len(ts))

    # Attach all series to patient dict (template reads from here)
    patient.update({
        'dates':                dates,
        'iso_dates':            iso_dates,
        'heart_rate_data':      heart_rate_data,
        'respiratory_data':     respiratory_data,
        'hrv_data':             hrv_data,
        'sleep_duration_data':  sleep_duration_data,
        'steps_data':           steps_data,
        'temperature_data':     temperature_data,
        'rem_data':             rem_data,
        'deep_data':            deep_data,
        'light_data':           light_data,
        'awake_data':           awake_data,
        'time_in_bed_data':     time_in_bed_data,
        'sleep_latency_data':   sleep_latency_data,
        'waso_data':            waso_data,
        'sleep_efficiency_data': sleep_efficiency_data,
        'spo2_data':            spo2_data,
        'sleep_start_data':     sleep_start_data,
        'sleep_end_data':       sleep_end_data,
    })

    # Demo patient: replace seeded/OMH data with hand-crafted HE clinical story
    if patient_id == DEMO_HE_PATIENT_ID:
        patient.update(_demo_he_overrides())

    # ── Clinical history ──────────────────────────────────────────────────
    clinical_history = None
    ds_key   = 'synthea' if patient_id.startswith('PT-3') else 'oura'
    _ch_rng  = random.Random(abs(hash(patient_id + '_ch')) % (2 ** 31))

    if ds_key == 'oura':
        try:
            age       = _ch_rng.randint(45, 72)
            sex       = _ch_rng.choice(['Male', 'Female'])
            meld_6mo  = _ch_rng.randint(9, 26)
            meld_now  = meld_6mo + _ch_rng.randint(-1, 5)
            meld_delta= round(meld_now - meld_6mo, 1)
            ammonia   = _ch_rng.randint(28, 110)
            bilirubin = round(_ch_rng.uniform(1.2, 9.0), 1)
            inr       = round(_ch_rng.uniform(1.2, 2.4), 1)
            creatinine= round(_ch_rng.uniform(0.7, 2.3), 1)
            sodium    = _ch_rng.randint(128, 140)
            conditions = ['Cirrhosis (Child-Pugh B)' if meld_now >= 15 else 'Cirrhosis (Child-Pugh A)']
            if ammonia > 60 or patient.get('status') in ('follow-up', 'outreach'):
                conditions.append('Covert Hepatic Encephalopathy')
            if inr > 1.8:
                conditions.append('Coagulopathy')
            if sodium < 133:
                conditions.append('Hyponatremia')
            if patient.get('has_oura'):
                conditions.append('Active Wearable Monitoring')
            enroll_dt = datetime(2024, 12, 31) - timedelta(days=_ch_rng.randint(300, 900))
            clinical_history = {
                'age': age, 'sex': sex,
                'conditions':      conditions,
                'enrollment_date': enroll_dt.strftime('%b %d, %Y'),
                'latest_vitals': {
                    'MELD-Na Score':  {'value': meld_now,   'unit': 'score'},
                    'Ammonia (NH₃)':  {'value': ammonia,    'unit': 'μmol/L'},
                    'Bilirubin':      {'value': bilirubin,  'unit': 'mg/dL'},
                    'INR':            {'value': inr,        'unit': ''},
                    'Creatinine':     {'value': creatinine, 'unit': 'mg/dL'},
                    'Serum Sodium':   {'value': sodium,     'unit': 'mEq/L'},
                    'Resting HR':     {'value': heart_rate_data[-1], 'unit': 'bpm'},
                    'HRV (rMSSD)':    {'value': hrv_data[-1],        'unit': 'ms'},
                },
                'trends': {
                    'MELD-Na': meld_delta,
                    'Ammonia': round(ammonia - _ch_rng.randint(20, 40), 1),
                },
                'data_points':       patient.get('outpatient', 0),
                'data_source_label': (
                    standard_wearable.metadata.get('data_source_label')
                    if standard_wearable is not None else 'Oura Ring V2 + EHR Flowsheets'
                ),
                'last_encounter': 'Dec 01, 2024',
            }
        except Exception as e:
            print(f"[patient_view] Oura clinical history error for {patient_id}: {e}")

    elif ds_key == 'synthea':
        try:
            from data.synthea_adapter import SyntheaAdapter
            adapter = SyntheaAdapter()
            fpath   = os.path.join('demo_data/demo_synthea', f"{patient_id}.json")
            pts     = (adapter.load_from_fhir(fpath, patient_id)
                       if os.path.isfile(fpath)
                       else adapter.load_demo_data(patient_id, patient.get('risk_level', 'medium')))
            sf, ts  = pts.static_features, pts.time_series
            conditions = [
                k.replace('_', ' ').title() for k, v in sf.items() if isinstance(v, bool) and v
            ]
            latest_vitals = {}
            vital_map = {
                'glucose_mgdl':   ('Blood Glucose', 'mg/dL'),
                'hba1c_pct':      ('HbA1c',         '%'),
                'systolic_bp':    ('Systolic BP',    'mmHg'),
                'diastolic_bp':   ('Diastolic BP',   'mmHg'),
                'bmi':            ('BMI',            'kg/m²'),
                'body_weight_kg': ('Weight',         'kg'),
                'heart_rate':     ('Heart Rate',     'bpm'),
            }
            if not ts.empty:
                row = ts.iloc[-1]
                for col, (label, unit) in vital_map.items():
                    if col in row.index and not pd.isna(row[col]):
                        latest_vitals[label] = {'value': round(float(row[col]), 1), 'unit': unit}
            trends = {}
            for col, label in [('glucose_mgdl', 'Glucose'), ('hba1c_pct', 'HbA1c'), ('systolic_bp', 'Systolic BP')]:
                if col in ts.columns and len(ts) >= 2:
                    fv, lv = ts[col].iloc[0], ts[col].iloc[-1]
                    if not pd.isna(fv) and not pd.isna(lv):
                        trends[label] = round(lv - fv, 1)
            enroll_str = pts.metadata.get('enrollment_date', '')
            try:
                enroll_str = datetime.strptime(enroll_str, '%Y-%m-%d').strftime('%b %d, %Y')
            except ValueError:
                pass
            clinical_history = {
                'age': sf.get('age'),
                'sex': 'Male' if sf.get('sex') == 'M' else 'Female' if sf.get('sex') == 'F' else None,
                'conditions':      conditions,
                'enrollment_date': enroll_str,
                'latest_vitals':   latest_vitals,
                'trends':          trends,
                'data_points':     pts.metadata.get('data_points_count', len(ts)),
                'data_source_label': pts.metadata.get('data_source_label', 'Synthea FHIR'),
                'last_encounter':  ts.index[-1].strftime('%b %d, %Y') if not ts.empty else None,
            }
        except Exception as e:
            print(f"[patient_view] Synthea clinical history error for {patient_id}: {e}")

    # ── Confounders & sleep rows ──────────────────────────────────────────
    confounders = _load_confounders().get(patient_id, {})
    sleep_rows  = []
    for i in range(len(dates)):
        tst   = sleep_duration_data[i]
        rem_h = rem_data[i]
        dp_h  = deep_data[i]
        # Support both legacy "%b %d" keys and new ISO keys
        flags = confounders.get(iso_dates[i], confounders.get(dates[i], []))
        sleep_rows.append({
            'date':             dates[i],
            'iso_date':         iso_dates[i],
            'tst':              tst,
            'rem_pct':          round(rem_h / max(tst, 0.01) * 100, 1),
            'deep_pct':         round(dp_h  / max(tst, 0.01) * 100, 1),
            'efficiency':       sleep_efficiency_data[i],
            'latency':          sleep_latency_data[i],
            'waso':             waso_data[i],
            'spo2':             spo2_data[i],
            'flagged':          bool(flags),
            'confounder_labels': flags,
        })

    n = len(sleep_efficiency_data)
    avg_eff  = round(sum(sleep_efficiency_data) / n, 1) if n else 0
    avg_rem  = round(sum(r / max(s, 0.01) * 100 for r, s in zip(rem_data, sleep_duration_data)) / n, 1) if n else 0
    avg_waso = int(round(sum(waso_data) / n))           if n else 0
    avg_spo2 = round(sum(spo2_data) / n, 1)             if n else 0

    return {
        'patient':           patient,
        'clinical_history':  clinical_history,
        'sleep_rows':        sleep_rows,
        'confounders':       confounders,
        'avg_eff':           avg_eff,
        'avg_rem':           avg_rem,
        'avg_waso':          avg_waso,
        'avg_spo2':          avg_spo2,
        'standard_wearable': standard_wearable,
        'auto_confounders':  _auto_detect_confounders(patient),
    }


@app.route('/patient/<patient_id>')
def patient_detail(patient_id):
    view = _compute_patient_view_data(patient_id)
    if view is None:
        return "Patient not found", 404
    return render_template('patient_detail.html',
                           patient=view['patient'],
                           active_tab='overview',
                           clinical_history=view['clinical_history'])


@app.route('/patient/<patient_id>/risk-analysis-lab')
def risk_analysis_lab(patient_id):
    """Show the Risk Analysis Lab — ML model selection, training, and results."""
    view = _compute_patient_view_data(patient_id)
    if not view:
        return "Patient not found", 404

    patient = view['patient']

    from datetime import datetime as _dt
    flagged_nights = []
    for iso, reasons in sorted(view['auto_confounders'].items()):
        try:
            display = _dt.strptime(iso, '%Y-%m-%d').strftime('%b %d')
        except ValueError:
            display = iso
        flagged_nights.append({'iso': iso, 'date': display, 'reasons': reasons})

    # ── Determine data source from patient ID pattern ────────────────────────
    # PT-3xxx → Synthea, everything else → Oura
    if patient_id.startswith('PT-3'):
        ds_key       = 'synthea'
        source_label = 'Synthea FHIR'
        data_points  = 12
    else:
        ds_key       = 'oura'
        source_label = 'Oura V2 API'
        data_points  = 30

    # ── Feature groups from the registry ────────────────────────────────────
    try:
        from data.feature_registry import get_feature_groups_for_source
        from data.base import DataSource
        source_enum    = DataSource.SYNTHEA if ds_key == 'synthea' else DataSource.OURA
        feature_groups = get_feature_groups_for_source(source_enum)
    except Exception:
        feature_groups = {}

    # Flat dict {column_name: display_name} used by the JS importance chart
    feature_display_names = {
        fc.name: fc.display_name
        for group_features in feature_groups.values()
        for fc in group_features
    }

    # ── Hardcoded demo experiment results ────────────────────────────────────
    results = {
        'auc':       0.91,
        'precision': 0.87,
        'recall':    0.84,
        'f1':        0.85,
    }

    if ds_key == 'synthea':
        feature_importance = [
            {'name': 'HbA1c',           'importance': 0.32},
            {'name': 'Blood Glucose',   'importance': 0.27},
            {'name': 'Systolic BP',     'importance': 0.19},
            {'name': 'BMI',             'importance': 0.12},
            {'name': 'LDL Cholesterol', 'importance': 0.06},
            {'name': 'Heart Rate',      'importance': 0.04},
        ]
    else:
        feature_importance = [
            {'name': 'REM Sleep %',     'importance': 0.31},
            {'name': 'HRV Balance',     'importance': 0.24},
            {'name': 'Deep Sleep %',    'importance': 0.18},
            {'name': 'Body Temp Dev.',  'importance': 0.13},
            {'name': 'Step Count',      'importance': 0.08},
            {'name': 'Resting HR',      'importance': 0.06},
        ]

    rng = random.Random(hash(patient_id))
    confidence_scores = [round(rng.uniform(0.20, 0.95), 2) for _ in range(30)]

    experiments = [
        {
            'id':         1,
            'model':      'XGBoost Classifier',
            'features':   '5 features',
            'auc':        0.91,
            'f1':         0.85,
            'is_current': True,
        },
        {
            'id':         2,
            'model':      'Random Forest',
            'features':   '5 features',
            'auc':        0.87,
            'f1':         0.82,
            'is_current': False,
        },
        {
            'id':         3,
            'model':      'LSTM',
            'features':   '5 features',
            'auc':        0.84,
            'f1':         0.79,
            'is_current': False,
        },
    ]

    return render_template(
        'risk_analysis_lab.html',
        patient=patient,
        active_tab='risk-analysis-lab',
        feature_groups=feature_groups,
        feature_display_names=feature_display_names,
        data_points=data_points,
        source_label=source_label,
        results=results,
        feature_importance=feature_importance,
        confidence_scores=confidence_scores,
        experiments=experiments,
        data_source=ds_key,
        flagged_nights=flagged_nights,
    )


@app.route('/patient/<patient_id>/cohort-data-explorer')
def cohort_data_explorer(patient_id):
    """Data Explorer — interactive multi-signal time series chart."""
    patients = load_patient_data()
    patient  = next((p for p in patients if p['id'] == patient_id), None)
    if not patient:
        return "Patient not found", 404

    # Detect data source (PT-3xxx → Synthea, all others → Oura).
    # If OMH/IEEE wearable records exist, the explorer should show that layer.
    ds_key = 'synthea' if patient_id.startswith('PT-3') else 'oura'
    standard_wearable = get_standard_wearable_series(patient_id)
    explorer_source_key = 'oura' if standard_wearable is not None else ds_key
    source_label = (
        standard_wearable.metadata.get('data_source_label', 'OMH/IEEE Wearable Records')
        if standard_wearable is not None
        else {'synthea': 'Synthea FHIR', 'oura': 'Oura V2 API'}[ds_key]
    )

    # Load feature registry
    try:
        from data.feature_registry import get_feature_groups_for_source, get_features_for_source
        from data.base import DataSource
        source_enum    = DataSource.SYNTHEA if explorer_source_key == 'synthea' else DataSource.OURA
        feature_groups = get_feature_groups_for_source(source_enum)
        all_features   = get_features_for_source(source_enum)
    except Exception:
        feature_groups = {}
        all_features   = []

    # For Synthea, load FHIR. For wearable patients, prefer OMH/IEEE records.
    import numpy as np
    synthea_ts = None
    standard_ts = standard_wearable.time_series if standard_wearable is not None else None
    if standard_ts is None and ds_key == 'synthea':
        try:
            from data.synthea_adapter import SyntheaAdapter
            fhir_path = f'demo_data/demo_synthea/{patient_id}.json'
            if os.path.isfile(fhir_path):
                synthea_ts = SyntheaAdapter().load_from_fhir(fhir_path, patient_id).time_series
        except Exception as e:
            print(f"[cohort-data-explorer] Could not load Synthea FHIR: {e}")

    if standard_ts is not None and not standard_ts.empty:
        idx      = standard_ts.index
        N_DAYS   = len(idx)
        col_data = {col: standard_ts[col].tolist() for col in standard_ts.columns}
    elif synthea_ts is not None and not synthea_ts.empty:
        idx      = synthea_ts.index
        N_DAYS   = len(idx)
        col_data = {col: synthea_ts[col].tolist() for col in synthea_ts.columns}
    else:
        N_DAYS = 90
        seed   = abs(hash(patient_id)) % (2 ** 31)
        rng    = np.random.default_rng(seed)
        idx    = pd.date_range(end='2024-12-31', periods=N_DAYS, freq='D')
        col_data = {
            'rem_sleep_pct':          rng.uniform(10, 30, N_DAYS),
            'deep_sleep_pct':         rng.uniform(8,  25, N_DAYS),
            'sleep_latency':          rng.uniform(5,  45, N_DAYS),
            'hrv_balance':            rng.uniform(20, 80, N_DAYS),
            'body_temp_deviation':    rng.uniform(-1.0, 1.0, N_DAYS),
            'resting_hr':             rng.uniform(50, 75, N_DAYS),
            'step_count':             rng.uniform(2000, 15000, N_DAYS),
            'inactivity_alerts':      rng.uniform(0, 10, N_DAYS),
            'heart_rate':             rng.uniform(58, 88, N_DAYS),
            'systolic_bp':            rng.uniform(108, 175, N_DAYS),
            'diastolic_bp':           rng.uniform(65, 110, N_DAYS),
            'respiratory_rate':       rng.uniform(13, 18, N_DAYS),
            'body_temperature':       rng.uniform(36.4, 37.2, N_DAYS),
            'body_weight_kg':         rng.uniform(60, 120, N_DAYS),
            'bmi':                    rng.uniform(20, 40, N_DAYS),
            'glucose_mgdl':           rng.uniform(75, 250, N_DAYS),
            'hba1c_pct':              rng.uniform(4.8, 11.0, N_DAYS),
            'total_cholesterol_mgdl': rng.uniform(150, 270, N_DAYS),
            'ldl_cholesterol_mgdl':   rng.uniform(80, 185, N_DAYS),
        }

    dates = [pd.Timestamp(d).strftime('%b %d') for d in idx]

    # Assign a color to each feature (by declaration order)
    features_list = []
    for i, fc in enumerate(all_features):
        color  = FEATURE_COLORS[i % len(FEATURE_COLORS)]
        values = [round(float(v), 3) for v in col_data.get(fc.name, [0.0] * N_DAYS)]
        features_list.append({
            'name':             fc.name,
            'display_name':     fc.display_name,
            'unit':             fc.unit,
            'group':            fc.group,
            'color':            color,
            'default_selected': fc.default_selected,
            'values':           values,
        })

    features_by_name = {f['name']: f for f in features_list}

    # Generate synthetic cohort bands (mean ± std) per feature per cohort.
    # Values are in the same units as the patient's data so JS can normalize
    # them on the same scale as the patient's own signal.
    cohort_band_cfg = [
        ('all',   0.00,  1.30),   # all 140 patients — widest spread
        ('stage', 0.05,  0.90),   # Child-Pugh B stage-matched (47 pts)
        ('age',  -0.03,  0.70),   # age/sex-matched (23 pts)
    ]
    cohort_bands: dict = {}
    cb_seed_base = abs(hash(patient_id)) % (2 ** 31)
    for cohort_id, bias_frac, spread_frac in cohort_band_cfg:
        cb_rng = np.random.default_rng(cb_seed_base + abs(hash(cohort_id)) % 100_000)
        feat_bands: dict = {}
        for feat in features_list:
            vals      = np.array(feat['values'], dtype=float)
            feat_mean = float(vals.mean())
            feat_rng  = float(vals.max() - vals.min()) if vals.max() != vals.min() else (abs(feat_mean) * 0.2 or 1.0)
            bias      = bias_frac * feat_rng
            # Gentle random walk so the band isn't perfectly flat
            walk = np.cumsum(cb_rng.normal(0, feat_rng * 0.015, N_DAYS))
            walk -= walk.mean()
            cohort_mean = np.clip(feat_mean + bias + walk,
                                  vals.min() - feat_rng * 0.25,
                                  vals.max() + feat_rng * 0.25)
            cohort_std  = np.full(N_DAYS, feat_rng * 0.18 * spread_frac)
            feat_bands[feat['name']] = {
                'mean': [round(float(v), 3) for v in cohort_mean.tolist()],
                'std':  [round(float(v), 3) for v in cohort_std.tolist()],
            }
        cohort_bands[cohort_id] = feat_bands

    return render_template(
        'cohort_data_explorer.html',
        patient=patient,
        active_tab='cohort-data-explorer',
        feature_groups=feature_groups,
        features_by_name=features_by_name,
        source_label=source_label,
        chart_data={'dates': dates, 'features': features_list, 'cohort_bands': cohort_bands},
        total_days=N_DAYS,
        data_source=ds_key,
    )



@app.route('/api/confounders/<patient_id>')
def get_confounders(patient_id):
    return jsonify(_load_confounders().get(patient_id, {}))


@app.route('/api/confounders/<patient_id>', methods=['POST'])
def set_confounder(patient_id):
    body  = request.get_json(silent=True) or {}
    date  = (body.get('date') or '').strip()
    flags = body.get('confounders', [])
    if not date:
        return jsonify({'error': 'date required'}), 400
    data = _load_confounders()
    pt   = data.setdefault(patient_id, {})
    if flags:
        pt[date] = flags
    else:
        pt.pop(date, None)
    _save_confounders(data)
    return jsonify({'ok': True})


@app.route('/api/confounders/<patient_id>/<path:date>', methods=['DELETE'])
def delete_confounder(patient_id, date):
    data = _load_confounders()
    data.get(patient_id, {}).pop(date, None)
    _save_confounders(data)
    return jsonify({'ok': True})


@app.route('/patient/<patient_id>/report')
def patient_report(patient_id):
    """Print-optimized patient report (save as PDF via browser)."""
    view = _compute_patient_view_data(patient_id)
    if view is None:
        return "Patient not found", 404
    return render_template('patient_report.html',
                           patient=view['patient'],
                           clinical_history=view['clinical_history'],
                           sleep_rows=view['sleep_rows'],
                           confounders=view['confounders'],
                           confounder_types=CONFOUNDER_TYPES,
                           avg_eff=view['avg_eff'],
                           avg_rem=view['avg_rem'],
                           avg_waso=view['avg_waso'],
                           avg_spo2=view['avg_spo2'],
                           report_date=datetime.now().strftime('%B %d, %Y'))


@app.route('/patient/<patient_id>/report.docx')
def patient_report_docx(patient_id):
    """Download patient report as a Word-compatible HTML document."""
    from flask import make_response
    patients = load_patient_data()
    if not next((p for p in patients if p['id'] == patient_id), None):
        return "Patient not found", 404
    html = patient_report(patient_id)
    if isinstance(html, tuple):
        return html
    resp = make_response(html)
    resp.headers['Content-Type']        = 'application/vnd.ms-word'
    resp.headers['Content-Disposition'] = f'attachment; filename="{patient_id}_report.doc"'
    return resp


@app.route('/patient/<patient_id>/ai-assistant')
def ai_assistant(patient_id):
    """AI Assistant — placeholder SHAP rationale view (Phase 1)."""
    patients = load_patient_data()
    patient  = next((p for p in patients if p['id'] == patient_id), None)
    if not patient:
        return "Patient not found", 404

    ds_key       = 'synthea' if patient_id.startswith('PT-3') else 'oura'
    source_label = {'synthea': 'Synthea FHIR', 'oura': 'Oura V2 API'}[ds_key]

    # ── Placeholder SHAP top-3 (static until Phase 2 LLM pipeline is wired) ──
    if ds_key == 'synthea':
        shap_top3 = [
            {'display_name': 'HbA1c',          'shap_value':  0.298, 'abs_shap': 0.298, 'direction': 'positive'},
            {'display_name': 'Blood Glucose',  'shap_value':  0.241, 'abs_shap': 0.241, 'direction': 'positive'},
            {'display_name': 'Systolic BP',    'shap_value':  0.187, 'abs_shap': 0.187, 'direction': 'positive'},
        ]
        rationale_text = (
            "HbA1c and fasting blood glucose are the dominant risk factors for this patient, "
            "indicating suboptimal glycaemic control. Elevated systolic blood pressure compounds "
            "cardiovascular risk. LLM-generated narrative will be available in Phase 2 once the "
            "Llama 3 / GPT-4o pipeline is integrated."
        )
    else:
        shap_top3 = [
            {'display_name': 'REM Sleep %',    'shap_value': -0.287, 'abs_shap': 0.287, 'direction': 'negative'},
            {'display_name': 'HRV Balance',    'shap_value': -0.221, 'abs_shap': 0.221, 'direction': 'negative'},
            {'display_name': 'Deep Sleep %',   'shap_value': -0.163, 'abs_shap': 0.163, 'direction': 'negative'},
        ]
        rationale_text = (
            "Reduced REM Sleep % and HRV Balance are the leading risk indicators for this patient, "
            "consistent with published findings on covert hepatic encephalopathy. Deep Sleep % also "
            "shows a suppressive pattern. LLM-generated narrative will be available in Phase 2 once "
            "the Llama 3 / GPT-4o pipeline is integrated."
        )

    # Placeholder cognitive match / confidence
    class _Rationale:
        pass

    rationale = _Rationale()
    rationale.rationale_text        = rationale_text
    rationale.cognitive_match_score = 1.0        # placeholder
    rationale.confidence            = 'High'
    rationale.top_features          = [e['display_name'] for e in shap_top3]

    return render_template(
        'ai_assistant.html',
        patient=patient,
        active_tab='ai-assistant',
        source_label=source_label,
        data_source=ds_key,
        rationale=rationale,
        shap_top3=shap_top3,
        n_features=len(shap_top3),
        n_train=72,
        model_name='XGBoost Classifier (placeholder)',
        error=None,
    )


@app.route('/api/notebooks')
def list_notebooks():
    """List all .ipynb files in the notebooks directory."""
    os.makedirs(NOTEBOOKS_DIR, exist_ok=True)
    notebooks = []
    for fname in sorted(os.listdir(NOTEBOOKS_DIR)):
        if not fname.endswith('.ipynb'):
            continue
        fpath = os.path.join(NOTEBOOKS_DIR, fname)
        stat  = os.stat(fpath)
        notebooks.append({
            'name':        fname,
            'created':     datetime.fromtimestamp(stat.st_ctime).strftime('%b %d, %Y'),
            'modified':    datetime.fromtimestamp(stat.st_mtime).strftime('%b %d, %Y %H:%M'),
            'size_kb':     round(stat.st_size / 1024, 1),
            'jupyter_url': f'{JUPYTER_BASE_URL}/lab/tree/{NOTEBOOKS_DIR}/{fname}',
            'path':        f'{NOTEBOOKS_DIR}/{fname}',
        })
    return jsonify({'notebooks': notebooks, 'jupyter_base': JUPYTER_BASE_URL})


@app.route('/api/notebooks/new', methods=['POST'])
def create_notebook():
    """Create a starter .ipynb pre-loaded with cohort sleep data."""
    body = request.get_json(silent=True) or {}
    name = re.sub(r'[^\w\- ]', '_', (body.get('name') or '').strip())
    if not name:
        name = f"cohort_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not name.endswith('.ipynb'):
        name += '.ipynb'

    os.makedirs(NOTEBOOKS_DIR, exist_ok=True)
    fpath = os.path.join(NOTEBOOKS_DIR, name)
    if os.path.exists(fpath):
        return jsonify({'error': f'Notebook "{name}" already exists'}), 409

    patients       = load_patient_data()
    cohort_stats   = _compute_cohort_stats(patients)
    sleep_summaries = _compute_sleep_summaries(patients)

    cohort_payload = {
        'patients': [
            {'id': p['id'], 'risk_level': p.get('risk_level', ''),
             'conditions': p.get('conditions', ''), 'data_source': p.get('data_source', 'oura')}
            for p in patients
        ],
        'sleep_summaries': sleep_summaries,
        'cohort_stats': {'total': cohort_stats['total'], 'by_risk': cohort_stats['by_risk']},
    }

    nb = _build_notebook(json.dumps(cohort_payload, indent=2), len(patients), cohort_stats)
    with open(fpath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)

    return jsonify({
        'name':        name,
        'path':        fpath,
        'jupyter_url': f'{JUPYTER_BASE_URL}/lab/tree/{NOTEBOOKS_DIR}/{name}',
    })


def _build_notebook(cohort_json: str, n: int, stats: dict) -> dict:
    """Return a Jupyter notebook dict pre-loaded with cohort data."""
    br   = stats.get('by_risk', {})
    now  = datetime.now().strftime('%Y-%m-%d %H:%M')
    return {
        'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.12.0'},
        },
        'cells': [
            {'cell_type': 'markdown', 'id': 'intro', 'metadata': {}, 'source': [
                f'# JupyterHealth — Cohort Analysis\n',
                f'\n',
                f'**Study:** Hepatic Encephalopathy Sleep Biomarkers  \n',
                f'**Cohort:** {n} participants | High: {br.get("high",0)} · Medium: {br.get("medium",0)} · Low: {br.get("low",0)}  \n',
                f'**Generated:** {now}\n',
                f'\n',
                f'`cohort_data` contains:\n',
                f'- `patients` — ID, risk level, conditions, data source\n',
                f'- `sleep_summaries` — 14-day avg sleep metrics (REM, Deep, Efficiency, WASO, SpO₂, …)\n',
                f'- `cohort_stats` — risk distribution\n',
            ]},
            {'cell_type': 'code', 'execution_count': None, 'id': 'imports', 'metadata': {}, 'outputs': [], 'source': [
                'import pandas as pd\n',
                'import numpy as np\n',
                'import matplotlib.pyplot as plt\n',
                'import matplotlib.patches as mpatches\n',
                'from matplotlib import rcParams\n',
                '\n',
                "rcParams['figure.figsize'] = (12, 4)\n",
                "plt.style.use('seaborn-v0_8-whitegrid')\n",
            ]},
            {'cell_type': 'code', 'execution_count': None, 'id': 'load-data', 'metadata': {}, 'outputs': [], 'source': [
                f'# ── Cohort data from JupyterHealth workbench ──────────────────────────\n',
                f'import json\n',
                f'cohort_data  = json.loads(r\'\'\'{cohort_json}\'\'\')\n',
                f'patients_df  = pd.DataFrame(cohort_data["patients"])\n',
                f'sleep_df     = pd.DataFrame(cohort_data["sleep_summaries"])\n',
                f'\n',
                f'print(f"Cohort: {{len(patients_df)}} participants")\n',
                f'sleep_df.head()\n',
            ]},
            {'cell_type': 'markdown', 'id': 'analysis-md', 'metadata': {}, 'source': [
                '## Sleep Biomarkers by Risk Level\n',
                '\n',
                'Compare REM %, Sleep Efficiency, and WASO across high / medium / low risk groups.\n',
            ]},
            {'cell_type': 'code', 'execution_count': None, 'id': 'sleep-by-risk', 'metadata': {}, 'outputs': [], 'source': [
                'merged   = sleep_df.merge(patients_df[["id","risk_level"]], on="id", how="left")\n',
                'merged   = merged[merged["risk_level"].isin(["high","medium","low"])]\n',
                'metrics  = ["avg_rem_pct","avg_deep_pct","avg_efficiency","avg_waso","avg_spo2"]\n',
                'labels   = ["REM %","Deep %","Efficiency %","WASO (min)","SpO₂ %"]\n',
                'colors   = {"high":"#ef4444","medium":"#f59e0b","low":"#22c55e"}\n',
                '\n',
                'fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4))\n',
                'for ax, metric, label in zip(axes, metrics, labels):\n',
                '    for risk in ["high","medium","low"]:\n',
                '        vals = merged[merged["risk_level"]==risk][metric]\n',
                '        ax.bar(risk, vals.mean(), color=colors[risk], alpha=0.85)\n',
                '        ax.errorbar(risk, vals.mean(), yerr=vals.std(), fmt="none", color="#475569", capsize=4)\n',
                '    ax.set_title(label, fontsize=11, fontweight="bold"); ax.set_xlabel("")\n',
                'handles = [mpatches.Patch(color=c,label=r.capitalize()) for r,c in colors.items()]\n',
                'axes[0].legend(handles=handles, fontsize=9)\n',
                'fig.suptitle("Sleep Biomarkers by Risk Level", fontsize=13, fontweight="bold", y=1.02)\n',
                'plt.tight_layout(); plt.show()\n',
            ]},
            {'cell_type': 'code', 'execution_count': None, 'id': 'correlation', 'metadata': {}, 'outputs': [], 'source': [
                'cols   = ["avg_tst","avg_rem_pct","avg_deep_pct","avg_efficiency","avg_latency","avg_waso","avg_spo2"]\n',
                'labels = ["TST","REM%","Deep%","Efficiency","Latency","WASO","SpO₂"]\n',
                'corr   = sleep_df[cols].corr()\n',
                '\n',
                'fig, ax = plt.subplots(figsize=(7, 6))\n',
                'im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)\n',
                'plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)\n',
                'ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(cols)))\n',
                'ax.set_xticklabels(labels, rotation=45, ha="right")\n',
                'ax.set_yticklabels(labels)\n',
                'for i in range(len(cols)):\n',
                '    for j in range(len(cols)):\n',
                '        v = corr.iloc[i,j]\n',
                '        ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=8,\n',
                '                color="white" if abs(v)>0.5 else "black")\n',
                'ax.set_title("Sleep Metric Correlation Matrix", fontsize=12, fontweight="bold")\n',
                'plt.tight_layout(); plt.show()\n',
            ]},
        ],
    }


@app.route('/api/notebooks/inject-chart', methods=['POST'])
def inject_chart():
    """
    Inject a matplotlib cell reproducing the current cohort-data-explorer chart into a notebook.
    Body: {patient_id, notebook (optional), features: [{key, label, unit, color, values}], dates: [str]}
    If notebook is omitted or blank, creates a new one.
    """
    body       = request.get_json(silent=True) or {}
    patient_id = (body.get('patient_id') or '').strip()
    nb_name    = (body.get('notebook') or '').strip()
    features   = body.get('features', [])
    dates      = body.get('dates', [])

    if not patient_id:
        return jsonify({'error': 'patient_id required'}), 400
    if not features:
        return jsonify({'error': 'no features selected'}), 400

    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    n   = len(dates)

    # Build the matplotlib cell source
    colors_repr = repr([f['color'] for f in features])
    labels_repr = repr([f['label'] for f in features])
    units_repr  = repr([f['unit']  for f in features])
    dates_repr  = repr(dates)
    data_repr   = repr([[round(float(v), 3) if v is not None else None
                         for v in f.get('values', [])] for f in features])

    cell_source = (
        f'# ── Data Explorer chart — {patient_id} ({len(features)} signals · {n} days)'
        f'  (added {now}) ──\n'
        f'import numpy as np\nimport matplotlib.pyplot as plt\n\n'
        f'dates    = {dates_repr}\n'
        f'signals  = {data_repr}\n'
        f'labels   = {labels_repr}\n'
        f'units    = {units_repr}\n'
        f'colors   = {colors_repr}\n\n'
        f'# Normalize each signal to 0–1 for overlay\n'
        f'def norm(vals):\n'
        f'    mn, mx = min(v for v in vals if v is not None), max(v for v in vals if v is not None)\n'
        f'    rng = mx - mn\n'
        f'    return [(v - mn)/rng if v is not None and rng else 0.5 for v in vals]\n\n'
        f'x = range(len(dates))\n'
        f'tick_step = max(1, len(dates)//7)\n'
        f'fig, ax = plt.subplots(figsize=(14, 5))\n'
        f'for i, (sig, label, color, unit) in enumerate(zip(signals, labels, colors, units)):\n'
        f'    y = norm(sig)\n'
        f'    ax.plot(x, y, color=color, linewidth=2, alpha=0.85, label=f"{{label}} ({{unit}})")\n\n'
        f'ax.set_xticks(range(0, len(dates), tick_step))\n'
        f'ax.set_xticklabels([dates[i] for i in range(0, len(dates), tick_step)], rotation=45, fontsize=8)\n'
        f'ax.set_ylabel("Normalized value (0–1 per signal)", fontsize=10)\n'
        f'ax.set_title("Data Explorer — {patient_id} · {len(features)} signals", fontsize=13, fontweight="bold")\n'
        f'ax.legend(loc="upper right", fontsize=9, ncol=2)\n'
        f'ax.spines[["top","right"]].set_visible(False)\n'
        f'plt.tight_layout()\nplt.show()\n'
    )

    os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

    if nb_name:
        fpath = os.path.join(NOTEBOOKS_DIR, nb_name)
        if not os.path.isfile(fpath):
            return jsonify({'error': f'Notebook "{nb_name}" not found'}), 404
        with open(fpath, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    else:
        nb_name = f"{patient_id}_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ipynb"
        fpath   = os.path.join(NOTEBOOKS_DIR, nb_name)
        nb = {
            'nbformat': 4, 'nbformat_minor': 5,
            'metadata': {
                'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                'language_info': {'name': 'python', 'version': '3.12.0'},
            },
            'cells': [
                {'cell_type': 'markdown', 'id': 'intro', 'metadata': {}, 'source':
                 [f'# Data Explorer Chart — {patient_id}\n\nGenerated by JupyterHealth Workbench on {now}\n']},
                {'cell_type': 'code', 'execution_count': None, 'id': 'imports', 'metadata': {},
                 'outputs': [], 'source': ['import numpy as np\nimport matplotlib.pyplot as plt\n']},
            ],
        }

    nb.setdefault('cells', []).append({
        'cell_type': 'code', 'execution_count': None,
        'id': f'chart-{uuid.uuid4().hex[:8]}', 'metadata': {}, 'outputs': [],
        'source': [cell_source],
    })

    with open(fpath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)

    # Best-effort Jupyter REST API push
    try:
        import urllib.request
        jup_url = f'{JUPYTER_BASE_URL}/api/contents/{NOTEBOOKS_DIR}/{nb_name}'
        payload = json.dumps({'type': 'file', 'format': 'text',
                              'content': json.dumps(nb)}).encode()
        req = urllib.request.Request(jup_url, data=payload, method='PUT',
                                     headers={'Content-Type': 'application/json'})
        urllib.request.urlopen(req, timeout=2)
    except Exception:
        pass

    return jsonify({
        'ok': True,
        'notebook': nb_name,
        'jupyter_url': f'{JUPYTER_BASE_URL}/lab/tree/{NOTEBOOKS_DIR}/{nb_name}',
    })


@app.route('/api/notebooks/inject-cell', methods=['POST'])
def inject_cell():
    """
    Inject a generated Python cell into an existing notebook file.
    Uses the Jupyter Contents REST API (PUT /api/contents/{path}) to write the
    updated .ipynb, so the change is visible next time Jupyter opens the file.
    """
    body          = request.get_json(silent=True) or {}
    notebook_name = (body.get('notebook') or '').strip()
    patient_ids   = body.get('patient_ids', [])
    analysis_type = body.get('analysis_type', 'sleep_comparison')
    metrics       = body.get('metrics', ['avg_rem_pct', 'avg_efficiency'])
    date_range    = int(body.get('date_range', 14))

    if not notebook_name:
        return jsonify({'error': 'notebook name required'}), 400

    fpath = os.path.join(NOTEBOOKS_DIR, notebook_name)
    if not os.path.isfile(fpath):
        return jsonify({'error': f'Notebook "{notebook_name}" not found'}), 404

    # Build the Python source for the new cell
    patients       = load_patient_data()
    sleep_summaries = _compute_sleep_summaries(patients)
    pt_map         = {s['id']: s for s in sleep_summaries}
    selected_pts   = [pt_map[pid] for pid in patient_ids if pid in pt_map] or sleep_summaries
    pt_json        = json.dumps(selected_pts, indent=2)
    now            = datetime.now().strftime('%Y-%m-%d %H:%M')

    metric_labels = {
        'avg_rem_pct':   'REM %',   'avg_deep_pct': 'Deep %',
        'avg_tst':       'TST (h)', 'avg_efficiency': 'Efficiency %',
        'avg_latency':   'Latency (min)', 'avg_waso': 'WASO (min)', 'avg_spo2': 'SpO₂ %',
    }
    m_list  = repr(metrics)
    m_labels = repr({k: metric_labels.get(k, k) for k in metrics})

    if analysis_type == 'sleep_comparison':
        cell_source = (
            f'# ── Sleep Comparison — {len(selected_pts)} patients · {date_range}-day avg'
            f'  (added {now}) ──\n'
            f'import pandas as pd, matplotlib.pyplot as plt, json\n\n'
            f'data = json.loads({repr(pt_json)})\n'
            f'df = pd.DataFrame(data).set_index("id")\n'
            f'metrics = {m_list}\n'
            f'labels  = {m_labels}\n'
            f'risk_colors = {{"high":"#ef4444","medium":"#f59e0b","low":"#10b981"}}\n'
            f'colors = [risk_colors.get(df.loc[pid,"risk"], "#3b82f6") for pid in df.index]\n\n'
            f'fig, axes = plt.subplots(1, max(1,len(metrics)), figsize=(4*max(1,len(metrics)), 5))\n'
            f'if len(metrics)==1: axes=[axes]\n'
            f'for ax, m in zip(axes, metrics):\n'
            f'    ax.bar(df.index, df[m], color=colors, alpha=0.85, edgecolor="white")\n'
            f'    ax.set_title(labels[m], fontsize=11, fontweight="bold")\n'
            f'    ax.tick_params(axis="x", rotation=60, labelsize=8)\n'
            f'    ax.spines[["top","right"]].set_visible(False)\n'
            f'plt.suptitle("Sleep Comparison — {date_range}-Day Avg · {len(selected_pts)} Patients",'
            f'fontsize=13, fontweight="bold")\n'
            f'plt.tight_layout()\nplt.show()\n'
        )
    elif analysis_type == 'cohort_stats':
        cell_source = (
            f'# ── Cohort Statistics (added {now}) ──\n'
            f'import pandas as pd, matplotlib.pyplot as plt, json\n\n'
            f'data = json.loads({repr(pt_json)})\n'
            f'df = pd.DataFrame(data)\n'
            f'metrics = {m_list}\n'
            f'labels  = {m_labels}\n\n'
            f'fig, axes = plt.subplots(1, max(1,len(metrics)), figsize=(3.5*max(1,len(metrics)), 5))\n'
            f'if len(metrics)==1: axes=[axes]\n'
            f'for ax, m in zip(axes, metrics):\n'
            f'    groups = [df[df["risk"]==r][m].dropna().tolist() for r in ["high","medium","low"]]\n'
            f'    bp = ax.boxplot(groups, labels=["High","Med","Low"], patch_artist=True)\n'
            f'    for patch, color in zip(bp["boxes"], ["#fecaca","#fef3c7","#dcfce7"]):\n'
            f'        patch.set_facecolor(color)\n'
            f'    ax.set_title(labels[m], fontsize=11, fontweight="bold")\n'
            f'    ax.spines[["top","right"]].set_visible(False)\n'
            f'plt.suptitle("Distribution by Risk — {len(selected_pts)} Patients", fontsize=13, fontweight="bold")\n'
            f'plt.tight_layout()\nplt.show()\n'
        )
    else:
        cell_source = (
            f'# ── {analysis_type} — {len(selected_pts)} patients (added {now}) ──\n'
            f'import pandas as pd, json\n\n'
            f'data = json.loads({repr(pt_json)})\n'
            f'df = pd.DataFrame(data)\nprint(df)\n'
        )

    # Load the notebook, append the cell, write back via Jupyter REST API
    with open(fpath, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    new_cell = {
        'cell_type': 'code',
        'execution_count': None,
        'id': f'injected-{uuid.uuid4().hex[:8]}',
        'metadata': {},
        'outputs': [],
        'source': [cell_source],
    }
    nb.setdefault('cells', []).append(new_cell)

    # Write locally (always works)
    with open(fpath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)

    # Also try Jupyter REST API so a live Jupyter session picks up the change
    jupyter_url = f'{JUPYTER_BASE_URL}/api/contents/{NOTEBOOKS_DIR}/{notebook_name}'
    try:
        import urllib.request
        payload = json.dumps({'type': 'file', 'format': 'text',
                              'content': json.dumps(nb)}).encode()
        req = urllib.request.Request(jupyter_url, data=payload, method='PUT',
                                     headers={'Content-Type': 'application/json'})
        urllib.request.urlopen(req, timeout=2)
    except Exception:
        pass  # local write already done; REST push is best-effort

    return jsonify({
        'ok': True,
        'notebook': notebook_name,
        'jupyter_url': f'{JUPYTER_BASE_URL}/lab/tree/{NOTEBOOKS_DIR}/{notebook_name}',
    })


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    app.run(debug=debug, host='0.0.0.0', port=port)
