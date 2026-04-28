"""
Clinical Coordinator Dashboard - Flask App
Exact mockup implementation
"""

from flask import Flask, render_template, jsonify, request
import pandas as pd
from datetime import datetime, timedelta
import json
import random
import re
import os
import uuid

app = Flask(__name__)

# Ordered color palette assigned to features by their declaration index.
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
        
        # Append Synthea patients from FHIR bundles
        patients.extend(load_synthea_patients())
        return patients
    except Exception as e:
        print(f"Error loading data: {e}")
        # Still try to return Synthea patients even if Excel fails
        return load_synthea_patients()


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
    return {
        'total': total,
        'by_risk': by_risk,
        'by_source': by_source,
        'conditions': sorted(condition_counts.items(), key=lambda x: -x[1])[:8],
        'active': sum(1 for p in patients if p.get('status') == 'active'),
    }


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
        resp = ("This workbench supports a **Cornell/Columbia clinical research study** — "
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
    cohort_stats = _compute_cohort_stats(patients)
    return render_template('dashboard.html', patients=patients, cohort_stats=cohort_stats)


@app.route('/api/patients')
def get_patients():
    patients = load_patient_data()
    return jsonify(patients)


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


@app.route('/patient/<patient_id>')
def patient_detail(patient_id):
    """Show detailed patient metrics view."""
    patients = load_patient_data()
    patient = next((p for p in patients if p['id'] == patient_id), None)

    if not patient:
        return "Patient not found", 404

    # Generate 14 days (2 weeks) of detailed metric data
    import random
    random.seed(hash(patient_id))  # Consistent data for same patient

    dates = []
    heart_rate_data = []
    respiratory_data = []
    hrv_data = []
    sleep_duration_data = []
    steps_data = []
    temperature_data = []

    base_date = datetime(2024, 12, 1)
    for i in range(14):
        date = base_date + timedelta(days=i)
        dates.append(date.strftime("%b %d"))
        heart_rate_data.append(random.randint(48, 72))
        respiratory_data.append(round(random.uniform(12, 18), 1))
        hrv_data.append(random.randint(20, 80))
        sleep_duration_data.append(round(random.uniform(5, 9), 1))
        steps_data.append(random.randint(2000, 15000))
        temperature_data.append(round(random.uniform(-1.0, 1.0), 2))

    patient['dates'] = dates
    patient['heart_rate_data'] = heart_rate_data
    patient['respiratory_data'] = respiratory_data
    patient['hrv_data'] = hrv_data
    patient['sleep_duration_data'] = sleep_duration_data
    patient['steps_data'] = steps_data
    patient['temperature_data'] = temperature_data

    # ── Clinical history ──────────────────────────────────────────────────
    clinical_history = None
    ds_key = 'synthea' if patient_id.startswith('PT-3') else 'oura'
    _ch_rng = random.Random(hash(patient_id))

    # ── Oura / Liver Disease patients ─────────────────────────────────────
    if ds_key == 'oura':
        try:
            age = _ch_rng.randint(45, 72)
            sex = _ch_rng.choice(['Male', 'Female'])

            # MELD-Na score (6–40; higher = worse prognosis)
            meld_6mo  = _ch_rng.randint(9, 26)
            meld_now  = meld_6mo + _ch_rng.randint(-1, 5)
            meld_delta = round(meld_now - meld_6mo, 1)

            # Liver labs
            ammonia   = _ch_rng.randint(28, 110)   # normal < 35 μmol/L
            bilirubin = round(_ch_rng.uniform(1.2, 9.0), 1)  # normal < 1.2 mg/dL
            inr       = round(_ch_rng.uniform(1.2, 2.4), 1)  # normal 0.9–1.1
            creatinine= round(_ch_rng.uniform(0.7, 2.3), 1)  # normal 0.6–1.2
            sodium    = _ch_rng.randint(128, 140)             # hyponatremia < 135

            # Wearable metrics from the 14-day window just generated
            latest_hr  = heart_rate_data[-1]
            latest_hrv = hrv_data[-1]

            # Active conditions
            conditions = ['Cirrhosis (Child-Pugh B)' if meld_now >= 15 else 'Cirrhosis (Child-Pugh A)']
            if ammonia > 60 or patient.get('status') in ('follow-up', 'outreach'):
                conditions.append('Covert Hepatic Encephalopathy')
            if inr > 1.8:
                conditions.append('Coagulopathy')
            if sodium < 133:
                conditions.append('Hyponatremia')
            if patient.get('has_oura'):
                conditions.append('Active Wearable Monitoring')

            # Enrollment date: ~2 years ago seeded per patient
            enroll_offset = _ch_rng.randint(300, 900)
            enroll_dt = datetime(2024, 12, 31) - timedelta(days=enroll_offset)
            enrollment_date = enroll_dt.strftime('%b %d, %Y')

            clinical_history = {
                'age':   age,
                'sex':   sex,
                'conditions':      conditions,
                'enrollment_date': enrollment_date,
                'latest_vitals': {
                    'MELD-Na Score':  {'value': meld_now,   'unit': 'score'},
                    'Ammonia (NH₃)':  {'value': ammonia,    'unit': 'μmol/L'},
                    'Bilirubin':      {'value': bilirubin,  'unit': 'mg/dL'},
                    'INR':            {'value': inr,        'unit': ''},
                    'Creatinine':     {'value': creatinine, 'unit': 'mg/dL'},
                    'Serum Sodium':   {'value': sodium,     'unit': 'mEq/L'},
                    'Resting HR':     {'value': latest_hr,  'unit': 'bpm'},
                    'HRV (rMSSD)':    {'value': latest_hrv, 'unit': 'ms'},
                },
                'trends': {
                    'MELD-Na': meld_delta,
                    'Ammonia': round(ammonia - _ch_rng.randint(20, 40), 1),
                },
                'data_points':       patient.get('outpatient', 0),
                'data_source_label': 'Oura Ring V2 + EHR Flowsheets',
                'last_encounter':    'Dec 01, 2024',
            }
        except Exception as e:
            print(f"[patient_detail] Oura clinical history error for {patient_id}: {e}")

    # ── Synthea / Metabolic patients ──────────────────────────────────────
    elif ds_key == 'synthea':
        try:
            from data.synthea_adapter import SyntheaAdapter
            adapter = SyntheaAdapter()
            fpath = os.path.join('demo_data/demo_synthea', f"{patient_id}.json")
            pts = (adapter.load_from_fhir(fpath, patient_id)
                   if os.path.isfile(fpath)
                   else adapter.load_demo_data(patient_id, patient.get('risk_level', 'medium')))

            sf = pts.static_features
            ts = pts.time_series

            conditions = [
                k.replace('_', ' ').title()
                for k, v in sf.items() if isinstance(v, bool) and v
            ]

            latest_vitals = {}
            vital_labels = {
                'glucose_mgdl':   ('Blood Glucose',   'mg/dL'),
                'hba1c_pct':      ('HbA1c',           '%'),
                'systolic_bp':    ('Systolic BP',      'mmHg'),
                'diastolic_bp':   ('Diastolic BP',     'mmHg'),
                'bmi':            ('BMI',              'kg/m²'),
                'body_weight_kg': ('Weight',           'kg'),
                'heart_rate':     ('Heart Rate',       'bpm'),
            }
            if not ts.empty:
                last_row = ts.iloc[-1]
                for col, (label, unit) in vital_labels.items():
                    if col in last_row.index and not pd.isna(last_row[col]):
                        latest_vitals[label] = {
                            'value': round(float(last_row[col]), 1),
                            'unit': unit,
                        }

            trends = {}
            for col, label in [('glucose_mgdl', 'Glucose'), ('hba1c_pct', 'HbA1c'), ('systolic_bp', 'Systolic BP')]:
                if col in ts.columns and len(ts) >= 2:
                    first_val, last_val = ts[col].iloc[0], ts[col].iloc[-1]
                    if not pd.isna(first_val) and not pd.isna(last_val):
                        trends[label] = round(last_val - first_val, 1)

            enrollment_date = pts.metadata.get('enrollment_date', '')
            if enrollment_date:
                try:
                    enrollment_date = datetime.strptime(enrollment_date, '%Y-%m-%d').strftime('%b %d, %Y')
                except ValueError:
                    pass

            clinical_history = {
                'age':   sf.get('age'),
                'sex':   'Male' if sf.get('sex') == 'M' else 'Female' if sf.get('sex') == 'F' else None,
                'conditions':      conditions,
                'enrollment_date': enrollment_date,
                'latest_vitals':   latest_vitals,
                'trends':          trends,
                'data_points':     pts.metadata.get('data_points_count', len(ts)),
                'data_source_label': pts.metadata.get('data_source_label', 'Synthea FHIR'),
                'last_encounter':  ts.index[-1].strftime('%b %d, %Y') if not ts.empty else None,
            }
        except Exception as e:
            print(f"[patient_detail] Synthea clinical history error for {patient_id}: {e}")

    return render_template('patient_detail.html', patient=patient,
                           active_tab='overview', clinical_history=clinical_history)


@app.route('/patient/<patient_id>/model-lab')
def model_lab(patient_id):
    """Show the Model Lab — ML model selection, training, and results."""
    patients = load_patient_data()
    patient  = next((p for p in patients if p['id'] == patient_id), None)

    if not patient:
        return "Patient not found", 404

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
        'model_lab.html',
        patient=patient,
        active_tab='model-lab',
        feature_groups=feature_groups,
        feature_display_names=feature_display_names,
        data_points=data_points,
        source_label=source_label,
        results=results,
        feature_importance=feature_importance,
        confidence_scores=confidence_scores,
        experiments=experiments,
        data_source=ds_key,
    )


@app.route('/patient/<patient_id>/data-explorer')
def data_explorer(patient_id):
    """Data Explorer — interactive multi-signal time series chart."""
    patients = load_patient_data()
    patient  = next((p for p in patients if p['id'] == patient_id), None)
    if not patient:
        return "Patient not found", 404

    # Detect data source (PT-3xxx → Synthea, all others → Oura)
    ds_key       = 'synthea' if patient_id.startswith('PT-3') else 'oura'
    source_label = {'synthea': 'Synthea FHIR', 'oura': 'Oura V2 API'}[ds_key]

    # Load feature registry
    try:
        from data.feature_registry import get_feature_groups_for_source, get_features_for_source
        from data.base import DataSource
        source_enum    = DataSource.SYNTHEA if ds_key == 'synthea' else DataSource.OURA
        feature_groups = get_feature_groups_for_source(source_enum)
        all_features   = get_features_for_source(source_enum)
    except Exception:
        feature_groups = {}
        all_features   = []

    # For Synthea, load real data from the FHIR bundle; otherwise generate synthetic
    import numpy as np
    synthea_ts = None
    if ds_key == 'synthea':
        try:
            from data.synthea_adapter import SyntheaAdapter
            fhir_path = f'demo_data/demo_synthea/{patient_id}.json'
            if os.path.isfile(fhir_path):
                synthea_ts = SyntheaAdapter().load_from_fhir(fhir_path, patient_id).time_series
        except Exception as e:
            print(f"[data-explorer] Could not load Synthea FHIR: {e}")

    if synthea_ts is not None and not synthea_ts.empty:
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

    return render_template(
        'data_explorer.html',
        patient=patient,
        active_tab='data-explorer',
        feature_groups=feature_groups,
        features_by_name=features_by_name,
        source_label=source_label,
        chart_data={'dates': dates, 'features': features_list},
        total_days=N_DAYS,
        data_source=ds_key,
    )



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


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    app.run(debug=debug, host='0.0.0.0', port=port)
