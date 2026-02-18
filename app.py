"""
Clinical Coordinator Dashboard - Flask App
Exact mockup implementation
"""

from flask import Flask, render_template, jsonify, request
import pandas as pd
from datetime import datetime, timedelta
import random

app = Flask(__name__)

# Ordered color palette assigned to features by their declaration index.
FEATURE_COLORS = [
    '#3b82f6', '#06b6d4', '#10b981', '#f59e0b',
    '#ef4444', '#8b5cf6', '#f97316', '#ec4899',
    '#84cc16', '#14b8a6', '#6366f1', '#a78bfa',
]


def load_patient_data():
    """Load and process patient data from Excel file."""
    try:
        # Try real data first, fall back to demo data
        import os
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
            
            patients.append({
                'id': f"PT-{str(int(mrn))[-4:]}",
                'mrn': mrn,
                'name': f"{first_row.get('first_name', '')} {first_row.get('last_name', '')}".strip(),
                'inpatient': inpatient_count,
                'outpatient': outpatient_count,
                'last_sync': last_sync_str,
                'has_oura': has_oura,
                'has_ehr': has_ehr,
                'status': status,
                'participation_start': format_date(inpatient_start or admit_date),
                'hospital_start': format_date(admit_date),
                'hospital_end': format_date(discharge_date),
                'sleep_score': sleep_score,
                'hrv_average': hrv_average,
                'activity_score': activity_score
            })
        
        return patients
    except Exception as e:
        print(f"Error loading data: {e}")
        return []


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


@app.route('/')
def dashboard():
    patients = load_patient_data()
    
    # Calculate counts
    all_count = len(patients)
    generating_count = len([p for p in patients if p['has_oura'] or p['has_ehr']])
    overlap_count = len([p for p in patients if p['has_oura'] and p['has_ehr']])
    complete_count = len([p for p in patients if p['status'] == 'complete'])
    follow_up_count = len([p for p in patients if p['status'] in ['follow-up', 'outreach']])
    
    return render_template('dashboard.html',
                          patients=patients,
                          all_count=all_count,
                          generating_count=generating_count,
                          overlap_count=overlap_count,
                          complete_count=complete_count,
                          follow_up_count=follow_up_count)


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
        ds     = DataSource.PPMI if patient_id.startswith('PT-2') else DataSource.OURA

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
            # PPMI features
            'baseline_updrs':      rng.uniform(10, 60, n_rows),
            'csf_alpha_synuclein': rng.uniform(1000, 3000, n_rows),
            'amyloid_beta':        rng.uniform(500, 1500, n_rows),
            'total_tau':           rng.uniform(100, 400, n_rows),
            'gba_mutation':        rng.integers(0, 3, n_rows).astype(float),
            'lrrk2_mutation':      rng.integers(0, 2, n_rows).astype(float),
            'apoe_status':         rng.integers(0, 3, n_rows).astype(float),
            'epworth_sleep':       rng.uniform(0, 18, n_rows),
            'schwab_england_adl':  rng.uniform(50, 100, n_rows),
            'datscan':             rng.uniform(0.5, 3.5, n_rows),
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
    heart_rate_data = []      # Lowest heart rate during sleep (bpm)
    respiratory_data = []      # Respiratory rate (breaths/min)
    hrv_data = []             # Heart Rate Variability (ms)
    sleep_duration_data = []   # Sleep duration (hours)
    steps_data = []           # Daily steps
    temperature_data = []      # Temperature deviation (°C)
    
    from datetime import datetime, timedelta
    base_date = datetime(2024, 12, 1)
    
    for i in range(14):  # 2 weeks of data
        date = base_date + timedelta(days=i)
        dates.append(date.strftime("%b %d"))
        
        # Generate realistic clinical data patterns
        heart_rate_data.append(random.randint(48, 72))           # Resting HR 48-72 bpm
        respiratory_data.append(round(random.uniform(12, 18), 1)) # RR 12-18 br/min
        hrv_data.append(random.randint(20, 80))                  # HRV 20-80 ms
        sleep_duration_data.append(round(random.uniform(5, 9), 1)) # Sleep 5-9 hours
        steps_data.append(random.randint(2000, 15000))           # Steps 2k-15k
        temperature_data.append(round(random.uniform(-1.0, 1.0), 2)) # Temp deviation -1 to +1°C
    
    patient['dates'] = dates
    patient['heart_rate_data'] = heart_rate_data
    patient['respiratory_data'] = respiratory_data
    patient['hrv_data'] = hrv_data
    patient['sleep_duration_data'] = sleep_duration_data
    patient['steps_data'] = steps_data
    patient['temperature_data'] = temperature_data
    
    return render_template('patient_detail.html', patient=patient, active_tab='overview')


@app.route('/patient/<patient_id>/model-lab')
def model_lab(patient_id):
    """Show the Model Lab — ML model selection, training, and results."""
    patients = load_patient_data()
    patient  = next((p for p in patients if p['id'] == patient_id), None)

    if not patient:
        return "Patient not found", 404

    # ── Determine data source from patient ID pattern ────────────────────────
    # PT-2xxx → PPMI (Parkinson's), everything else → Oura (liver/HE)
    if patient_id.startswith('PT-2'):
        ds_key       = 'ppmi'
        source_label = 'PPMI Dataset'
        data_points  = 5   # visit timepoints: 0, 6, 12, 24, 36 months
    else:
        ds_key       = 'oura'
        source_label = 'Oura V2 API'
        data_points  = 30

    # ── Feature groups from the registry ────────────────────────────────────
    try:
        from data.feature_registry import get_feature_groups_for_source
        from data.base import DataSource
        source_enum    = DataSource.PPMI if ds_key == 'ppmi' else DataSource.OURA
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

    if ds_key == 'ppmi':
        feature_importance = [
            {'name': 'Baseline UPDRS',  'importance': 0.34},
            {'name': 'GBA Mutation',    'importance': 0.22},
            {'name': 'DaTscan SBR',     'importance': 0.18},
            {'name': 'CSF Alpha-syn',   'importance': 0.14},
            {'name': 'LRRK2 Mutation',  'importance': 0.07},
            {'name': 'Epworth Sleep',   'importance': 0.05},
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

    # Detect data source (PT-2xxx → PPMI, all others → Oura)
    ds_key       = 'ppmi' if patient_id.startswith('PT-2') else 'oura'
    source_label = 'PPMI Dataset' if ds_key == 'ppmi' else 'Oura V2 API'

    # Load feature registry
    try:
        from data.feature_registry import get_feature_groups_for_source, get_features_for_source
        from data.base import DataSource
        source_enum   = DataSource.PPMI if ds_key == 'ppmi' else DataSource.OURA
        feature_groups = get_feature_groups_for_source(source_enum)
        all_features   = get_features_for_source(source_enum)
    except Exception:
        feature_groups = {}
        all_features   = []

    # Generate 90 days of synthetic data (same seed as /api/run-experiment)
    import numpy as np
    N_DAYS = 90
    seed   = abs(hash(patient_id)) % (2 ** 31)
    rng    = np.random.default_rng(seed)
    idx    = pd.date_range(end='2024-12-31', periods=N_DAYS, freq='D')

    col_data = {
        'rem_sleep_pct':       rng.uniform(10, 30, N_DAYS),
        'deep_sleep_pct':      rng.uniform(8,  25, N_DAYS),
        'sleep_latency':       rng.uniform(5,  45, N_DAYS),
        'hrv_balance':         rng.uniform(20, 80, N_DAYS),
        'body_temp_deviation': rng.uniform(-1.0, 1.0, N_DAYS),
        'resting_hr':          rng.uniform(50, 75, N_DAYS),
        'step_count':          rng.uniform(2000, 15000, N_DAYS),
        'inactivity_alerts':   rng.uniform(0, 10, N_DAYS),
        'baseline_updrs':      rng.uniform(10, 60, N_DAYS),
        'csf_alpha_synuclein': rng.uniform(1000, 3000, N_DAYS),
        'amyloid_beta':        rng.uniform(500,  1500, N_DAYS),
        'total_tau':           rng.uniform(100,  400, N_DAYS),
        'gba_mutation':        rng.integers(0, 3, N_DAYS).astype(float),
        'lrrk2_mutation':      rng.integers(0, 2, N_DAYS).astype(float),
        'apoe_status':         rng.integers(0, 3, N_DAYS).astype(float),
        'epworth_sleep':       rng.uniform(0, 18, N_DAYS),
        'schwab_england_adl':  rng.uniform(50, 100, N_DAYS),
        'datscan':             rng.uniform(0.5, 3.5, N_DAYS),
    }

    dates = [d.strftime('%b %d') for d in idx]

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


@app.route('/patient/<patient_id>/tournament')
def tournament(patient_id):
    """Tournament tab — placeholder."""
    patients = load_patient_data()
    patient  = next((p for p in patients if p['id'] == patient_id), None)
    if not patient:
        return "Patient not found", 404
    return render_template('tournament.html', patient=patient, active_tab='tournament')


@app.route('/patient/<patient_id>/ai-assistant')
def ai_assistant(patient_id):
    """AI Assistant tab — placeholder."""
    patients = load_patient_data()
    patient  = next((p for p in patients if p['id'] == patient_id), None)
    if not patient:
        return "Patient not found", 404
    return render_template('ai_assistant.html', patient=patient, active_tab='ai-assistant')


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    app.run(debug=debug, host='0.0.0.0', port=port)
