"""
Clinical Coordinator Dashboard - Flask App
Exact mockup implementation
"""

from flask import Flask, render_template, jsonify, request
import pandas as pd
from datetime import datetime, timedelta
import random

app = Flask(__name__)


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
    
    return render_template('patient_detail.html', patient=patient)


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    app.run(debug=debug, host='0.0.0.0', port=port)
