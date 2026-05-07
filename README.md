# JupyterHealth Clinical Workbench

> Clinician-facing web workbench for wearable + clinical data research, ML disease progression prediction, and explainable AI

A browser-based platform that enables clinical researchers to monitor patient cohorts via wearable and clinical data, run ML experiments for disease progression prediction, and explore data directly from or alongside a Jupyter notebook instance.

---

## [Live Demo](https://jupyterhealth-clinical-workbench.onrender.com/)

---

## Clinical Context

Built for Dr. Adam Buckholz's Hepatic Encephalopathy (HE) sleep biomarkers study at Cornell Medicine. The study tracks ~140 cirrhosis patients (target 150) using Oura Ring wearables, with the hypothesis that subtle REM sleep and circadian rhythm changes can detect covert HE earlier than standard pen-and-paper cognitive tests.

**Key biomarkers tracked**: REM Sleep %, Deep Sleep %, HRV Balance, Body Temperature Deviation, Resting HR, Step Count, Sleep Latency, SpO₂

---

## Features

### Cohort Dashboard

- **Patient table** — all patients with 14-day sleep sparklines, risk badges, and week-over-week efficiency trend
- **Enrollment stats** — enrolled / target / years of data pulled from `study_config.json` (no code change needed to update)
- **Cohort Data Explorer** — two-column layout:
  - Left: patient multi-select with risk filter chips (All / High / Medium / Low), Select All / None
  - Right: Data Preview tab (sortable sleep table, highlights selected rows) + Notebook Builder tab
- **Notebook Builder** — choose analysis type (sleep comparison, cohort stats, risk scatter, etc.), date range, metrics; live Python code preview; create a new notebook or inject a cell into an existing one via the Jupyter REST API

### Patient Views

Two tabs in the patient nav bar: **Biometrics** and **Risk Analysis**. An Export button opens a print-optimized report.

**Biometrics** (`/patient/<id>`) — two sub-views toggled within the page:

| Sub-view | Description |
|----------|-------------|
| **Overview** | Key Biometrics chart (HR / HRV / Sleep) with 7d / 14d / All time window toggle and per-point hover tooltip showing exact values |
| **Sleep Detail** | Full sleep architecture breakdown: Sleep Architecture (REM / Deep / Light / Awake), Time in Bed vs Total Sleep Time, Sleep Efficiency, Sleep Latency, WASO, SpO₂; confounder flag overlay (amber markers on flagged nights) |

**Risk Analysis** (`/patient/<id>/risk-analysis-lab`) — ML experiment runner:

- Feature group selection (Sleep Architecture, Physiological, Activity)
- Model picker: XGBoost, Random Forest, LSTM, TFT
- Hyperparameter fields (learning rate, max depth, n estimators)
- Results: AUC-ROC, Precision, Recall, F1, feature importance bar chart
- Saved experiment history (compare runs)

**Export Report** (`/patient/<id>/report`) — print-optimized patient summary; downloadable as Word-compatible `.doc`

### Confounder Tracking

Per-night flags (sleep apnea, narcotics, alcohol use) stored in `demo_data/confounders.json`, keyed by ISO date (YYYY-MM-DD). Flagged nights display amber markers below the x-axis on all sleep detail charts.

### Research Chat Assistant

Rule-based assistant in the persistent right-side panel. Answers questions about the study, patient metrics, and clinical thresholds. (Swap for LLM in Phase 2.)

---

## Tech Stack

| Layer | Choice |
|-------|--------|
| Backend | Python 3.12, Flask |
| Frontend | Vanilla HTML5 / CSS3 / JS — no React, Vue, or npm |
| Charts | Custom SVG drawn client-side — no Plotly in production |
| Data | pandas, NumPy, openpyxl |
| ML | scikit-learn, XGBoost, PyTorch (TFT/LSTM) |
| Persistence | JSON flat files — no database |
| Notebooks | Jupyter (REST API integration for cell injection) |
| Deployment | Render (free tier), Gunicorn |

---

## Project Structure

```
oura-clinical-workbench/
├── app.py                           # All Flask routes — single file
├── config.py                        # App-level configuration
├── requirements.txt
├── render.yaml                      # Render deployment config
│
├── data/                            # Data abstraction layer
│   ├── base.py                      # PatientTimeSeries dataclass + DataSource enum
│   ├── standard_wearable_adapter.py # OMH/IEEE wearable loader (lru_cache)
│   ├── synthea_adapter.py           # Synthea FHIR JSON bundle loader
│   ├── oura_adapter.py              # Oura Ring V2 API stub (Phase 2)
│   ├── open_wearables_adapter.py    # Open Wearables unified API stub (Q1 2026)
│   ├── feature_registry.py          # Maps DataSource → feature groups for Model Lab
│   └── conditions_dict.json         # SNOMED/ICD-10 condition dictionary (~400 entries)
│
├── models/                          # ML model layer
│   ├── experiment.py                # Experiment config, training, result storage
│   ├── xgboost_model.py
│   ├── random_forest_model.py
│   ├── lstm_model.py
│   ├── tft_model.py
│   └── explainability.py            # SHAP + LLM rationale (Phase 2)
│
├── templates/                       # Jinja2 HTML templates
│   ├── base_layout.html             # Global shell: chat panel, topnav slot
│   ├── base.html                    # Patient shell: extends base_layout; tabs
│   ├── dashboard.html               # Cohort dashboard (standalone)
│   ├── patient_detail.html          # Biometrics + Sleep Detail view
│   ├── cohort_data_explorer.html           # Multi-signal chart overlay
│   ├── risk_analysis_lab.html               # ML experiment runner
│   ├── ai_assistant.html            # SHAP rationale shell (Phase 2)
│   └── patient_report.html          # Print-optimized report (standalone)
│
├── demo_data/                       # Synthetic/fake data — safe to commit
│   ├── demo_omh_ieee/               # OMH/IEEE wearable records (PT-3001–PT-3008)
│   ├── demo_synthea/                # Synthea FHIR bundles
│   ├── study_config.json            # Enrollment numbers + conditions of interest
│   └── confounders.json             # Per-patient per-night confounder flags
│
└── notebooks/                       # .ipynb files gitignored (PHI risk)
```

---

## Quick Start (Local Development)

```bash
git clone https://github.com/dicristea/oura-clinical-workbench.git
cd oura-clinical-workbench
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open [http://localhost:5000](http://localhost:5000).

### Optional: Jupyter integration

To use the "Add to Notebook" and Notebook Builder features, run a Jupyter server alongside the app:

```bash
jupyter notebook --NotebookApp.token='' --NotebookApp.password='' --port=8888
```

Set `JUPYTER_BASE_URL=http://localhost:8888` if running on a different port.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OURA_API_TOKEN` | Phase 2 | Oura Ring V2 API token |
| `PATIENT_MRNS` | Phase 2 | Comma-separated patient MRNs |
| `FLOWSHEET_FILE` | Phase 2 | Path to flowsheet Excel file |
| `OPEN_WEARABLES_API_KEY` | Q1 2026 | Open Wearables unified API key |
| `JUPYTER_BASE_URL` | No | Jupyter server URL (default: `http://localhost:8888`) |
| `LLM_API_KEY` | Phase 2 | API key for LLM rationale generation |
| `PORT` | No | Server port (default: 5000) |

---

## Security — PHI / HIPAA

**Never commit:**
- `data.xlsx` or any file containing real patient names, MRNs, or identifiable data
- `.env` files (API tokens)
- `.ipynb` files (gitignored — PHI risk from cell outputs)

**Safe to commit:**
- `demo_data/` — synthetic and fake data only
- `.env.example` — placeholder values only
- Code, templates, documentation

No authentication exists in the current build. Do not connect real PHI data until auth is added (see Phase 2 backlog in `HANDOFF.md`).

---

## Testing

```bash
pytest tests/
```

---

## References

- [Oura V2 API](https://cloud.ouraring.com/v2/docs)
- [Open Wearables](https://www.openwearables.io/)
- [OpenMHealth data standard](https://www.openmhealth.org/)
- [Synthea](https://github.com/synthetichealth/synthea)
- SHAP: Lundberg & Lee, 2017 (NeurIPS)
- Temporal Fusion Transformers: Lim et al., 2021 (Int. J. Forecasting)
- Explorable Explainability: Solano-Kamaiko et al., 2024 (CHI '24)

---

## License

This project handles Protected Health Information (PHI). Ensure compliance with HIPAA regulations, IRB requirements, applicable data use agreements, and patient consent requirements before connecting any real patient data.
