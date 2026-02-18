# CLAUDE.md — JupyterHealth Clinical Workbench

> Persistent context for Claude Code. This file is read automatically at the start of every session.

## Project Overview

This is a **clinician-facing web workbench** built on the JupyterHealth platform. It enables clinical researchers to monitor patients via wearable + clinical data, run ML models for disease progression prediction, and get explainable AI rationales — all through a browser-based UI.

The workbench supports **three disease contexts** through a unified interface:

| Context       | Data Source                       | Condition                          | PI / Study                          |
| ------------- | --------------------------------- | ---------------------------------- | ----------------------------------- |
| Liver Disease | Oura Ring V2 API + EHR Flowsheets | Hepatic Encephalopathy (cirrhosis) | Dr. Adam Buckholz, Cornell/Columbia |
| Parkinson's   | PPMI Dataset (LONI IDA)           | PD Motor Progression               | ML for Health course project        |
| Synthetic     | Synthea FHIR Bundles              | Multiple (demo/testing)            | Future development                  |

## Clinical Context

### Liver Disease / Hepatic Encephalopathy (Primary Study)

- **Condition**: Hepatic encephalopathy — confusion caused by liver failing to clear toxins (ammonia)
- **Patient cohort**: ~140 patients enrolled across Cornell/Columbia (target 150)
- **Data collected**: 20,000+ days of Oura ring data over 6 years
- **Key finding from pilot**: Subtle REM sleep and circadian rhythm changes in covert disease patients
- **Clinical gap**: Only 10% of clinicians perform proper screening; current methods use cumbersome pen-and-paper cognitive tests
- **Goal**: Use digital biomarkers (sleep architecture, HRV, temperature) to flag at-risk patients early
- **Important features**: REM Sleep %, Deep Sleep %, HRV Balance, Body Temp Deviation, Resting HR, Step Count, Inactivity Alerts, Sleep Latency
- **Confounders to watch**: sleep apnea, narcotics use, alcohol use

### Parkinson's Disease

- **Dataset**: PPMI (Parkinson's Progression Markers Initiative) from LONI IDA
- **Cohort**: 423 de novo PD patients, 196 healthy controls, 65 prodromal participants
- **Target variable**: MDS-UPDRS Part III (Motor Examination Score)
- **Key features**: Baseline UPDRS, GBA Mutation Status, CSF Alpha-synuclein, Amyloid-beta, Total Tau, APOE/LRRK2 status, Epworth Sleepiness Scale, Schwab & England ADL
- **Visit schedule**: 0, 6, 12, 24, 36 months (longitudinal, irregular)
- **Prediction horizon**: 48 months (regression + rapid vs. slow progressor classification)
- **Models**: Temporal Fusion Transformer (primary), XGBoost (baseline), LSTM/GRU (temporal baseline)
- **Novelty**: TFT for PD forecasting, LLM-based SHAP explanation pipeline, Cognitive Match Score for hallucination detection

## Tech Stack

- **Backend**: Python 3.12, Flask
- **Frontend**: HTML5, CSS3, vanilla JavaScript (no React/Vue — keep it simple)
- **Data**: Pandas, NumPy, openpyxl
- **ML**: scikit-learn, XGBoost, PyTorch (for TFT/LSTM)
- **Explainability**: SHAP, LLM API (Llama 3 or GPT-4o via API)
- **Charts**: Custom SVG in templates (no Plotly in production — keep lightweight)
- **Deployment**: Render (free tier), Gunicorn
- **Notebooks**: Jupyter (HF-Notebook/ directory for exploratory analysis)

## Architecture

```
oura-clinical-workbench/
├── CLAUDE.md                        # THIS FILE - Claude Code context
├── app.py                           # Flask application + all routes
├── config.py                        # App-level configuration
├── requirements.txt                 # Production dependencies
├── requirements_deploy.txt          # Deployment-only dependencies
├── render.yaml                      # Render deployment config
│
├── data/                            # Data abstraction layer
│   ├── __init__.py
│   ├── base.py                      # PatientTimeSeries dataclass + DataSource enum
│   ├── oura_adapter.py              # Oura Ring V2 API + flowsheet Excel loader
│   ├── ppmi_adapter.py              # PPMI LONI CSV loader (de novo cohort)
│   ├── synthea_adapter.py           # Synthea FHIR JSON bundle loader
│   └── feature_registry.py          # Maps data source → available features + groups
│
├── models/                          # ML model layer
│   ├── __init__.py
│   ├── experiment.py                # Experiment config, training, result storage
│   ├── xgboost_model.py             # XGBoost classifier/regressor wrapper
│   ├── random_forest_model.py       # Random Forest wrapper
│   ├── lstm_model.py                # LSTM/GRU temporal model
│   ├── tft_model.py                 # Temporal Fusion Transformer
│   └── explainability.py            # SHAP computation + LLM rationale generation
│
├── templates/                       # Jinja2 HTML templates
│   ├── base.html                    # Shared layout (nav bar, patient header)
│   ├── dashboard.html               # Main coordinator dashboard (patient list)
│   ├── patient_detail.html          # Overview tab (existing biometrics view)
│   ├── data_explorer.html           # Raw signal overlay + data browsing
│   ├── model_lab.html               # ML model selection, training, results
│   ├── tournament.html              # Side-by-side model comparison
│   └── ai_assistant.html            # LLM explainability / Trust Workbench
│
├── static/                          # Static assets (CSS, JS if extracted)
│   └── style.css                    # Shared styles (optional — currently inline)
│
├── demo_data/                       # Sample/demo data (safe to commit)
│   ├── demo_oura.xlsx               # Fake Oura patient data
│   ├── demo_ppmi/                   # Sample PPMI CSVs (synthetic or subset)
│   └── demo_synthea/                # Sample Synthea FHIR bundles
│
├── HF-Notebook/                     # Original Jupyter notebooks (exploratory)
│   ├── config.py                    # Notebook-specific env loader
│   ├── vis.py                       # Plotly visualization helpers
│   ├── requirements.txt             # Notebook dependencies
│   ├── .env.example                 # Template for API tokens
│   └── README.md                    # Notebook documentation
│
└── tests/                           # Unit tests
    ├── test_data_adapters.py
    ├── test_models.py
    └── test_routes.py
```

## Key Design Decisions

### 1. Data Adapter Pattern

All data sources output `PatientTimeSeries` objects. The UI never knows or cares whether data came from Oura, PPMI, or Synthea. Each adapter handles its own parsing, cleaning, and normalization.

```python
@dataclass
class PatientTimeSeries:
    patient_id: str
    data_source: DataSource          # OURA | PPMI | SYNTHEA
    static_features: dict            # e.g., {"GBA_mutation": True, "age": 65}
    time_series: pd.DataFrame        # DatetimeIndex, columns = feature names
    metadata: dict                   # e.g., {"cohort": "de_novo", "risk_level": "high"}
    feature_groups: dict[str, list]  # e.g., {"Genetic": ["GBA", "LRRK2", "APOE"]}
```

### 2. Feature Registry

Each data source registers its available features and how they group in the UI:

- **Oura**: Sleep Architecture (REM %, Deep %, Latency), Physiological (HRV, Body Temp, Resting HR), Activity (Steps, Inactivity, Sedentary Time)
- **PPMI**: Genetic (GBA, LRRK2, APOE), Biofluid (CSF alpha-syn, amyloid-beta, tau), Clinical (Baseline UPDRS, Epworth, Schwab & England), Imaging (DaTscan)
- **Synthea**: Varies by generated condition

### 3. Model Lab is Disease-Agnostic

The Model Lab template renders whatever features/models are available for the current patient's data source. The feature checkboxes, model dropdown, hyperparameter fields, results cards, and feature importance chart all populate dynamically from the experiment config.

### 4. No Client-Side Frameworks

The frontend uses vanilla HTML/CSS/JS with Jinja2 templating. SVG charts are generated server-side or with minimal client-side JS. This keeps the stack simple and avoids build tooling. Plotly is used only in the Jupyter notebooks, not in the production Flask app.

### 5. Explainability Pipeline (Phase 2)

```
Model Prediction → SHAP Values → LLM (Llama 3 / GPT-4o) → Clinical Rationale Text
                                                          → Cognitive Match Score (hallucination check)
```

The Cognitive Match Score = IR Precision of generated text against top-3 SHAP features. If the LLM mentions a feature not in the top-3 SHAP contributors, the score drops. Below a threshold → output "Uncertain" instead of a potentially hallucinated explanation.

## Routes (Flask)

| Route                               | Template            | Description                                          |
| ----------------------------------- | ------------------- | ---------------------------------------------------- |
| `GET /`                             | dashboard.html      | Patient list with sparklines, status badges, filters |
| `GET /patient/<id>`                 | patient_detail.html | Overview tab — biometric time series                 |
| `GET /patient/<id>/data-explorer`   | data_explorer.html  | Raw signal overlay, feature browsing                 |
| `GET /patient/<id>/model-lab`       | model_lab.html      | ML model selection, training, results                |
| `GET /patient/<id>/tournament`      | tournament.html     | Side-by-side model comparison table                  |
| `GET /patient/<id>/ai-assistant`    | ai_assistant.html   | LLM explainability interface                         |
| `POST /api/run-experiment`          | JSON                | Run a model experiment, return results               |
| `GET /api/experiments/<patient_id>` | JSON                | List saved experiments for a patient                 |
| `GET /api/patients`                 | JSON                | Patient list data                                    |

## UI Layout (from mockup screenshots)

### Patient View — Top Navigation Bar

```
← Patient PT-XXXX  [High Risk]    | Overview | Data Explorer | Model Lab | Tournament | AI Assistant |    [Export Findings] [AB]
```

### Model Lab Layout (Left Panel)

```
Analysis Window: [7d] [14d] [30d] [90d] [All]
Data points: N (Source API)

Input Features:
  [Feature Group 1]
    ☑ Feature A
    ☑ Feature B
    ☐ Feature C
  [Feature Group 2]
    ☑ Feature D
    ☐ Feature E

Raw Signal Overlay: [small chart showing selected signals]
```

### Model Lab Layout (Right Panel)

```
Select Model: [Dropdown: XGBoost / Random Forest / LSTM / TFT]    [Run Analysis]
Hyperparameters: [Learning Rate] [Max Depth] [N Estimators] [Min Child Weight]

Results: [AUC-ROC: 0.91] [Precision: 0.87] [Recall: 0.84] [F1: 0.85]

Feature Importance: [horizontal bar chart]

Prediction Confidence Over Time: [line chart with alert marker]

Saved Experiments: [table with Model, Features, AUC, F1, Actions]
```

## Data Files — CRITICAL SECURITY RULES

### NEVER commit these:

- `data.xlsx` (real patient PHI)
- `.env` files (API tokens, MRNs)
- Any file with real patient names, MRNs, or identifiable data
- PPMI data files downloaded from LONI (governed by DUA)

### Safe to commit:

- `demo_data.xlsx` / `demo_data/` (synthetic/fake data only)
- `.env.example` (template with placeholder values)
- Code, templates, documentation

## Coding Conventions

- **Python**: Use type hints, dataclasses, f-strings. Follow PEP 8.
- **HTML/CSS**: Inline styles are OK in templates for now; extract to `static/style.css` if they grow.
- **JavaScript**: Vanilla JS only. No npm, no bundler. Keep chart logic in `<script>` tags in templates.
- **Error handling**: All data loading wrapped in try/except. Graceful fallback to demo data.
- **Testing**: pytest for unit tests. Test data adapters with small fixture CSVs.

## Common Commands

```bash
# Run locally
python app.py

# Run with specific port
PORT=8080 python app.py

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Deploy (auto via Render on git push)
git push origin main
```

## Environment Variables

| Variable         | Required          | Description                               |
| ---------------- | ----------------- | ----------------------------------------- |
| `OURA_API_TOKEN` | For Oura patients | Oura Ring API token                       |
| `PATIENT_MRNS`   | For Oura patients | Comma-separated MRNs                      |
| `FLOWSHEET_FILE` | For Oura patients | Path to flowsheet Excel                   |
| `PPMI_DATA_DIR`  | For PPMI patients | Path to PPMI CSV directory                |
| `FLASK_DEBUG`    | No                | Enable debug mode (default: True locally) |
| `PORT`           | No                | Server port (default: 5000)               |
| `LLM_API_KEY`    | For AI Assistant  | API key for LLM rationale generation      |

## Known Issues / TODOs

- [ ] Dashboard uses `random.randint` for sparkline data — replace with real Oura API calls
- [ ] Patient detail generates fake metric data — wire up real data adapters
- [ ] No authentication — add before any real PHI touches the app
- [ ] PPMI adapter not yet implemented
- [ ] Synthea adapter not yet implemented
- [ ] Model Lab template not yet built
- [ ] TFT model not yet implemented (depends on PyTorch + pytorch-forecasting)
- [ ] Explainability pipeline not yet implemented
- [ ] No unit tests yet

## References

- Oura V2 API: https://cloud.ouraring.com/v2/docs
- PPMI: https://www.ppmi-info.org/
- Temporal Fusion Transformers: Lim et al., 2021 (Int. J. Forecasting)
- SHAP: Lundberg & Lee, 2017 (NeurIPS)
- Explorable Explainability: Solano-Kamaiko et al., 2024 (CHI '24)
- OpenMHealth data standard: https://www.openmhealth.org/
