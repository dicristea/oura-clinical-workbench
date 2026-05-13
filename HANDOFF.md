# Project Handoff — JupyterHealth Clinical Workbench

> **How to use this file**
> Update the Open section at the end of every Claude session. The Frozen section only changes when a deliberate architectural decision is overturned — treat it as immutable by default. This file is the single source of truth for project state.

---

## ━━ FROZEN ━━
*Decisions that are final. Do not re-litigate these unless there is a strong reason and the user explicitly agrees.*

### Study Context
- **PI**: Dr. Adam Buckholz — Cornell Medicine
- **Disease**: Hepatic Encephalopathy (HE) in cirrhosis patients
- **Cohort**: ~140 enrolled, 150 target, 20,000+ days of Oura ring data over 6 years
- **Core hypothesis**: Wearable sleep biomarkers (REM%, HRV, circadian rhythm) detect *covert HE* earlier than pen-and-paper cognitive tests
- **Key confounders to track**: sleep apnea, narcotics, alcohol use

### Tech Stack (final, do not add build tooling)
| Layer | Choice |
|---|---|
| Backend | Python 3.12 + Flask |
| Frontend | Vanilla HTML/CSS/JS — **no React, Vue, or npm** |
| Charts | Custom SVG drawn client-side — **no Plotly in production templates** |
| Data | pandas, numpy, openpyxl |
| ML | scikit-learn, XGBoost, PyTorch (TFT/LSTM) |
| Persistence | JSON flat files (no database) |
| Notebooks | Jupyter (HF-Notebook/ dir, exploratory only) |

### File Layout (do not rename or restructure)
```
app.py                          # All Flask routes — single file, keep it that way
templates/
  base_layout.html              # Global shell: chat panel, topnav slot
  base.html                     # Patient shell: extends base_layout; subbar + tabs
  dashboard.html                # Cohort dashboard (standalone, does NOT extend base.html)
  patient_detail.html           # Biometrics + Sleep Detail view
  patient_report.html           # Print-optimized report (standalone, no nav)
  cohort_data_explorer.html            # Multi-signal chart overlay
  risk_analysis_lab.html                # ML experiment runner
  ai_assistant.html             # SHAP + LLM rationale (Phase 2 shell)
data/
  base.py                       # PatientTimeSeries dataclass + DataSource enum
  standard_wearable_adapter.py  # OMH/IEEE wearable loader
  synthea_adapter.py            # Synthea FHIR JSON loader
  feature_registry.py           # Maps DataSource → feature groups for Model Lab
  conditions_dict.json          # SNOMED/ICD-10 condition dictionary (~400 entries)
demo_data/
  demo_omh_ieee/                # OMH/IEEE wearable records (8 patients: PT-3001–PT-3008)
  demo_synthea/                 # Synthea FHIR bundles (PT-3001–PT-300N, grows with Add Patient)
  study_config.json             # Persists conditions of interest across sessions
  confounders.json              # Per-patient per-night confounder flags
notebooks/
  .gitkeep                      # Directory tracked; .ipynb files are gitignored (PHI)
```

### Key Constants (app.py top of file)
```python
STUDY_CONFIG_PATH             = 'demo_data/study_config.json'
CONDITIONS_DICT_PATH          = 'data/conditions_dict.json'
STANDARD_WEARABLE_DATASET_DIR = 'demo_data/demo_omh_ieee'
CONFOUNDERS_PATH              = 'demo_data/confounders.json'
NOTEBOOKS_DIR                 = 'notebooks'
JUPYTER_BASE_URL              = os.environ.get('JUPYTER_BASE_URL', 'http://localhost:8888')
```

### Route Map (all in app.py)
| Method | Route | Description |
|---|---|---|
| GET | `/` | Cohort dashboard |
| GET | `/patient/<id>` | Biometrics + Sleep Detail view |
| GET | `/patient/<id>/risk-analysis-lab` | ML experiment runner |
| GET | `/patient/<id>/cohort-data-explorer` | Multi-signal overlay |
| GET | `/patient/<id>/report` | Print-optimized patient report |
| GET | `/patient/<id>/report.docx` | Same report as Word-compatible .doc download |
| GET | `/patient/<id>/ai-assistant` | SHAP rationale shell (Phase 2) |
| POST | `/api/chat` | Rule-based research assistant (swap for LLM in Phase 2) |
| GET | `/api/patients` | Patient list JSON (⚠ broken — see Open) |
| GET | `/api/conditions/search` | SNOMED/ICD autocomplete |
| POST | `/api/study/conditions` | Add condition to study |
| DELETE | `/api/study/conditions/<name>` | Remove tracked condition |
| POST | `/api/run-experiment` | Run ML experiment, return metrics |
| POST | `/api/generate-patient` | Generate + save synthetic Synthea patient |
| GET | `/api/confounders/<id>` | Get confounder flags for patient |
| POST | `/api/confounders/<id>` | Set/clear flag for a specific night |
| DELETE | `/api/confounders/<id>/<date>` | Remove flag for a specific night |
| GET | `/api/notebooks` | List .ipynb files in notebooks/ |
| POST | `/api/notebooks/new` | Create pre-loaded cohort analysis notebook |

### Patient ID Conventions (final)
- `PT-XXXX` (non-3xxx) → Oura Ring / EHR Flowsheet patients (liver disease study)
- `PT-3XXX` → Synthea FHIR synthetic patients (metabolic demo cohort)
- The `data_source` field on the patient dict drives which adapter and template logic to use

### Data Patterns (final)
- **Seeded demo data**: `random.seed(hash(patient_id))` before generating per-patient arrays in `patient_detail()`. The seed must be consumed in the same call order every time or values will differ. Use `random.Random(abs(hash(patient_id)) % (2**31))` for isolated seeds that don't affect global state.
- **Real OMH/IEEE data**: loaded via `StandardWearableAdapter`, cached via `@lru_cache(maxsize=1)`. Cache is **not invalidated** on new file adds — server restart required.
- **Sleep summaries**: `_compute_sleep_summaries(patients)` returns 14-day averages + week-over-week efficiency trend (`efficiency_trend`, `trend_dir`) for every patient. Passed to dashboard as `sleep_summaries` (list) and `sleep_summary_map` (dict keyed by patient ID).
- **Confounder flags**: stored as `{patient_id: {date_str: [flag_list]}}` in `confounders.json`. Date strings match the `"%b %d"` format used in patient detail (e.g. `"Dec 01"`).

### Template Inheritance (final)
```
base_layout.html          ← global shell: chat panel, {% block topnav %}, {% block main %}
  └── base.html           ← patient shell: topnav + patient subbar; {% block page_head %}, {% block content %}
        └── patient_detail.html
        └── risk_analysis_lab.html
        └── cohort_data_explorer.html
        └── ai_assistant.html
  └── dashboard.html      ← extends base_layout directly (no patient subbar)
patient_report.html       ← standalone (no extends; print-only page)
```

### Security Rules (non-negotiable)
- Never commit `data.xlsx`, `.env`, or any file containing real patient identifiers
- Never commit `.ipynb` files (PHI risk — gitignored by `*.ipynb` in `.gitignore`)
- No authentication exists — do not connect real PHI data until auth is added
- `demo_data/` is safe to commit (synthetic/fake only)

---

## ━━ OPEN ━━
*Current session results, known bugs, and next tasks. Update this section at the end of every session.*

### Last Session — 2026-05-07 (session 4 — Dr. Buckholz presentation prep)

**Renames / cleanup shipped:**
1. **`data_explorer` → `cohort_data_explorer`** — Template renamed (`templates/data_explorer.html` → `templates/cohort_data_explorer.html`), route updated in `app.py`, nav link in `base.html`, all docs updated. (A sed double-pass bug created `cohort_cohort_data_explorer` in some docs — fixed with a second sed pass.)
2. **`model_lab` → `risk_analysis_lab`** — Template renamed, route `/patient/<id>/model-lab` → `/patient/<id>/risk-analysis-lab`, function `model_lab` → `risk_analysis_lab`, nav link and aria-selected in `base.html`, all docs updated.
3. **README.md rewrite** — Replaced stale/inaccurate README with accurate current state: Cornell Medicine context, actual features (Key Biometrics with window+hover, Sleep Detail, Cohort Data Explorer, Notebook Builder, auto-detected confounders, SHAP Feature Importance), accurate file tree, Jupyter integration section, full env vars table.

**New features shipped:**
4. **Individual biometric charts (5 charts)** — Below the Key Biometrics overview chart in `patient_detail.html`, added a `.bio-grid` (2-column CSS grid) with cards for: Resting HR, HRV rMSSD, Sleep Duration, Body Temp Deviation, Daily Steps. Each card has `drawBioLineChart()` — same window/hover/crosshair pattern as the combined chart, with a clinically-relevant reference line (50ms HRV, 7h sleep, 7,500 steps, 0°C baseline). `drawAllBioCharts()` replaces `drawCombinedChart()` at all call sites.
5. **Auto-detected confounders** — `_auto_detect_confounders(patient: dict) -> dict` added to `app.py`. Threshold-based: SpO₂ < 93% → "Low SpO₂ (possible apnea)"; temp deviation > 0.8°C → "Elevated temp deviation (fever / alcohol)"; HRV < 50% of patient median → "Low HRV (acute stressor)"; latency > 45min AND WASO > 60min → "Fragmented sleep (behavioural disruption)". Returns `{iso_date: [reasons]}`.
6. **Flagged Nights panel in Risk Analysis Lab** — Sidebar shows auto-detected nights with date, reason chips, and an "Exclude flagged nights from training" checkbox. `runAnalysis()` passes `excluded_dates` in the POST body; `api_run_experiment()` reads it and includes `excluded_nights` in the response. No dashboard badges (intentional — avoids notification overload for clinicians).
7. **SHAP prefix on Feature Importance** — Label changed to "SHAP Feature Importance" in `risk_analysis_lab.html`.
8. **Removed Raw Signal Overlay** — Entire sidebar card (CSS, HTML, `drawOverlayChart()`, `OVERLAY_COLORS`, all call sites) removed from `risk_analysis_lab.html`. Declutters the lab; signal overlay lives in Cohort Data Explorer.
9. **Fixed condition modal not opening** — `openCondModal` was defined inside an IIFE in `dashboard.html` but not exposed to `window`. Added `window.openCondModal = openCondModal` alongside the existing `window.closeModal` and `window.handleOverlayClick` assignments.
10. **Demo patient PT-1042 "Synthea Demo"** — `DEMO_HE_PATIENT_ID = 'PT-1042'` constant added. `_make_demo_he_patient()` returns a fully-wired patient dict (risk_level='high', data_source='oura', conditions='Covert Hepatic Encephalopathy', name='Synthea Demo', model_status_label='Sleep Risk Signal Detected'). `_demo_he_overrides()` provides hand-crafted 14-day arrays telling a coherent HE deterioration story:
    - Sleep efficiency: 87.8% → 66.3% (declining)
    - REM%: 21% → 11% (declining)
    - HRV: 16–31ms (consistently suppressed)
    - SpO₂: dips to 91.8% on day 8 → triggers SpO₂ flag
    - Temp deviation: spikes to +0.93°C on day 8 → triggers fever/alcohol flag
    - Latency > 45min and WASO > 60min on days 8 and 14 → triggers fragmented sleep flag on both nights
    - Steps: 7,200 → 2,100 (halves over the period, visible in individual chart)
    - PT-1042 is inserted at position 0 in `load_patient_data()` (both try and except paths) so it always appears first on the dashboard.

**Files changed this session:**
- `app.py` — `DEMO_HE_PATIENT_ID` constant; `_auto_detect_confounders()`; `_make_demo_he_patient()`; `_demo_he_overrides()`; `load_patient_data()` inserts PT-1042 first; `_compute_patient_view_data()` applies overrides and adds `auto_confounders`; `risk_analysis_lab()` route refactored to use `_compute_patient_view_data()` and pass `flagged_nights`; `api_run_experiment()` reads `excluded_dates`; route + function rename model_lab→risk_analysis_lab
- `templates/patient_detail.html` — `.bio-grid` / `.bio-grid-card` / `.bio-grid-area` CSS; 5 chart card HTML blocks; `_fmtBio()`; `drawBioLineChart()`; `drawAllBioCharts()`; `DOMContentLoaded` and `setBioWindow` updated
- `templates/risk_analysis_lab.html` — Removed Raw Signal Overlay; added SHAP prefix; added Flagged Nights panel (CSS + HTML + JS payload update)
- `templates/dashboard.html` — `window.openCondModal = openCondModal` fix
- `templates/base.html` — nav link and aria-selected: model-lab → risk-analysis-lab
- `templates/data_explorer.html` → `templates/cohort_data_explorer.html` — file renamed
- `templates/model_lab.html` → `templates/risk_analysis_lab.html` — file renamed
- `README.md` — full rewrite
- `HANDOFF.md`, `CLAUDE.md`, `RULES.md` — rename references updated

**All committed and pushed (commit `052eded`).**

---

### Last Session — 2026-05-10 (session 5)

**New features shipped:**
1. **Cohort comparison bands in per-patient Data Explorer** — `cohort_data_explorer()` route now generates synthetic `cohort_bands` dict (mean ± std per feature per cohort) using a per-patient seeded numpy RNG. Three cohorts: All Patients (140, wider spread), Stage-Matched Child-Pugh B (47), Age-Matched (23). Bands injected into `chart_data.cohort_bands`. In the template, `drawChart()` reads `window.__activeCohorts` (exposed from the cohort IIFE) and draws a filled `<polygon>` (10% opacity) + dashed mean `<polyline>` (45% opacity) behind patient lines for each active cohort. Colors: indigo / green / amber. `document.addEventListener('cohortChanged', drawChart)` triggers instant redraw on chip toggle.
2. **Data Explorer tab added to patient nav** — `base.html` was missing the tab entirely. Added `<a href="/patient/{{ patient.id }}/cohort-data-explorer">Data Explorer</a>` alongside Biometrics and Risk Analysis.
3. **Flagged night markers on explorer chart** — `cohort_data_explorer()` now calls `_compute_patient_view_data()` to get `auto_confounders` and passes `flagged_dates` (list of `{date, reasons}`) into `chart_data`. `drawChart()` draws amber dashed vertical lines + downward triangle caps at those dates.
4. **7-day rolling mean toggle** — Checkbox at the bottom of the Signals sidebar. `showRollingMean` state variable; `rollingMean(vals, w)` causal helper. When enabled, draws a thicker (stroke-width 3, opacity 0.6) line per visible signal using the same normalization as the raw data.
5. **Signal statistics table** — Appears below the chart whenever signals are selected. Shows Mean, SD, Min, Max, Trend (% change first→last quartile, ↑↓→ arrow with color) per signal. `updateStats()` called at the end of every `drawChart()` call.
6. **CSV export button** — Next to "Add to Notebook" in the chart header. Client-side Blob download of visible signals × dates, respecting current window and feature selection.
7. **PT-1042 display name** — Changed from `DEMO_HE_PATIENT_ID` to `'Synthea Demo'` in `_make_demo_he_patient()`.

**Files changed this session:**
- `app.py` — `cohort_data_explorer()`: cohort band generation; `_compute_patient_view_data()` call for flagged dates; `flagged_dates` added to `chart_data`; PT-1042 name → `'Synthea Demo'`
- `templates/cohort_data_explorer.html` — `.stats-section` / `.stats-table` CSS; rolling mean checkbox HTML; Export CSV button HTML; stats table HTML; `showRollingMean` state; `rollingMean()` helper; flagged markers in `drawChart()`; rolling mean lines in `drawChart()`; `updateStats()`; `toggleRollingMean()`; `exportCSV()`; `window.__activeCohorts` exposure; `cohortChanged` listener
- `templates/base.html` — Added Data Explorer tab to patient nav

**Commits pushed:**
- `052eded` — session 4 features (large)
- `c290d09` — cohort bands + patient nav tab
- `25937dc` — flagged markers, rolling mean, stats table, CSV export

### Known Bugs
| Bug | Impact | Status |
|---|---|---|
| `@lru_cache` on wearable adapter | New files in `demo_omh_ieee/` not picked up until server restart | Acceptable for now |
| Word export is HTML-as-.doc | Not a real .docx; formatting depends on Word's HTML importer | Acceptable for demo; use python-docx for proper Word export |
| Clinical history for PT-1042 (MELD/ammonia) | Driven by seeded RNG, not hand-crafted; values are stable but not tuned for the HE story | Low priority — values are clinically plausible |

### Phase 2 Backlog (not started)
- [ ] **Real Oura API** — wire `OURA_API_TOKEN` env var into `oura_adapter.py`; replace seeded random with live data
- [ ] **SHAP + LLM pipeline** — `ai_assistant.html` is a shell; wire SHAP values + GPT-4o/Llama3 rationale into `explainability.py`
- [ ] **Authentication** — required before any real PHI enters the app; recommend Flask-Login + session tokens
- [ ] **Longitudinal progression score** — rolling 30/90-day aggregate HE risk index per patient; needs time-series trend analysis beyond 14-day window
- [ ] **Automated threshold alerts** — email/pager when patient metrics cross clinical thresholds (REM% < 15% for 3+ nights, efficiency < 75%, SpO₂ < 93%)
- [ ] **Epic SMART on FHIR** — pull flowsheet data directly vs Excel upload
- [ ] **Multi-site** — separate cohort views by site if the study expands
- [ ] **python-docx Word export** — replace HTML-as-.doc with a real .docx (add `python-docx` to requirements.txt)
- [ ] **`/api/patient/<id>/timeseries` JSON endpoint** — Notebook Builder time-series overlay template has a TODO stub; needs a real endpoint to serve per-patient daily data as JSON
- [ ] **Notebook Builder "Add to Existing" dropdown** — currently populated server-side at page load; notebooks created after page load won't appear without a refresh
- [ ] **`/api/notebooks/new` extended params** — endpoint accepts `patient_ids`, `analysis_type`, `metrics`, `date_range` in POST body (wired from Notebook Builder) but `_build_notebook()` doesn't use them yet; still generates the default cohort template
- [ ] **PT-1042 clinical history override** — optionally hand-craft MELD-Na, ammonia, bilirubin, INR values for PT-1042 to complete the HE story (currently seeded-random but clinically stable)

### Open Questions for Next Session
- After the Buckholz meeting (2026-05-07): which features resonated? Which backlog items should be promoted to active work?
- Should the Notebook Builder's "Add to Existing" dropdown refresh without a page reload (fetch on open)?
- For the real Oura API integration: should it replace the OMH/IEEE demo layer or run alongside it (allowing mixed real + synthetic cohorts)?
- Key Biometrics chart: should the "All" window expand to show a third Y-axis for SpO₂, or keep the 3-series layout?
- Highest-value next feature identified: **longitudinal overlay chart** on the dashboard Cohort Data Explorer tab — select 2–4 patients and overlay their trajectories on one chart. User confirmed interest; not yet implemented.
