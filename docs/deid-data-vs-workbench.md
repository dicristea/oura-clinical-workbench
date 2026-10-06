# De-identified HE Dataset vs. Current Oura Clinical Workbench

_Last updated: 2026-10-06. Sources: Granola notes for **Spec Projects Weekly** (Sep 15, 22, 29) and **Hep Dashboards Spec** with Dr. Buckholz (Sep 16), plus this repo's code._

> **How to read this:** ✅ = confirmed in meetings, 🔍 = still to check with
> `notebooks/deid_oura_data_exploration.py` (Step 20 writes a
> `workbench_parameter_coverage.csv` you can paste in below), ❌ = not in the
> data as far as we know.

---

## 1. What the real dataset is (from the meetings)

| Fact | Source |
|---|---|
| **118 participants**, ~**15,000 participant-days**, **51 variables** | Henry, Sep 29 ✅ |
| Typical participant ~180 days; median ~115 days | Henry, Sep 29 ✅ |
| Three levels: **participant** (demographics, baseline clinical), **day** (daily summaries such as HR and steps), **intraday** (activity sequences) | Henry, Sep 29 ✅ |
| `valid_day` flag = ≥80% of the day has valid data (made by the study's statisticians, not by us); ~80% meet it | Henry and JP, Sep 29 ✅ |
| SpO₂ and heart rate are **not available for everyone** | Henry, Sep 29 ✅ |
| Links Oura data with **lab values, hospitalizations, PROs and neurocognitive testing**; examples given: age, gender, liver disease severity, ascites (yes/no), medications | Adam, Sep 16 ✅ (specific tests not named) |
| Oura **CSV export** (JSON also exists), not the live API | Adam, Sep 16 ✅ |
| Sleep stages are a **5-minute epoch string in one cell**; nightly summary totals also included | Adam, Sep 16 ✅ |
| **All timestamps are GMT** → convert to America/New_York (watch DST and date shifts) | Adam, Sep 16 ✅ |
| HR / HRV raw files are **20–30 GB**, too big to open casually | Adam, Sep 16 ✅ |
| Menstrual-cycle data | Not mentioned ❌. The notebook checks for it and offers a temperature-based hint (Step 12) |

## 2. What the workbench shows today (all synthetic)

Pulled from `data/feature_registry.py`, `templates/risk_analysis_lab.html`, `templates/patient_detail.html` and `app.py`.

| Group | Parameters in the workbench | Where |
|---|---|---|
| Sleep architecture | REM %, REM fragmentation, Deep %, Light %, Latency, Efficiency, Total sleep, WASO | Biometrics, Sleep Detail, Risk Lab |
| Physiology | HRV balance, Nighttime HRV (2–6 am), Body temp deviation, Resting HR, Respiratory rate, SpO₂ | Biometrics, Dashboard, Risk Lab |
| Activity | Steps, Inactivity alerts, Activity score | Biometrics, Risk Lab |
| Liver labs | Ammonia, Albumin, Bilirubin, INR, ALT, AST | Risk Lab |
| Severity | MELD, Child-Pugh | Risk Lab, Data Explorer cohort bands |
| Cognitive | NCT-A, NCT-B, HE grade | Risk Lab |
| Medications | Lactulose, Rifaximin compliance, Diuretic, Beta-blocker | Risk Lab |
| Confounders | Sleep apnea, alcohol, narcotics; auto-flags from SpO₂ < 93%, temp > 0.8 °C, low HRV, fragmented sleep | Confounder flags |
| Other screens | Cohort comparison bands, ML model runner (XGBoost etc.), SHAP, AI assistant | Data Explorer, Risk Lab, AI Assistant |

## 3. Side-by-side: keep, verify or drop

| Workbench parameter | Expected in real data? | Suggestion for simpler dashboard |
|---|---|---|
| Total sleep, REM %, Deep %, Light %, Efficiency, Latency, WASO | ✅ nightly summaries (REM % etc. may need computing from minutes or seconds; the notebook does it) | **Keep.** Core HE story (pilot finding: REM and circadian changes) |
| REM fragmentation, Nighttime HRV 2–6 am | 🔍 derivable from the 5-min stage string / intraday files | Later; needs extra processing |
| HRV, Resting HR, Temp deviation, Resp. rate | 🔍 likely present (standard Oura export) | **Keep** HRV, Resting HR and Temp; Resp. rate optional |
| SpO₂ | ✅ present but **partial coverage** | Show only when available; don't base alerts on it |
| Steps | ✅ day level | **Keep** |
| Inactivity alerts, Activity score | 🔍 | Drop unless coverage is good |
| Bedtime / circadian timing | 🔍 (needs GMT → ET conversion) | **Add.** Not in workbench today but central to the HE hypothesis |
| Night-to-night variability | Computed (notebook Step 14) | **Add** as a simple "stability" indicator |
| Age, sex | ✅ | Keep in header |
| Ascites history, disease severity (MELD / Child-Pugh) | ✅ ascites, 🔍 exact severity score | Use for **between-patient comparisons** (Adam's main ask) |
| HE status / grade, neurocognitive tests (NCT, PHES, Stroop…) | ✅ "neurocognitive testing" exists, 🔍 which tests | Keep the outcome label; test names to confirm |
| Ammonia, Albumin, Bilirubin, INR, ALT, AST | 🔍 "lab values" exist, list unknown | Keep only labs that exist and are repeated over time |
| Lactulose / Rifaximin compliance | 🔍 medications mentioned; compliance unlikely | Probably **drop** compliance %, keep yes/no use |
| PROs | ✅ exist, 🔍 which ones | New: candidate for the patient view |
| Hospitalizations | ✅ mentioned | New: event markers on the timeline |
| Menstrual cycle | ❌ not mentioned | Ask Adam; otherwise the temp-based hint only |
| ML Risk Lab, SHAP, AI assistant, cohort bands | n/a (research features) | **Move to the research prototype**, not the clinician dashboard |

### Paste notebook results here
<!-- Copy exploration_outputs/workbench_parameter_coverage.csv (counts only, no rows) -->

| workbench parameter | found? | % valid days with value | % people with any value |
|---|---|---|---|
| _run Step 20_ | | | |

## 4. Suggested shape of the new, simpler repo

1. **One patient view**: 6–8 tiles at most (sleep duration, REM %, deep %, efficiency, HRV, resting HR, temp deviation, steps), each with a 7-day average and the person's own baseline.
2. **One cohort view**: compare groups Adam cares about (HE vs. no HE, ascites history, severity, sex) using per-person averages.
3. **Data quality always visible**: show `valid_day` coverage so clinicians know when a trend can be trusted.
4. Leave ML, SHAP and AI chat in the existing research workbench.
5. Same research/product format as today; SMART on FHIR launch later (JP's Medplum demo shows the path; only the MRN link is needed).

## 5. Action item check (from Granola)

| Date | Action item | Owner | Status |
|---|---|---|---|
| Sep 16 | Explore wearable data in Jupyter, prepare visualizations (start with demographics and wearable data) | **Diana**, Henry | 🟡 Henry shared an overview Sep 29; this notebook covers Diana's part. Run it on the real data |
| Sep 16 | Share de-identified dataset and data dictionary | Adam | ✅ data received (by Sep 29) |
| Sep 16 | Add Diana and Henry to the IRB protocol | Adam | ❓ not confirmed in notes. Follow up |
| Sep 22 | Set up Jupyter AI locally | Diana | ✅ said done on Sep 29 |
| Sep 22 | Set up a Jupyter AI chat with Henry and Fisayo (setup and local models) | Diana | ❓ not mentioned as done |
| Sep 22 | Explore open-source models to run locally | Diana | ❓ open |
| Sep 22 | Share Product Studio calendar with JP | Diana | ❓ Sep 29 transcript suggests not yet sent |
| Sep 29 | Message Product Studio coach about Milestone 3 conflict | Diana | ❓ |
| Sep 29 | Share Jupyter notebook in Slack | Henry | ❓ check Slack |
| Sep 29 | Check whether the data can live on Cornell Tech JupyterHub | JP | ⏳ pending |
| Sep 16 | Schedule follow-up with Adam in 1–2 weeks | Adam | ❓ not seen in notes yet |

**Questions to bring to Adam:** Which neurocognitive tests and PROs are included? Which labs are repeated over time? Any menstrual or menopause info? What is the digit code in the sleep-stage string? Are IRB additions done?
