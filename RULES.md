# Rules & Principles — JupyterHealth Clinical Workbench

> Read this at the start of every Claude session. These are standing rules that apply to all work on this project. They do not need to be re-stated in each conversation.

---

## Code Style

**Keep the stack simple.**
No React, Vue, Angular, Svelte, or any JS framework. No npm, no bundler, no transpilation. Vanilla HTML/CSS/JS only. If something is complex enough that you want a framework, simplify the feature instead.

**SVG charts are client-side and handwritten.**
No Plotly, Chart.js, or D3 in production Flask templates. All charts are drawn with `<svg>` and vanilla JS. Plotly is only allowed in `HF-Notebook/` and `notebooks/` (Jupyter). This keeps page loads fast and eliminates CDN dependencies.

**Flask stays in one file.**
All routes live in `app.py`. Do not split into blueprints unless the file exceeds ~2,000 lines and the user explicitly requests it. The flat structure is intentional for a small research tool.

**No database.**
Persistence uses JSON flat files: `study_config.json`, `confounders.json`. Do not introduce SQLite, PostgreSQL, or ORMs without explicit approval. This keeps deployment on Render free-tier trivial.

**No code comments explaining what the code does.**
Only add a comment when there is a non-obvious *why* — a hidden constraint, a deliberate workaround, a subtle invariant. Never add docstrings that just restate the function name.

---

## Data & Security (Non-Negotiable)

**Never commit PHI.**
`data.xlsx`, `.env`, `*.ipynb`, and any file with real patient identifiers must never be committed. The `.gitignore` enforces `data.xlsx` and `*.ipynb`. Verify before any `git add`.

**Seeded random for stable demo data.**
Demo patient data is generated with `random.seed(hash(patient_id))` (global) or `random.Random(abs(hash(patient_id)) % (2**31))` (isolated instance). Use the isolated form when you don't want to advance the global random state. The seed + call order together determine the values — if you change the order of random calls, data will shift. This is intentional stability, not a bug.

**`@lru_cache` on the wearable adapter means new demo files require a server restart.**
`_standard_wearable_series_by_patient()` caches on first call. If you add a file to `demo_data/demo_omh_ieee/`, stop and restart the Flask dev server before expecting it to appear.

**No authentication exists.**
Do not connect real PHI data under any circumstances until Flask-Login or equivalent is added. All demo data is synthetic.

---

## Templates & Frontend

**Template inheritance is fixed — respect the chain.**
```
base_layout.html → base.html → patient views (detail, risk-analysis-lab, cohort-data-explorer, ai-assistant)
base_layout.html → dashboard.html (directly)
standalone: patient_report.html (no extends)
```
Adding a new patient-view page: extend `base.html`, use `{% block page_head %}` for page-specific CSS (inside `{% block head %}`), `{% block content %}` for body.

**`{% block page_head %}` is inside `{% block head %}` in base.html.**
Child templates must use `page_head`, not `head`, or they will wipe the nav CSS.

**Modals need their close functions exposed on `window`.**
When writing modal JS inside an IIFE, always add `window.closeModal = closeModal` and `window.handleOverlayClick = handleOverlayClick` so `onclick` attributes on modal HTML can reach them.

**JS IIFEs for scoping.**
Dashboard and patient-detail JS is wrapped in `(function() { 'use strict'; ... })()`. Functions that need to be called from HTML `onclick` attributes must be explicitly assigned to `window`.

---

## Claude Session Protocol

**Start each session by reading HANDOFF.md.**
The Open section tells you the current state. The Frozen section tells you what not to change. Don't re-derive project structure from the code alone — trust the handoff.

**End each session by updating HANDOFF.md.**
Update the Open section with:
- What was shipped (files changed, features added)
- Any new bugs discovered
- Any new open questions
- Status of the pending commit (pushed or not)

**Do not refactor unless the user asks.**
A bug fix does not justify surrounding cleanup. A new feature does not justify restructuring adjacent code. Three similar lines of code is better than a premature abstraction. Scope changes to what was requested.

**Prefer editing existing files to creating new ones.**
Before creating a helper module, ask whether the logic fits in `app.py` or the existing template. New files add navigation cost for future sessions.

**When fixing a modal or JS bug, check for IIFE scoping first.**
The most common JS bug in this project is a function defined inside an IIFE not being reachable from an inline `onclick` attribute. Always check whether the function is assigned to `window`.

---

## Clinical Context (Always Relevant)

**The study is about covert hepatic encephalopathy, not generic sleep health.**
Features, copy, and alert thresholds should reflect HE clinical context:
- REM% < 15% sustained = significant signal (normal 18–25%)
- Sleep efficiency < 75% = concerning; < 85% = borderline
- WASO > 30 min = elevated; > 45 min = high
- SpO₂ < 93% = flag (sleep apnea confounder)
- HRV decline during sleep = early HE marker

**Confounders matter clinically.**
Narcotics, alcohol, and sleep apnea exacerbations can produce sleep deterioration that mimics HE progression. Flagged nights should be noted in analysis and excluded/down-weighted in model training.

**The audience is clinical researchers, not patients.**
UI copy should be precise and clinical. Avoid consumer-health softening ("your sleep was great!"). Show numbers, reference ranges, and data provenance.

---

## Deployment

**Render free-tier.**
`render.yaml` is present. Deploy by pushing to `main`. Gunicorn serves the app. Environment variables (`OURA_API_TOKEN`, `FLASK_DEBUG`, `PORT`, `LLM_API_KEY`, `JUPYTER_BASE_URL`) are set in Render dashboard, not committed.

**Local dev:**
```bash
python app.py          # runs on port 5000
PORT=8080 python app.py
```
No build step. No `npm install`. Just `pip install -r requirements.txt`.
