# `data_syn` Layout

This directory is organized by responsibility so the Oura extraction and standards-conversion workflow is easier to navigate.

## Structure

```text
data_syn/
├── config/
│   ├── mapping_config.json
│   └── oura-openapi-1.28.json
├── outputs/
│   ├── raw/
│   │   └── oura/
│   │       ├── *.json
│   │       └── manifest.json
│   └── processed/
│       ├── converted_records.json
│       └── synthetic_examples.json
├── examples/
│   └── synthetic_records.py
├── schemas/
│   ├── ieee/
│   └── omh/
├── scripts/
│   ├── oura_data_pull.py
│   └── oura_standard_converter.py
├── utils/
│   ├── paths.py
│   ├── record_builder.py
│   ├── schema_downloader.py
│   └── schema_registry.py
├── oura_data_pull.py
└── oura_standard_converter.py
```

## What Goes Where

- `config/`: source-of-truth configuration files
- `outputs/raw/oura/`: raw Oura API responses
- `outputs/processed/`: normalized conversion outputs
- `examples/`: optional synthetic/demo generators kept outside the main pipeline
- `scripts/`: main workflows
- `utils/`: shared implementation helpers and path definitions
- `schemas/`: local OMH and IEEE schema files

## Entry Points

The top-level files stay in place as compatibility wrappers, so these commands still work:

```bash
python data_syn/oura_data_pull.py
python data_syn/oura_standard_converter.py
```

The actual implementation now lives in:

- `data_syn/scripts/oura_data_pull.py`
- `data_syn/scripts/oura_standard_converter.py`

If you want to generate synthetic sample records for schema examples, use:

```bash
python -m data_syn.examples.synthetic_records
```

## Oura Pull Script

`data_syn/scripts/oura_data_pull.py` fetches all `GET /v2/usercollection/*` endpoints listed in `config/oura-openapi-1.28.json`.

- Authentication is read from the repository root `.env` file first, and falls back to the `OURA_ACCESS_TOKEN` environment variable.
- The script automatically applies the configured date window.
- For endpoints like `heartrate`, it derives the required datetime window from the start and end dates.
- Each endpoint is saved as its own raw JSON file in `outputs/raw/oura/`.
- A `manifest.json` file records which endpoints succeeded, failed, how many pages were fetched, and record counts.

### Local Secret Setup

Create a local `.env` file at the repository root:

```bash
cp .env.example .env
```

Then edit `.env` and add your token:

```bash
OURA_ACCESS_TOKEN=your_access_token_here
```

The real `.env` file is ignored by Git, while [.env.example](../.env.example) can be committed as a template.

## Converter Overview

`data_syn/scripts/oura_standard_converter.py` converts saved Oura JSON into a review-friendly extract plus standardized OMH / IEEE records.

### Inputs

- `config/mapping_config.json`: source-to-standard mapping rules
- `outputs/raw/oura/*.json`: raw Oura endpoint payloads
- `schemas/omh/*.json` and `schemas/ieee/*.json`: local schema references used by the project

### Output Shape

The converter writes `outputs/processed/converted_records.json` with five top-level sections:

- `summary`: counts for mappings, missing files, missing fields, generated records, and schema coverage
- `focused_data`: a compact view of the source fields we care about by endpoint
- `records_by_schema`: standardized OMH / IEEE records grouped by schema id
- `counts_by_schema`: per-schema record counts
- `mapping_audit`: coverage and gap tracking for config entries, missing source files or fields, and unimplemented targets

### What The Converter Does

- Loads only the endpoints referenced in `mapping_config.json`.
- Extracts the source fields we care about into `focused_data`, while preserving identifiers and key timestamps like `id`, `day`, `timestamp`, `bedtime_start`, and `bedtime_end`.
- Summarizes nested array payloads in `focused_data` so large series remain inspectable without dumping every raw value.
- Normalizes schema ids and field paths from `mapping_config.json` into the actual OMH / IEEE record structure.
- Merges multiple Oura fields into a single standard record when they belong to the same schema instance.
- Adds default `effective_time_frame` intervals for day-level and sleep-level records when the source item provides enough timing information.
- Adds descriptive statistics like daily sums or nightly averages where the mapping requires them.
- Tracks mapping gaps explicitly in `mapping_audit` instead of silently dropping them.

### Current Standardized Outputs

The current converter generates these standard records:

- `omh:calories-burned:2.0` from `daily_activity.active_calories`
- `omh:oxygen-saturation:2.0` from `daily_spo2.spo2_percentage.average`
- `omh:respiratory-rate:2.0` from `sleep.average_breath`
- `omh:heart-rate:2.0` from sleep-level summary values such as average and lowest heart rate
- `omh:sleep-episode:1.1` from sleep timing and wake metrics
- `omh:total-sleep-time:1.0` from sleep duration
- `ieee:sleep-episode:1.0` from sleep timing and wake metrics
- `ieee:time-in-bed:1.0` from sleep timing plus `time_in_bed`
- `ieee:heart-rate:1.0` as a metadata data-series built from `sleep.heart_rate.items`
- `ieee:physical-activity:1.0` as one daily aggregate record with activity name `Total Daily Physical Activity`

### Endpoint-Specific Handling

- `sleep.heart_rate.items` is converted into one IEEE heart-rate series per sleep item, using the source timestamp and interval to reconstruct measurement times.
- Sleep summary heart-rate fields such as average and lowest heart rate remain separate OMH summary measurements.
- Daily activity is modeled as a whole-day IEEE physical activity record containing steps, walking-equivalent distance, active calories, and low / medium / high activity durations.
- `focused_data` currently covers `sleep`, `daily_activity`, `daily_spo2`, `daily_sleep`, and `daily_readiness`, even when some fields do not yet have a standard target.

## Git Hygiene

Local Oura data files are ignored by Git:

- `data_syn/outputs/raw/oura/*.json`
- `data_syn/outputs/processed/*.json`

This keeps raw pulls and local conversion outputs out of the repository while preserving the code and directory layout.
