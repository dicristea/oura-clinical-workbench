# De-identified Hepatology Dataset Conversion and Adapter

This pipeline prepares the de-identified hepatology dataset for the Python dashboard.

```text
RDS source data
    -> convert_hepatology_rds.R
    -> normalized JSON files
    -> HepatologyDataAdapter
    -> dashboard-ready Python objects
```

Relevant repository files:

```text
di_data/convert_hepatology_rds.R
data/hepatology_adapter.py
data/base.py
data/feature_registry.py
```

The last two files add the hepatology data source and relative study-day support to the shared data model.

## Data structure

The source contains three analytical levels:

- **Participant:** demographics and baseline clinical variables
- **Participant-day:** daily wearable summaries and data-quality measures
- **Within-day:** 1-minute and 5-minute activity sequences

The dataset uses a relative `study_day` axis because calendar dates are not available.

## 1. Convert RDS to JSON

Required R packages: `dplyr`, `digest`, `haven`, and `jsonlite`.

From the repository root:

```bash
conda activate oura-r
Rscript di_data/convert_hepatology_rds.R
```

The default command exports a 10-participant sample to:

```text
di_data/exports/hepatology_sample_10/
```

Generated files:

```text
manifest.json
participants.json
daily_records.json
within_day_records.json
variable_dictionary.json
validation_report.json
```

To export the full cohort without within-day arrays:

```bash
Rscript di_data/convert_hepatology_rds.R \
  di_data/de-iden_data_Adam_2026-09-22.rds \
  di_data/exports/hepatology_full \
  all \
  none
```

The fourth argument may be an integer, `per-participant`, `all`, or `none`.
Set `HEPATOLOGY_ID_SALT` before producing a long-lived shared dataset so pseudonymous IDs remain stable across data updates.

## 2. Load the converted data

The Python adapter does not limit the number of participants.

```python
from data.hepatology_adapter import HepatologyDataAdapter

adapter = HepatologyDataAdapter("di_data/exports/hepatology_sample_10")

print(adapter.describe_dataset())

patient_id = adapter.list_patient_ids()[0]
patient = adapter.load_patient(patient_id)
daily = adapter.load_patient_daily(patient_id, valid_only=True)

print(patient.static_features)
print(daily.head())
```

Additional methods:

```python
adapter.load_participants()
adapter.cohort_dataframe()
adapter.load_all()
adapter.load_within_day(patient_id, study_day=1)
```

The adapter preserves the relative study-day index and only renames fields whose meanings are sufficiently clear.

## Data safety

Do **not** upload the RDS file or generated JSON data to GitHub. Only upload the conversion script, adapter code, tests, and documentation. The data are de-identified but should remain in an approved local or institutional environment.
