# `demo_data/omh_ieee_generator`

## What This Generator Uses From Synthea

The generator reads only a small subset of the Synthea bundle:

- `Patient`
  - patient id
  - name
  - gender
  - birth date
- `Condition`
  - condition labels from `code.text` or `coding.display`

It does not currently derive measurements directly from Synthea observations, medications, or encounters.

## Quick Start

Run the combined dashboard demo generator from the project root:

```bash
python demo_data/generate_demo.py
```

By default, the top-level generator chooses a new random seed and prints it. To reproduce the same FHIR and OMH / IEEE demo dataset, pass the seed explicitly:

```bash
python demo_data/generate_demo.py --seed 12345
```

That first writes the FHIR patient profile bundles to `demo_data/demo_synthea/`, then writes the OMH / IEEE wearable records to `demo_data/demo_omh_ieee/`. The generator does not create new Synthea clinical profiles; it reads the existing dashboard FHIR bundles and adds simulated wearable records in OMH / IEEE format.

## Outputs

The generator writes:

- `manifest.json`
  - dataset-level summary
  - schema counts
  - patient list
- `patients/<patient_id>/records.json`
  - patient metadata sidecar
  - generated OMH / IEEE records grouped by schema
  - per-patient counts
  - validation warnings

For dashboard demo bundles, `<patient_id>` comes from the FHIR filename, such as `PT-3001.json`, so the wearable records can line up with the same dashboard patient. The original FHIR `Patient.id` is preserved separately as `source_fhir_patient_id`.

## Standard Records Generated Per Day

For each generated patient-day, the current version emits:

- `ieee:physical-activity:1.0`
- `ieee:heart-rate:1.0`
- `ieee:sleep-episode:1.0`
- `ieee:time-in-bed:1.0`
- `omh:heart-rate:2.0`
- `omh:respiratory-rate:2.0`
- `omh:oxygen-saturation:2.0`
- `omh:total-sleep-time:1.0`
- `omh:sleep-episode:1.1`

The generator does not currently load local JSON schema files at runtime. The schema identifiers above are written directly into each record header, and `validate.py` performs lightweight semantic checks such as value ranges and time consistency.

## Simulation Logic

The generator uses a two-step process:

1. convert each Synthea patient into a compact internal profile
2. estimate one overall `risk_score` for that patient and let multiple wearable signals co-vary with that score

This means the data is not purely random. A patient with a higher simulated risk tends to have:

- shorter and less efficient sleep
- more wake-after-sleep-onset
- lower REM and deep sleep share
- lower daily activity
- higher nightly heart rate
- slightly higher respiratory rate
- slightly lower oxygen saturation

## Risk Score Rules

The generator computes one patient-level `risk_score` and clamps it into the range `0.0` to `0.85`.

The score starts from a small baseline and increases with:

- age above 40
- presence of certain condition keywords
- number of conditions, up to a cap

The current keyword list is:

- `diabetes`
- `chronic kidney`
- `hypertension`
- `heart`
- `copd`
- `asthma`
- `depression`
- `sleep`
- `obesity`

These keywords are only heuristics. They are used to create internally consistent demo data, not to represent a medical risk model.

## How Sleep Is Simulated

For each day, the generator creates one main overnight sleep episode.

### Sleep Start

- bedtime starts around 10 PM UTC
- a small random offset is added
- higher risk slightly delays bedtime

### Sleep Latency

`latency_to_sleep_onset` is generated from:

- a base delay
- added delay for higher risk
- a small weekend effect
- random noise

### Time In Bed

`time_in_bed` is generated from:

- a baseline around 7.6 hours
- a reduction for older age
- a reduction for higher risk
- random day-to-day variation

The result is clipped into a plausible range:

- minimum `6 hours`
- maximum `9 hours`

### Total Sleep Time

`total_sleep_time` is computed as:

```text
time_in_bed - sleep_latency - wake_after_sleep_onset
```

Then it is clipped so it never becomes implausibly short or longer than the full time interval.

### Sleep Stage Split

The total sleep time is split into:

- deep sleep
- REM sleep
- light sleep

Rules:

- deep sleep percentage decreases as risk increases
- REM sleep percentage also decreases as risk increases
- light sleep is the remainder

### Sleep Efficiency

Sleep efficiency is derived, not sampled independently:

```text
total_sleep_time / time_in_bed
```

So when latency or wake time goes up, efficiency naturally goes down.

## How Activity Is Simulated

Daily activity is built around a step count and then translated into derived movement quantities.

### Step Count

Step count starts from a baseline near `9000` steps/day and then:

- decreases with higher risk
- decreases with older age
- gets a mild sinusoidal drift across days so the series does not look flat
- gets random daily noise

The result is clipped so it never falls below `1500` steps/day.

### Distance

Distance is derived from step count with a simple stride-length multiplier.

### Active Calories

Active calories are loosely tied to step count with added random variation.

### Activity Durations

The generator separately creates:

- light activity duration
- moderate activity duration
- vigorous activity duration

Higher risk generally reduces moderate and vigorous activity.

## How Heart Rate Is Simulated

The generator creates two heart-rate views:

- one nightly average heart rate for OMH summary output
- one 5-minute interval heart-rate time series for IEEE output

### Average Heart Rate

Nightly average heart rate is based on:

- a healthy baseline
- higher values for higher risk
- a mild age effect
- random variation

### Heart-Rate Series

The series starts at sleep onset plus latency and runs across the total sleep period.

Rules:

- one point every 5 minutes
- at least 12 points per night
- values follow a smooth nightly curve rather than white noise
- random noise is added
- higher risk slightly increases volatility
- values are clipped to avoid unrealistic lows

This produces a series that looks more like a sleep trace and less like independent random samples.

## How Respiratory Rate and SpO2 Are Simulated

Respiratory rate and oxygen saturation are generated as nightly summary measurements.

### Respiratory Rate

- starts from a healthy baseline
- increases slightly with risk
- includes small random noise

### Oxygen Saturation

- starts near a healthy baseline
- decreases slightly with risk
- includes small random noise
- is clipped to avoid implausibly low values in this demo generator

## Reproducibility

The dataset is deterministic for a given:

- `seed`
- `patient_id`
- number of days
- end date

Internally, the random generator uses a combined seed of:

```text
<seed>:<patient_id>
```

So if you rerun the generator with the same inputs, the same patient will get the same synthetic trajectory.

## Validation Rules

This generator currently performs lightweight semantic validation, not full formal schema validation.

The checks include:

- required bodies exist
- durations are positive
- `total_sleep_time` does not exceed the enclosing time interval
- activity quantities are not negative
- heart rate stays in a reasonable range
- respiratory rate stays in a reasonable range
- oxygen saturation stays in a reasonable range

Warnings are written into each patient's `records.json`.

## Important Limitations

This generator is useful for:

- UI development
- OMH / IEEE integration testing
- batch data generation for demos
- validating downstream file handling

It is not suitable for:

- clinical decision support
- scientific benchmarking
- realistic disease progression research
- evaluation of physiological inference methods

Specific limitations:

- only a few FHIR resource types are used
- patient conditions affect data only through simple keyword heuristics
- all sleep is modeled as one main overnight episode
- timezone handling is intentionally simple
- the generator currently does not simulate medications, labs, encounters, or multi-device disagreement

## File Map

- `fhir.py`
  - loads patient identity and condition labels from Synthea FHIR bundles
- `simulate.py`
  - creates daily and nightly synthetic wearable measurements
- `records.py`
  - converts simulated values into OMH / IEEE records
- `validate.py`
  - runs lightweight semantic checks
- `pipeline.py`
  - orchestrates the full dataset generation flow

## Future Extensions

If we want a more realistic generator later, the next easiest upgrades are:

- use Synthea observations and encounters to condition the wearable simulation
- add cohort templates such as `stable`, `frail`, `post-hospitalization`
- add formal JSON Schema validation
- add more OMH / IEEE schemas
- add timezone-aware local-day generation
