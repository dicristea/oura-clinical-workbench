# Demo Data

This directory contains the dashboard demo data workflow.

## Active Workflow

Run the demo generator from the project root:

```bash
python demo_data/generate_demo.py
```

By default, each run uses a new random seed and prints the seed it used. To reproduce a specific generated dataset, pass the seed explicitly:

```bash
python demo_data/generate_demo.py --seed 12345
```

The generator creates two linked synthetic datasets:

```text
demo_data/generate_demo.py
-> demo_data/demo_synthea/
-> demo_data/demo_omh_ieee/
-> dashboard
```

## Active Outputs

- `demo_synthea/`: generated Synthea-style FHIR Bundle JSON files, one file per demo patient.
- `demo_omh_ieee/`: generated OMH / IEEE wearable records derived from the FHIR patient profiles.

`demo_omh_ieee/` is ignored by Git because it is a local generated output. Recreate it by running `python demo_data/generate_demo.py`.

## Generator Code

- `generate_demo.py`: top-level demo data entry point.
- `omh_ieee_generator/`: helper modules that turn generated FHIR patient profiles into Oura-like OMH / IEEE wearable records.

## Runtime Readers

The dashboard reads these outputs through adapters in `data/`:

- `data/synthea_adapter.py` reads `demo_synthea/`.
- `data/standard_wearable_adapter.py` reads `demo_omh_ieee/`.

## Archived Assets

- `unused/`: legacy demo assets kept for reference, but not used by the current dashboard flow.
- `unused/legacy_data_syn/`: archived Oura API pull/converter/schema code that may be useful later if real Oura-to-OMH/IEEE conversion is revived.
