# Legacy `data_syn` Archive

This directory preserves useful pieces from the earlier `data_syn/` workspace for possible future reuse. It is not part of the current dashboard demo data flow.

The active flow is:

```text
demo_data/generate_demo.py
-> demo_data/demo_synthea/
-> demo_data/demo_omh_ieee/
-> data/standard_wearable_adapter.py
-> dashboard
```

## Why This Is Archived

These files may be useful later if the project returns to either of these directions:

- pulling real Oura API data
- converting raw Oura API JSON into OMH / IEEE records
- adding stricter schema-reference validation for generated OMH / IEEE records

## Contents Worth Keeping

- `config/mapping_config.json`: old source-to-standard field mapping for Oura JSON.
- `config/oura-openapi-1.28.json`: local Oura API spec reference.
- `schemas/`: local OMH / IEEE schema reference files.
- `scripts/oura_data_pull.py`: old Oura API pull workflow.
- `scripts/oura_standard_converter.py`: old raw Oura JSON to OMH / IEEE converter.
- `utils/`: helper code used by the old scripts.
- `examples/`: early synthetic example records.

## Not Active

The current OMH / IEEE demo generator does not import this archive. If these files are revived, they should be moved into a clearly named module such as `tools/oura_converter/` or `standards/schemas/` and updated before use.
