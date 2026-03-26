from __future__ import annotations

from pathlib import Path


DATA_SYN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DATA_SYN_ROOT.parent

CONFIG_DIR = DATA_SYN_ROOT / "config"
SCHEMAS_DIR = DATA_SYN_ROOT / "schemas"
OUTPUTS_DIR = DATA_SYN_ROOT / "outputs"
RAW_OURA_DIR = OUTPUTS_DIR / "raw" / "oura"
PROCESSED_DIR = OUTPUTS_DIR / "processed"
EXAMPLES_DIR = DATA_SYN_ROOT / "examples"

MAPPING_CONFIG_PATH = CONFIG_DIR / "mapping_config.json"
OURA_OPENAPI_PATH = CONFIG_DIR / "oura-openapi-1.28.json"
CONVERTED_RECORDS_PATH = PROCESSED_DIR / "converted_records.json"
SYNTHETIC_EXAMPLES_PATH = PROCESSED_DIR / "synthetic_examples.json"

