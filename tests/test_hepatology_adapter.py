from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data.hepatology_adapter import HepatologyDataAdapter
from data.base import DataSource


class HepatologyDataAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)

        payloads = {
            "manifest.json": {"schema_version": "1.1"},
            "participants.json": [
                {
                    "patient_id": "HEP-TEST-1",
                    "data_source": "hepatology_deidentified_rds",
                    "static_features": {"age_yrs": 60, "ctp_cat": "A"},
                    "coverage": {
                        "total_days": 2,
                        "valid_days": 1,
                        "first_study_day": 1,
                        "last_study_day": 2,
                    },
                },
                {
                    "patient_id": "HEP-TEST-2",
                    "data_source": "hepatology_deidentified_rds",
                    "static_features": {"age_yrs": 55, "ctp_cat": "B"},
                    "coverage": {
                        "total_days": 1,
                        "valid_days": 1,
                        "first_study_day": 1,
                        "last_study_day": 1,
                    },
                },
            ],
            "daily_records.json": [
                {
                    "patient_id": "HEP-TEST-1",
                    "study_day": 1,
                    "steps": 1000,
                    "average_heart_rate": 70,
                    "average_breath": 15.2,
                    "spo2": 97.0,
                    "valid80": 1,
                },
                {
                    "patient_id": "HEP-TEST-1",
                    "study_day": 2,
                    "steps": 800,
                    "average_heart_rate": 73,
                    "average_breath": 16.0,
                    "spo2": None,
                    "valid80": 0,
                },
                {
                    "patient_id": "HEP-TEST-2",
                    "study_day": 1,
                    "steps": 2500,
                    "average_heart_rate": 65,
                    "average_breath": 14.0,
                    "spo2": 98.0,
                    "valid80": 1,
                },
            ],
            "within_day_records.json": [
                {
                    "patient_id": "HEP-TEST-1",
                    "study_day": 1,
                    "series_lengths": {
                        "class_5_min": 2,
                        "met_val": 3,
                        "class5_val": 2,
                        "class1_val": 3,
                    },
                    "series": {
                        "class_5_min": [1, 2],
                        "met_val": [0.9, 1.0, 1.1],
                        "class5_val": [1, 2],
                        "class1_val": [1, 1, 2],
                    },
                }
            ],
            "variable_dictionary.json": [
                {"variable": "name", "level": "participant"},
                {"variable": "steps", "level": "participant_day"},
            ],
        }

        for filename, payload in payloads.items():
            (root / filename).write_text(json.dumps(payload), encoding="utf-8")

        self.adapter = HepatologyDataAdapter(root)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_loads_every_patient_without_limit(self) -> None:
        self.assertEqual(self.adapter.list_patient_ids(), ["HEP-TEST-1", "HEP-TEST-2"])
        self.assertEqual(len(self.adapter.load_all()), 2)

    def test_returns_study_day_patient_time_series(self) -> None:
        patient = self.adapter.load_patient("HEP-TEST-1")
        self.assertEqual(patient.data_source, DataSource.HEPATOLOGY)
        self.assertEqual(patient.time_axis, "study_day")
        self.assertNotIsInstance(patient.time_series.index, pd.DatetimeIndex)
        self.assertEqual(list(patient.time_series.index), [1, 2])
        self.assertIn("step_count", patient.time_series.columns)
        self.assertIn("respiratory_rate", patient.time_series.columns)
        self.assertIn("spo2_pct", patient.time_series.columns)

    def test_valid_only_and_analysis_window(self) -> None:
        valid = self.adapter.load_patient_daily("HEP-TEST-1", valid_only=True)
        self.assertEqual(list(valid.index), [1])
        patient = self.adapter.load_patient("HEP-TEST-1")
        self.assertEqual(list(patient.get_analysis_window(1).index), [2])

    def test_loads_within_day_arrays(self) -> None:
        record = self.adapter.load_within_day("HEP-TEST-1", 1)
        self.assertEqual(record["series_lengths"]["met_val"], 3)
        self.assertEqual(record["series"]["met_val"], [0.9, 1.0, 1.1])

    def test_describes_dataset(self) -> None:
        summary = self.adapter.describe_dataset()
        self.assertEqual(summary["participants"], 2)
        self.assertEqual(summary["participant_days"], 3)
        self.assertEqual(summary["valid80_days"], 2)


if __name__ == "__main__":
    unittest.main()
