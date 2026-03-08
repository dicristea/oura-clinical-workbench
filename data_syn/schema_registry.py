from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class SchemaRecord:
    standard: str          # "ieee" or "omh"
    schema_id: str         # e.g. "ieee:sleep-episode:1.0"
    path: Path
    schema: Dict[str, Any]


class SchemaRegistry:
    def __init__(self) -> None:
        self._by_id: Dict[str, SchemaRecord] = {}

    def register_directory(self, directory: str | Path, standard: str) -> None:
        """
        Load all *.json files under a directory and register them.
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Schema directory not found: {directory}")

        for path in directory.rglob("*.json"):
            try:
                with path.open("r", encoding="utf-8") as f:
                    schema = json.load(f)
            except Exception as e:
                print(f"[WARN] Failed to load {path}: {e}")
                continue

            schema_id = self._infer_schema_id(schema, path, standard)
            if not schema_id:
                print(f"[WARN] Could not infer schema_id for {path}")
                continue

            record = SchemaRecord(
                standard=standard,
                schema_id=schema_id,
                path=path,
                schema=schema,
            )
            self._by_id[schema_id] = record

    def get(self, schema_id: str) -> Optional[SchemaRecord]:
        return self._by_id.get(schema_id)

    def require(self, schema_id: str) -> SchemaRecord:
        record = self.get(schema_id)
        if record is None:
            raise KeyError(f"Schema not found: {schema_id}")
        return record

    def resolve_preferred(
        self,
        ieee_schema_id: Optional[str],
        omh_schema_id: Optional[str],
    ) -> Optional[SchemaRecord]:
        """
        Prefer IEEE; if not available, fall back to OMH.
        """
        if ieee_schema_id:
            record = self.get(ieee_schema_id)
            if record:
                return record

        if omh_schema_id:
            record = self.get(omh_schema_id)
            if record:
                return record

        return None

    def list_ids(self) -> list[str]:
        return sorted(self._by_id.keys())

    def _infer_schema_id(
        self,
        schema: Dict[str, Any],
        path: Path,
        standard: str,
    ) -> Optional[str]:
        """
        Try multiple strategies to infer a canonical schema_id.

        Preferred output format:
          - ieee:sleep-episode:1.0
          - omh:heart-rate:2.0
        """
        # Strategy 1: explicit custom field if present
        for key in ("schema_id", "$id", "id"):
            value = schema.get(key)
            if isinstance(value, str) and value.strip():
                normalized = self._normalize_schema_id(value, standard)
                if normalized:
                    return normalized

        # Strategy 2: infer from filename like sleep-episode-1.0.json
        stem = path.stem  # e.g. sleep-episode-1.0
        inferred = self._infer_from_filename(stem, standard)
        if inferred:
            return inferred

        return None

    def _normalize_schema_id(self, raw: str, standard: str) -> Optional[str]:
        raw = raw.strip()

        # already canonical
        if raw.startswith("ieee:") or raw.startswith("omh:"):
            return raw

        # URL-like ids -> keep last segment if possible
        # Example fallback: ".../heart-rate-2.0.json" -> omh:heart-rate:2.0
        candidate = Path(raw).stem
        inferred = self._infer_from_filename(candidate, standard)
        if inferred:
            return inferred

        return None

    def _infer_from_filename(self, stem: str, standard: str) -> Optional[str]:
        """
        Convert:
          heart-rate-2.0 -> omh:heart-rate:2.0
          sleep-episode-1.0 -> ieee:sleep-episode:1.0
        """
        parts = stem.split("-")
        if len(parts) < 2:
            return None

        version = parts[-1]
        name = "-".join(parts[:-1])

        # crude version check
        if not version[0].isdigit():
            return None

        return f"{standard}:{name}:{version}"


def build_default_registry(
    ieee_dir: str | Path = "schemas/ieee",
    omh_dir: str | Path = "schemas/omh",
) -> SchemaRegistry:
    registry = SchemaRegistry()
    registry.register_directory(ieee_dir, standard="ieee")
    registry.register_directory(omh_dir, standard="omh")
    return registry


if __name__ == "__main__":
    registry = build_default_registry()
    print(f"Loaded {len(registry.list_ids())} schemas")
    for schema_id in registry.list_ids()[:20]:
        print(schema_id)
