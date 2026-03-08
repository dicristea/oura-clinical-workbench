from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class SchemaSource:
    standard: str   # "ieee" or "omh"
    schema_id: str  # e.g. "omh:heart-rate:2.0"
    url: str


class SchemaDownloader:
    def __init__(self, base_dir: str | Path = "schemas") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def download_many(
        self,
        sources: Iterable[SchemaSource],
        overwrite: bool = False,
        timeout: int = 30,
    ) -> None:
        for source in sources:
            self.download_one(source, overwrite=overwrite, timeout=timeout)

    def download_one(
        self,
        source: SchemaSource,
        overwrite: bool = False,
        timeout: int = 30,
    ) -> Path:
        target_dir = self.base_dir / source.standard
        target_dir.mkdir(parents=True, exist_ok=True)

        filename = self._schema_id_to_filename(source.schema_id)
        target_path = target_dir / filename

        if target_path.exists() and not overwrite:
            print(f"[SKIP] {source.schema_id} -> {target_path}")
            return target_path

        print(f"[GET ] {source.schema_id} <- {source.url}")
        try:
            with urllib.request.urlopen(source.url, timeout=timeout) as response:
                raw = response.read()

            # validate JSON before saving
            parsed = json.loads(raw.decode("utf-8"))

            with target_path.open("w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2, ensure_ascii=False)

            print(f"[SAVE] {target_path}")
            return target_path

        except Exception as e:
            raise RuntimeError(f"Failed to download {source.schema_id} from {source.url}: {e}") from e

    def _schema_id_to_filename(self, schema_id: str) -> str:
        """
        Convert:
          omh:heart-rate:2.0 -> heart-rate-2.0.json
          ieee:sleep-episode:1.0 -> sleep-episode-1.0.json
        """
        parts = schema_id.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid schema_id format: {schema_id}")
        _, name, version = parts
        return f"{name}-{version}.json"


def get_default_sources() -> list[SchemaSource]:
    """
    Fill this list with the data_syn files your project actually uses.
    Replace the placeholder URLs with the real raw JSON URLs from the
    Open mHealth / IEEE repositories.
    """
    return [
        # OMH examples
        SchemaSource(
            standard="omh",
            schema_id="omh:heart-rate:2.0",
            url="https://raw.githubusercontent.com/openmhealth/schemas/main/schema/omh/heart-rate-2.0.json",
        ),
        SchemaSource(
            standard="omh",
            schema_id="omh:calories-burned:2.0",
            url="https://raw.githubusercontent.com/openmhealth/schemas/main/schema/omh/calories-burned-2.0.json",
        ),
        SchemaSource(
            standard="omh",
            schema_id="omh:respiratory-rate:2.0",
            url="https://raw.githubusercontent.com/openmhealth/schemas/main/schema/omh/respiratory-rate-2.0.json",
        ),
        SchemaSource(
            standard="omh",
            schema_id="omh:sleep-episode:1.1",
            url="https://raw.githubusercontent.com/openmhealth/schemas/main/schema/omh/sleep-episode-1.1.json",
        ),
        SchemaSource(
            standard="omh",
            schema_id="omh:oxygen-saturation:2.0",
            url="https://raw.githubusercontent.com/openmhealth/schemas/main/schema/omh/oxygen-saturation-2.0.json",
        ),
        SchemaSource(
            standard="omh",
            schema_id="omh:total-sleep-time:1.0",
            url="https://raw.githubusercontent.com/openmhealth/schemas/main/schema/omh/total-sleep-time-1.0.json",
        ),

        # IEEE examples
        # Replace these URLs with the actual raw files from your IEEE 1752 source
        SchemaSource(
            standard="ieee",
            schema_id="ieee:sleep-episode:1.0",
            url="https://w3id.org/ieee/ieee-1752-schema/sleep-episode.json",
        ),
        SchemaSource(
            standard="ieee",
            schema_id="ieee:time-in-bed:1.0",
            url="https://w3id.org/ieee/ieee-1752-schema/time-in-bed.json",
        ),
        SchemaSource(
            standard="ieee",
            schema_id="ieee:physical-activity:1.0",
            url="https://w3id.org/ieee/ieee-1752-schema/physical-activity.json",
        ),
    ]


if __name__ == "__main__":
    downloader = SchemaDownloader(base_dir="schemas")
    sources = get_default_sources()
    downloader.download_many(sources, overwrite=False)
