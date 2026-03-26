"""
Fetch Oura usercollection endpoints and save each endpoint response as JSON.

Authentication is read from a local `.env` file or the OURA_ACCESS_TOKEN
environment variable so secrets are not stored in the repository.
"""

import json
import os
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

import requests

try:
    from data_syn.utils.paths import OURA_OPENAPI_PATH, RAW_OURA_DIR, REPO_ROOT
except ModuleNotFoundError:
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from data_syn.utils.paths import OURA_OPENAPI_PATH, RAW_OURA_DIR, REPO_ROOT


OUTPUT_DIR = RAW_OURA_DIR
OPENAPI_PATH = OURA_OPENAPI_PATH
BASE_URL = "https://api.ouraring.com"
ACCESS_TOKEN_ENV_VAR = "OURA_ACCESS_TOKEN"
DOTENV_PATH = REPO_ROOT / ".env"

START_DATE = "2026-03-23"
END_DATE = "2026-03-29"
LOCAL_TIMEZONE = ZoneInfo("America/New_York")


def build_datetime_window(start_date_str: str, end_date_str: str) -> tuple[str, str]:
    """
    Oura heartrate expects an inclusive start and an exclusive end datetime.
    We derive them from START_DATE / END_DATE so the datetime window cannot
    drift out of sync with the date window.
    """
    start_day = date.fromisoformat(start_date_str)
    end_day = date.fromisoformat(end_date_str)
    if end_day < start_day:
        raise ValueError(f"END_DATE must be on or after START_DATE: {start_date_str=} {end_date_str=}")

    start_dt = datetime.combine(start_day, time.min, tzinfo=LOCAL_TIMEZONE)
    end_dt = datetime.combine(end_day + timedelta(days=1), time.min, tzinfo=LOCAL_TIMEZONE)
    return start_dt.isoformat(), end_dt.isoformat()


START_DATETIME, END_DATETIME = build_datetime_window(START_DATE, END_DATE)


def load_dotenv(path: Path = DOTENV_PATH) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def build_headers() -> Dict[str, str]:
    load_dotenv()
    token = os.getenv(ACCESS_TOKEN_ENV_VAR, "").strip()
    if not token:
        raise RuntimeError(
            "Missing Oura access token. Add it to the repository `.env` file "
            f"or set the {ACCESS_TOKEN_ENV_VAR} environment variable before running."
        )

    authorization = token if token.lower().startswith("bearer ") else f"Bearer {token}"
    return {"Authorization": authorization}


def load_usercollection_endpoints(openapi_path: Path) -> List[Dict[str, Any]]:
    spec = json.loads(openapi_path.read_text(encoding="utf-8"))
    endpoints: List[Dict[str, Any]] = []

    for route, methods in sorted(spec.get("paths", {}).items()):
        if not route.startswith("/v2/usercollection/"):
            continue
        if "{document_id}" in route:
            continue
        if "get" not in methods:
            continue

        params = [
            param.get("name")
            for param in methods["get"].get("parameters", [])
            if isinstance(param, dict) and param.get("name")
        ]
        endpoints.append(
            {
                "route": route,
                "name": route.rsplit("/", 1)[-1],
                "params": params,
            }
        )

    return endpoints


def build_request_params(param_names: List[str], next_token: str | None = None) -> Dict[str, str]:
    params: Dict[str, str] = {}

    if "start_date" in param_names:
        params["start_date"] = START_DATE
    if "end_date" in param_names:
        params["end_date"] = END_DATE
    if "start_datetime" in param_names:
        params["start_datetime"] = START_DATETIME
    if "end_datetime" in param_names:
        params["end_datetime"] = END_DATETIME
    if next_token:
        params["next_token"] = next_token

    return params


def fetch_endpoint(
    session: requests.Session,
    endpoint: Dict[str, Any],
    headers: Dict[str, str],
) -> Dict[str, Any]:
    url = f"{BASE_URL}{endpoint['route']}"
    page_count = 0
    next_token: str | None = None
    combined_data: List[Any] = []

    while True:
        params = build_request_params(endpoint["params"], next_token=next_token)
        response = session.get(url, headers=headers, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        page_count += 1

        if isinstance(payload.get("data"), list):
            combined_data.extend(payload["data"])
            next_token = payload.get("next_token")
            if next_token:
                continue

            return {
                "fetch_metadata": {
                    "endpoint": endpoint["name"],
                    "route": endpoint["route"],
                    "request_params": build_request_params(endpoint["params"]),
                    "pages": page_count,
                    "record_count": len(combined_data),
                    "fetched_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                },
                "data": combined_data,
            }

        return {
            "fetch_metadata": {
                "endpoint": endpoint["name"],
                "route": endpoint["route"],
                "request_params": build_request_params(endpoint["params"]),
                "pages": page_count,
                "fetched_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            },
            "payload": payload,
        }


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    endpoints = load_usercollection_endpoints(OPENAPI_PATH)
    headers = build_headers()

    manifest: Dict[str, Any] = {
        "date_window": {
            "start_date": START_DATE,
            "end_date": END_DATE,
            "start_datetime": START_DATETIME,
            "end_datetime": END_DATETIME,
        },
        "openapi_path": str(OPENAPI_PATH),
        "output_dir": str(OUTPUT_DIR),
        "endpoints": [],
    }

    with requests.Session() as session:
        for endpoint in endpoints:
            output_path = OUTPUT_DIR / f"{endpoint['name']}.json"
            try:
                payload = fetch_endpoint(session, endpoint, headers)
                save_json(output_path, payload)

                record_count = payload.get("fetch_metadata", {}).get("record_count")
                manifest["endpoints"].append(
                    {
                        "name": endpoint["name"],
                        "route": endpoint["route"],
                        "status": "ok",
                        "output_file": output_path.name,
                        "pages": payload.get("fetch_metadata", {}).get("pages"),
                        "record_count": record_count,
                    }
                )
                print(f"Saved {endpoint['name']} -> {output_path}")
            except requests.RequestException as exc:
                manifest["endpoints"].append(
                    {
                        "name": endpoint["name"],
                        "route": endpoint["route"],
                        "status": "error",
                        "error": str(exc),
                    }
                )
                print(f"Failed {endpoint['name']}: {exc}")

    save_json(OUTPUT_DIR / "manifest.json", manifest)
    print(f"Saved manifest -> {OUTPUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
