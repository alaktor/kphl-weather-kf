# run_stations.py
import sys
import subprocess
from typing import Dict, List

STATIONS: List[Dict[str, str]] = [
    {
        "station": "KPHL",
        "lat": "39.8731",
        "lon": "-75.2424",
        "obs_url": "https://forecast.weather.gov/data/obhistory/KPHL.html",
    },
    {
        "station": "KNYC",
        "lat": "40.7834",
        "lon": "-73.965",
        "obs_url": "https://forecast.weather.gov/data/obhistory/KNYC.html",
    },
    # add more here...
]


def run_one(cfg: dict) -> int:
    cmd = [
        sys.executable,
        "kphl-weather-kf.py",
        "--station",
        cfg["station"],
        "--lat",
        cfg["lat"],
        "--lon",
        cfg["lon"],
        "--obs_url",
        cfg["obs_url"],
    ]
    print(f"\n=== Running {cfg['station']} ===")
    try:
        # capture output so GH Actions log stays readable but still available on failure
        p = subprocess.run(cmd, text=True, capture_output=True)
        if p.returncode != 0:
            print(f"--- {cfg['station']} FAILED (exit {p.returncode}) ---")
            if p.stdout:
                print("STDOUT:\n" + p.stdout[-4000:])  # tail to avoid massive logs
            if p.stderr:
                print("STDERR:\n" + p.stderr[-4000:])
            return p.returncode
        else:
            if p.stdout:
                print(p.stdout)
            if p.stderr:
                # warnings often go to stderr; still print them
                print(p.stderr)
            print(f"--- {cfg['station']} OK ---")
            return 0
    except Exception as e:
        print(f"--- {cfg['station']} EXCEPTION ---\n{e}")
        return 99


def main() -> int:
    failures = []
    for cfg in STATIONS:
        rc = run_one(cfg)
        if rc != 0:
            failures.append((cfg["station"], rc))

    if failures:
        print("\n=== SUMMARY: FAILURES ===")
        for st, rc in failures:
            print(f"- {st}: exit {rc}")
        return 1

    print("\n=== SUMMARY: ALL STATIONS OK ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
