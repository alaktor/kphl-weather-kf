# run_stations.py
import subprocess
import sys

STATIONS = [
    {
        "station": "KPHL",
        "lat": "39.8729",
        "lon": "-75.2437",
        "obs_url": "https://forecast.weather.gov/data/obhistory/KPHL.html",
    },
    {
        "station": "KNYC",
        "lat": "40.7834",
        "lon": "-73.965",
        "obs_url": "https://forecast.weather.gov/data/obhistory/KNYC.html",
    },
]

def run_one(cfg: dict) -> None:
    cmd = [
        sys.executable, "run_kphl_kf.py",
        "--station", cfg["station"],
        "--lat", cfg["lat"],
        "--lon", cfg["lon"],
        "--obs_url", cfg["obs_url"],
    ]
    print(f"\n=== Running {cfg['station']} ===")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    for cfg in STATIONS:
        run_one(cfg)
