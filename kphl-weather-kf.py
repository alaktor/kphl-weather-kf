# -*- coding: utf-8 -*-
#
import requests
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime
from zoneinfo import ZoneInfo
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

def to_utc_hourly_index(idx: pd.DatetimeIndex, assume_tz: str | None = None) -> pd.DatetimeIndex:
    """
    Ensure an index is timezone-aware UTC and floored to the hour.
    If idx is tz-naive and assume_tz is provided, localize to that tz first.
    """
    idx = pd.DatetimeIndex(idx)

    if idx.tz is None:
        if assume_tz:
            idx = idx.tz_localize(assume_tz)
        else:
            # If you truly don't know, treat as UTC rather than mixing naive+aware later
            idx = idx.tz_localize("UTC")

    idx = idx.tz_convert("UTC")
    return idx.floor("H")

def get_nws_hourly_forecast_df(lat=39.8733, lon=-75.2268):
    import requests, pandas as pd

    HEADERS = {"User-Agent": "google-colab-weather-script"}

    points = requests.get(
        f"https://api.weather.gov/points/{lat},{lon}",
        headers=HEADERS, timeout=30
    ).json()

    hourly_url = points["properties"]["forecastHourly"]
    hourly = requests.get(hourly_url, headers=HEADERS, timeout=30).json()
    periods = hourly["properties"]["periods"]

    rows = []
    for p in periods:
        rows.append({
            "time": p["startTime"],
            "temp_F": p["temperature"],
            "dewpoint_F": p.get("dewpoint", {}).get("value"),
            "wind_mph": p.get("windSpeed"),
            "wind_dir": p.get("windDirection"),
            "rh_pct": p.get("relativeHumidity", {}).get("value"),
            "sky_cover": p.get("shortForecast")  # proxy (categorical)
        })

    fcst_df = pd.DataFrame(rows)
    fcst_df["time"] = pd.to_datetime(fcst_df["time"])
    fcst_df = fcst_df.sort_values("time").reset_index(drop=True)
    return fcst_df

def get_nws_kphl_obs_df(url="https://forecast.weather.gov/data/obhistory/KPHL.html",
                        tz_name="America/New_York"):
    html = requests.get(url, timeout=30).text
    df = pd.read_html(StringIO(html))[0].copy()

    # Flatten headers
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join([str(x) for x in tup if str(x) != "nan"]).strip() for tup in df.columns.to_list()]
    else:
        df.columns = [str(c).strip().replace("\n", " ") for c in df.columns]

    date_col = "Date Date Date"                         # day of month only
    time_col = "Time (est) Time (est) Time (est)"       # HH:MM 24-hour
    temp_col = "Temperature (ºF) Air Air"

    # Build ET datetime using current month/year
    now_et = datetime.now(ZoneInfo(tz_name))
    year, month = now_et.year, now_et.month

    day_2 = (df[date_col].astype(str).str.extract(r"(\d{1,2})", expand=False)
             .fillna("").str.zfill(2))

    dt_str = f"{year:04d}-{month:02d}-" + day_2 + " " + df[time_col].astype(str).str.strip()
    time_et = pd.to_datetime(dt_str, format="%Y-%m-%d %H:%M", errors="coerce")

    # Clean temperature
    temp = (df[temp_col].astype(str)
            .str.replace(r"[^0-9\.\-]", "", regex=True)
            .replace("", np.nan))
    temp = pd.to_numeric(temp, errors="coerce")

    obs_df = pd.DataFrame({"time": time_et, "temp_F": temp})
    obs_df = obs_df.dropna().sort_values("time").reset_index(drop=True)
    return obs_df


def kalman_bias_update(b, P, residual, Q, R):
    """
    1D Kalman filter update for a time-varying temperature bias.

    Model:
      b_{t+1} = b_t + w_t        (random walk)
      residual_t = T_obs - T_fcst = b_t + v_t

    Parameters
    ----------
    b : float
        Prior bias estimate (°F)
    P : float
        Prior bias variance
    residual : float
        Observation residual (°F)
    Q : float
        Process noise variance (bias drift)
    R : float
        Observation noise variance

    Returns
    -------
    b_post : float
        Posterior bias estimate
    P_post : float
        Posterior bias variance
    K : float
        Kalman gain
    """

    # Predict
    b_pred = b
    P_pred = P + Q

    # Update
    K = P_pred / (P_pred + R)
    b_post = b_pred + K * (residual - b_pred)
    P_post = (1 - K) * P_pred

    return b_post, P_post, K


def upsert_master_by_time(master_path, new_block, time_col="time"):
    """
    Option A master:
    - Keep a single row per hour (keyed by `time`)
    - Append new hours as they appear
    - Update existing hours with new non-null values
    - Never overwrite existing values with NaN/blank
    - Preserve ALL historical rows (no trimming)
    """
    nb = new_block.copy()
    nb[time_col] = pd.to_datetime(nb[time_col])
    nb = nb.sort_values(time_col).reset_index(drop=True)

    # Load existing master
    if os.path.exists(master_path):
        master = pd.read_csv(master_path)
        master[time_col] = pd.to_datetime(master[time_col])
    else:
        master = pd.DataFrame(columns=nb.columns)

    # Ensure both have same columns (union)
    for c in nb.columns:
        if c not in master.columns:
            master[c] = pd.NA
    for c in master.columns:
        if c not in nb.columns:
            nb[c] = pd.NA

    # # Index by time for clean upsert
    # master = master.set_index(time_col)
    # nb = nb.set_index(time_col)
    # Index by time for clean upsert
    master = master.set_index(time_col)
    nb = nb.set_index(time_col)

    # Only overwrite if new value is non-null (prevents wiping with NaN)
    for col in nb.columns:
        if col not in master.columns:
            master[col] = pd.NA
        mask = nb[col].notna()
        master.loc[mask, col] = nb.loc[mask, col]


    # Update: only write where new values are NOT null
    master.update(nb)

    # Append: add times that are new
    master = pd.concat([master, nb[~nb.index.isin(master.index)]], axis=0)

    # Sort + save
    master = master.sort_index().reset_index()
    master.to_csv(master_path, index=False)
    return master


def run_pipeline(
    out_dir="out",
    station_id="KPHL",
    lat=39.8733,
    lon=-75.2268,
    obs_url="https://forecast.weather.gov/data/obhistory/KPHL.html",
    horizon_hours=48,
    tz_name="America/New_York"
):

    TZ = ZoneInfo(tz_name)
    os.makedirs(out_dir, exist_ok=True)

    # 1) Fetch
    fcst_df = get_nws_hourly_forecast_df(lat=lat, lon=lon)
    # obs_df = get_nws_kphl_obs_df(tz_name=tz_name)
    obs_df = get_nws_kphl_obs_df(url=obs_url, tz_name=tz_name)


    # 2) Standardize time (drop tz for alignment)
    fcst_df["time"] = pd.to_datetime(fcst_df["time"]).dt.tz_localize(None)
    obs_df["time"] = pd.to_datetime(obs_df["time"]).dt.tz_localize(None)

    # Minimal KF inputs
    fcst_kf = fcst_df[["time", "temp_F"]].copy()
    obs_kf = obs_df[["time", "temp_F"]].copy()

    fcst_kf["temp_F"] = pd.to_numeric(fcst_kf["temp_F"], errors="coerce")
    obs_kf["temp_F"] = pd.to_numeric(obs_kf["temp_F"], errors="coerce")
    fcst_kf = fcst_kf.dropna().sort_values("time")
    obs_kf = obs_kf.dropna().sort_values("time")

    # Align to hourly grid
    obs_kf["time_hr"] = obs_kf["time"].dt.round("h")
    fcst_kf["time_hr"] = fcst_kf["time"].dt.floor("h")

    obs_kf_hr = (obs_kf.sort_values("time")
                 .groupby("time_hr", as_index=False)
                 .agg(temp_F=("temp_F", "mean")))

    fcst_kf_hr = (fcst_kf.sort_values("time")
                  .groupby("time_hr", as_index=False)
                  .agg(temp_F=("temp_F", "mean")))

    merged = pd.merge(
        obs_kf_hr.rename(columns={"time_hr": "time", "temp_F": "T_obs"}),
        fcst_kf_hr.rename(columns={"time_hr": "time", "temp_F": "T_fcst"}),
        on="time",
        how="inner"
    ).sort_values("time")

    common = obs_hourly.index.intersection(fcst_hourly.index)
    if len(common) == 0:
        # Try a last-ditch: maybe one side is naive UTC and the other is UTC-aware.
        # Convert both again defensively and retry.
        obs_hourly.index = pd.DatetimeIndex(obs_hourly.index).tz_convert("UTC").floor("H")
        fcst_hourly.index = pd.DatetimeIndex(fcst_hourly.index).tz_convert("UTC").floor("H")
        common = obs_hourly.index.intersection(fcst_hourly.index)

    if len(common) == 0:
        print("DEBUG obs aligned:", obs_hourly.index.min(), "->", obs_hourly.index.max(), "n=", len(obs_hourly))
        print("DEBUG fcst aligned:", fcst_hourly.index.min(), "->", fcst_hourly.index.max(), "n=", len(fcst_hourly))
        raise RuntimeError("No overlapping times between obs and forecast after hourly alignment.")
    
    if merged.empty:
        raise RuntimeError("No overlapping times between obs and forecast after hourly alignment.")
        
    print("DEBUG obs aligned:", obs_hourly.index.min(), "->", obs_hourly.index.max(), "n=", len(obs_hourly))
    print("DEBUG fcst aligned:", fcst_hourly.index.min(), "->", fcst_hourly.index.max(), "n=", len(fcst_hourly))
    print("DEBUG obs tz:", getattr(obs_hourly.index, "tz", None))
    print("DEBUG fcst tz:", getattr(fcst_hourly.index, "tz", None))

    # Example: after building obs df
    # NOAA/NWS obhistory pages are typically local station time; for KNYC/KPHL that’s America/New_York.
    obs.index = to_utc_hourly_index(obs.index, assume_tz="America/New_York")

    # Example: after building forecast df
    # Many NWS/forecast APIs are already UTC; if tz-naive, treat as UTC.
    fcst.index = to_utc_hourly_index(fcst.index, assume_tz="UTC")

    # 3) KF bias update loop
    Q = 0.3**2
    R = 2.0**2
    b = 0.0
    P = 3.0**2

    rows = []
    for _, r in merged.iterrows():
        residual = float(r["T_obs"] - r["T_fcst"])
        b, P, K = kalman_bias_update(b, P, residual, Q, R)
        rows.append({
            "time": r["time"],
            "T_obs": r["T_obs"],
            "T_fcst": r["T_fcst"],
            "residual_F": residual,
            "bias_F": b,
            "bias_std_F": float(np.sqrt(P)),
            "K": K
        })

    bias_df = pd.DataFrame(rows)
    t_now = pd.to_datetime(bias_df["time"].iloc[-1])
    b_now = float(bias_df["bias_F"].iloc[-1])
    bias_std_now = float(bias_df["bias_std_F"].iloc[-1])

    # 4) Forecast correction for next horizon_hours
    future = fcst_kf[fcst_kf["time"] >= t_now].head(horizon_hours).copy()
    if future.empty:
        raise RuntimeError("Forecast does not extend beyond the last observation time.")
    future["T_corr"] = future["temp_F"] + b_now
    max_h = float(future["T_corr"].max())
    
# 5) Build progressive block and FILL bias for full forecast timeline
    obs_block = obs_kf_hr.rename(columns={"time_hr": "time", "temp_F": "obs_temp_F"}).copy()
    obs_block["time"] = pd.to_datetime(obs_block["time"]).dt.floor("h")

    # Use the full forecast dataframe so the master stores all forecast vars
    fcst_block = fcst_df.copy()
    fcst_block["time"] = pd.to_datetime(fcst_block["time"]).dt.floor("h")
    
    # Rename forecast columns to avoid collisions with obs
    fcst_block = fcst_block.rename(columns={
        "temp_F": "fcst_temp_F",
        "dewpoint_F": "fcst_dewpoint_C",   # see dewpoint note below
        "rh_pct": "fcst_rh_pct",
        "wind_mph": "fcst_wind_mph",
        "wind_dir": "fcst_wind_dir",
        "sky_cover": "fcst_sky_cover",
    })

    
    fcst_block["time"] = pd.to_datetime(fcst_block["time"]).dt.floor("h")

    # Hourly bias series (where available)
    bias_hourly = bias_df[["time", "bias_F", "bias_std_F"]].copy()
    bias_hourly["time"] = pd.to_datetime(bias_hourly["time"]).dt.floor("h")
    bias_hourly = bias_hourly.groupby("time", as_index=False).last().sort_values("time")

    # Forward-fill bias onto forecast times; fill remaining with latest b_now
    bias_idx = bias_hourly.set_index("time")[["bias_F", "bias_std_F"]]
    fcst_idx = fcst_block.sort_values("time").set_index("time")

    fcst_idx[["bias_F", "bias_std_F"]] = bias_idx.reindex(fcst_idx.index).ffill()
    fcst_idx["bias_F"] = fcst_idx["bias_F"].fillna(b_now)
    fcst_idx["bias_std_F"] = fcst_idx["bias_std_F"].fillna(bias_std_now)
    fcst_idx["corr_temp_F"] = fcst_idx["fcst_temp_F"] + fcst_idx["bias_F"]

    fcst_block2 = fcst_idx.reset_index()

    block = (fcst_block2
             .merge(obs_block, on="time", how="outer")
             .sort_values("time")
             .reset_index(drop=True))

    block["run_time_et"] = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")

    # 6) Save/update progressive master locally (Actions will upload to Drive)
    # master_path = os.path.join(out_dir, "KPHL_master_progressive.csv")
    master_path = os.path.join(out_dir, f"{station_id}_master_progressive.csv")
    master_df = upsert_master_by_time(master_path, block, time_col="time")

    fcst_df.to_csv(os.path.join(out_dir, f"{station_id}_fcst_latest.csv"), index=False)
    obs_df.to_csv(os.path.join(out_dir, f"{station_id}_obs_latest.csv"), index=False)
    bias_df.to_csv(os.path.join(out_dir, f"{station_id}_bias_latest.csv"), index=False)
    future.to_csv(os.path.join(out_dir, f"{station_id}_future_corr_latest.csv"), index=False)

    # 7) Save a plot PNG (no GUI needed)
    fig = plt.figure(figsize=(18, 5))

    # Observations
    plt.plot(obs_kf["time"], obs_kf["temp_F"], marker="o", linewidth=1, label="Observed T")

    # Full forecast
    plt.plot(fcst_kf["time"], fcst_kf["temp_F"], linewidth=1.5, alpha=0.6, label="NWS Forecast T (7-day)")

    # Corrected next horizon
    plt.plot(future["time"], future["T_corr"], marker="o", linewidth=2,
             label=f"Bias-corrected Forecast (next {horizon_hours}h)")

    plt.xlabel("Date / Time")
    plt.ylabel("Temperature (°F)")
    # plt.title(f"KPHL Kalman Bias Correction | Next {horizon_hours}h Max = {max_h:.1f}°F | Bias={b_now:+.2f}°F")
    plt.title(f"{station_id} Kalman Bias Correction | Next {horizon_hours}h Max = {max_h:.1f}°F | Bias={b_now:+.2f}°F")

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))

    # y-axis ticks: major 5, minor 1
    ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.grid(True, which="major", axis="y", linestyle="-", linewidth=0.8)
    ax.grid(True, which="minor", axis="y", linestyle="--", linewidth=0.4)
    ax.grid(True, which="major", axis="x", linestyle="--", linewidth=0.6)

    # x-limits: 8 hours ago to +48 hours (as in your notebook)
    t_ref = pd.to_datetime(t_now)
    ax.set_xlim(t_ref - pd.Timedelta(hours=8), t_ref + pd.Timedelta(hours=48))

    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()

    # plot_path = os.path.join(out_dir, "KPHL_plot_latest.png")
    plot_path = os.path.join(out_dir, f"{station_id}_plot_latest.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    # 8) Run log
    run_log = pd.DataFrame([{
        "run_time_et": datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S"),
        "t_now": str(t_now),
        "bias_now_F": b_now,
        "bias_std_now_F": bias_std_now,
        "max_next_horizon_corr_F": max_h,
        "horizon_hours": horizon_hours,
    }])
    # run_log.to_csv(os.path.join(out_dir, "KPHL_run_log_latest.csv"), index=False)
    run_log.to_csv(os.path.join(out_dir, f"{station_id}_run_log_latest.csv"), index=False)


    print("DONE")
    print(run_log.iloc[0].to_dict())
    print("Wrote outputs to:", out_dir)
    print("Progressive master rows:", len(master_df))

if __name__ == "__main__":
    run_pipeline()
