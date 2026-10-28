"""Archived ECMWF IFS runs from the Open-Meteo Single Runs API, one file per run.

Each run initialised at 00 UTC is requested for the eight points of ``benchmark/points.toml`` with ``forecast_days=2``
(hours 0-47 after initialisation). A run counts as available ``nwp_delay_hours`` after its initialisation
(``benchmark/spec.toml``), which is what makes the weather input "as of" the forecast origin: the day-ahead
forecast for D, issued on D-1 at 10:00 local time, sees the 00 UTC run of D-1 and nothing newer.

Weather data by Open-Meteo.com (CC BY 4.0), model ECMWF IFS HRES (ECMWF open data, CC BY 4.0).

Usage:
    python -m plfb nwp --start 2024-06-13 --end 2026-09-30
"""
from __future__ import annotations

import logging
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from plfb.config import points

log = logging.getLogger(__name__)

URL = "https://single-runs-api.open-meteo.com/v1/forecast"
MODEL = "ecmwf_ifs"
VARIABLES = [
    "wind_speed_100m", "wind_direction_100m", "wind_speed_10m", "temperature_2m", "relative_humidity_2m",
    "cloud_cover", "shortwave_radiation", "direct_radiation", "diffuse_radiation",
]
RUN_HOURS = (0,)


class DailyLimitError(RuntimeError):
    pass


class RunMissing(RuntimeError):
    """The archive has no data for this run (a few days in August 2025 and possibly elsewhere)."""


def run_path(root: Path, init: datetime) -> Path:
    return root / "raw" / "nwp" / MODEL / f"{init:%Y}" / f"run_{init:%Y%m%dT%H}.parquet"


def missing_path(root: Path, init: datetime) -> Path:
    return run_path(root, init).with_suffix(".missing")


def _parse(payload: list | dict, ids: list[str], init: datetime) -> pd.DataFrame:
    if isinstance(payload, dict):
        payload = [payload]
    frames = []
    for pid, loc in zip(ids, payload, strict=True):
        hourly = loc["hourly"]
        df = pd.DataFrame({v: hourly[v] for v in VARIABLES}, dtype="float32")
        df.insert(0, "valid_utc", pd.to_datetime(hourly["time"]).tz_localize("UTC"))
        df.insert(0, "point", pid)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "init_utc", pd.Timestamp(init))
    return out


def fetch_run(init: datetime, session: requests.Session, timeout: float = 120.0, days: int = 2) -> pd.DataFrame:
    pts = points()
    params = {
        "latitude": ",".join(f"{v:.2f}" for v in pts["lat"]),
        "longitude": ",".join(f"{v:.2f}" for v in pts["lon"]),
        "hourly": ",".join(VARIABLES),
        "models": MODEL,
        "run": f"{init:%Y-%m-%dT%H:%M}",
        "forecast_days": days,
        "timezone": "GMT",
        "wind_speed_unit": "ms",
    }
    for attempt in range(6):
        try:
            r = session.get(URL, params=params, timeout=timeout)
        except requests.RequestException as e:
            log.warning("run %s: %s, retry %d", init, e, attempt + 1)
            time.sleep(10 * (attempt + 1))
            continue
        if r.status_code == 429:
            reason = r.text[:300]
            if "Daily" in reason or "daily" in reason:
                raise DailyLimitError(reason)
            wait = 3600 - (time.time() % 3600) + 30 if ("Hourly" in reason or "hourly" in reason) else 65
            log.warning("429 (%s), waiting %.0f s", reason.strip(), wait)
            time.sleep(wait)
            continue
        if r.status_code == 400 and "not available" in r.text:
            raise RunMissing(r.text[:200])
        if r.status_code >= 500:
            log.warning("HTTP %s for run %s, retry %d", r.status_code, init, attempt + 1)
            time.sleep(10 * (attempt + 1))
            continue
        r.raise_for_status()
        try:
            df = _parse(r.json(), list(pts.index), init)
        except (ValueError, KeyError) as e:          # truncated or malformed body
            log.warning("run %s: bad response (%s), retry %d", init, e, attempt + 1)
            time.sleep(10 * (attempt + 1))
            continue
        if df[VARIABLES].isna().all(axis=1).mean() > 0.5:
            raise RunMissing("run returned without values")
        return df
    raise RuntimeError(f"run {init}: no response after retries")


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _extend_previous(root: Path, init: datetime, session: requests.Session) -> None:
    """Re-fetch the newest archived run before ``init`` long enough to cover the day after ``init``."""
    for back in range(1, 8):
        prev = init - timedelta(days=back)
        path = run_path(root, prev)
        if not path.exists():
            continue
        days = back + 2
        have = pd.read_parquet(path, columns=["valid_utc"])["valid_utc"].max()
        if have >= pd.Timestamp(prev) + pd.Timedelta(days=days) - pd.Timedelta(hours=1):
            return
        try:
            _save(fetch_run(prev, session, days=days), path)
            log.info("run %s re-fetched with %d days", prev, days)
        except RunMissing:
            pass
        return


def download(root: Path, start: date, end: date, pause: float = 7.0) -> int:
    """Fetch every missing run between the two dates (inclusive); returns the number of runs downloaded.

    When a run is not in the archive, a ``.missing`` marker is written and the newest archived run before it is
    fetched again with a longer horizon, so that the newest available run still covers the next delivery day.
    """
    session = requests.Session()
    n = 0
    day = start
    while day <= end:
        for hour in RUN_HOURS:
            init = datetime(day.year, day.month, day.day, hour, tzinfo=UTC)
            path = run_path(root, init)
            if path.exists() or missing_path(root, init).exists():
                continue
            t0 = time.monotonic()
            try:
                _save(fetch_run(init, session), path)
                n += 1
            except RunMissing as e:
                log.warning("run %s not in the archive (%s)", init, e)
                missing_path(root, init).write_text(f"{e}\n", encoding="utf-8")
                _extend_previous(root, init, session)
            if n and n % 25 == 0:
                log.info("%d runs downloaded, last %s", n, init)
            time.sleep(max(0.0, pause - (time.monotonic() - t0)))
        day += timedelta(days=1)
    return n


def load_runs(root: Path) -> pd.DataFrame:
    files = sorted((root / "raw" / "nwp" / MODEL).glob("*/run_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no NWP runs under {root / 'raw' / 'nwp'}; run `plfb nwp` first")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
