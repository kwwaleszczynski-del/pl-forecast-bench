"""The forecast rows of each track: which origins exist and which quarter-hours each origin must forecast.

A row is (``origin_utc``, ``target_utc``), where ``target_utc`` is the start of a 15-minute interval.

- Day-ahead: one origin per delivery day D at D-1 ``issue_local`` (Polish time); targets are all quarter-hours of D
  in Polish local time (92, 96 or 100 on DST days).
- Intraday: an origin at every full hour; targets start ``first_lead_minutes`` .. ``last_lead_minutes`` after it.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from plfb.config import QH, TZ, spec


def days(lo: date, hi: date) -> list[date]:
    return [lo + timedelta(days=i) for i in range((hi - lo).days + 1)]


def split_days(split: str) -> list[date]:
    lo, hi = spec().splits[split]
    return days(lo, hi)


def local_midnight(d: date) -> pd.Timestamp:
    return pd.Timestamp(datetime(d.year, d.month, d.day)).tz_localize(TZ)


def day_ahead_origin(d: date) -> pd.Timestamp:
    """Issue time (UTC) of the day-ahead forecast for delivery day ``d``."""
    prev = d - timedelta(days=1)
    t = spec().issue_local
    return pd.Timestamp(datetime(prev.year, prev.month, prev.day, t.hour, t.minute)).tz_localize(TZ).tz_convert("UTC")


def day_ahead_rows(ds: list[date]) -> pd.DataFrame:
    parts = []
    for d in ds:
        lo, hi = local_midnight(d).tz_convert("UTC"), local_midnight(d + timedelta(days=1)).tz_convert("UTC")
        targets = pd.date_range(lo, hi, freq=QH, inclusive="left")
        parts.append(pd.DataFrame({"origin_utc": day_ahead_origin(d), "target_utc": targets, "delivery_day": d}))
    out = pd.concat(parts, ignore_index=True)
    return _finish(out)


def intraday_origins(ds: list[date]) -> pd.DatetimeIndex:
    lo = local_midnight(min(ds)).tz_convert("UTC")
    hi = local_midnight(max(ds) + timedelta(days=1)).tz_convert("UTC")
    step = pd.Timedelta(minutes=spec().intraday["origin_step_minutes"])
    out = pd.date_range(lo, hi, freq=step, inclusive="left")
    return out[pd.Index(out.tz_convert(TZ).date).isin(set(ds))]


def intraday_leads() -> np.ndarray:
    s = spec().intraday
    return np.arange(s["first_lead_minutes"], s["last_lead_minutes"] + 1, 15)


def intraday_rows(ds: list[date], origins: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    origins = intraday_origins(ds) if origins is None else origins
    leads = intraday_leads()
    out = pd.DataFrame({"origin_utc": pd.DatetimeIndex(origins).repeat(len(leads))})
    out["target_utc"] = out["origin_utc"] + pd.to_timedelta(np.tile(leads, len(origins)), unit="min")
    out["delivery_day"] = out["origin_utc"].dt.tz_convert(TZ).dt.date
    return _finish(out)


def _finish(out: pd.DataFrame) -> pd.DataFrame:
    out["origin_utc"] = pd.to_datetime(out["origin_utc"], utc=True).dt.as_unit("ns")
    out["target_utc"] = pd.to_datetime(out["target_utc"], utc=True).dt.as_unit("ns")
    out["lead_minutes"] = ((out["target_utc"] - out["origin_utc"]) / pd.Timedelta(minutes=1)).astype(int)
    return out


def rows(track: str, split: str | list[date]) -> pd.DataFrame:
    ds = split_days(split) if isinstance(split, str) else split
    return day_ahead_rows(ds) if track == "day_ahead" else intraday_rows(ds)
