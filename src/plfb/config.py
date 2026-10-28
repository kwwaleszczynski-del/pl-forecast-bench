"""Benchmark definition (benchmark/*.toml) and data locations.

The data directory is ``--data-dir`` on the command line, else the ``PLFB_DATA`` environment variable, else ``data``
in the current directory.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from datetime import date, time
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

TZ = "Europe/Warsaw"
QH = pd.Timedelta(minutes=15)
REPO = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO / "benchmark"
VARIABLES = ("wind", "pv", "demand", "price")
TRACKS = ("day_ahead", "intraday")


def data_dir(arg: str | Path | None = None) -> Path:
    return Path(arg or os.environ.get("PLFB_DATA") or "data")


@dataclass(frozen=True)
class Spec:
    raw: dict

    @property
    def splits(self) -> dict[str, tuple[date, date]]:
        s = self.raw["splits"]
        return {k: (date.fromisoformat(s[k][0]), date.fromisoformat(s[k][1])) for k in ("train", "val", "test")}

    @property
    def data_start(self) -> date:
        return date.fromisoformat(self.raw["splits"]["data_start"])

    @property
    def quantiles(self) -> list[float]:
        return list(self.raw["quantiles"]["levels"])

    @property
    def issue_local(self) -> time:
        return time.fromisoformat(self.raw["tracks"]["day_ahead"]["issue_local"])

    def track_variables(self, track: str) -> list[str]:
        return list(self.raw["tracks"][track]["variables"])

    @property
    def intraday(self) -> dict:
        return self.raw["tracks"]["intraday"]

    @property
    def obs_lag(self) -> pd.Timedelta:
        return pd.Timedelta(minutes=self.raw["availability"]["observation_lag_minutes"])

    @property
    def nwp_delay(self) -> pd.Timedelta:
        return pd.Timedelta(hours=self.raw["availability"]["nwp_delay_hours"])

    def variable(self, name: str) -> dict:
        return self.raw["variables"][name]

    def tasks(self) -> list[tuple[str, str]]:
        return [(t, v) for t in TRACKS for v in self.track_variables(t)]


@cache
def spec() -> Spec:
    with open(BENCH_DIR / "spec.toml", "rb") as f:
        return Spec(tomllib.load(f))


@cache
def points() -> pd.DataFrame:
    with open(BENCH_DIR / "points.toml", "rb") as f:
        pts = tomllib.load(f)["points"]
    df = pd.DataFrame(pts).set_index("id")
    for col, w in (("wind_mw", "w_wind"), ("pv_mw", "w_pv"), ("population", "w_pop")):
        df[w] = df[col] / df[col].sum()
    return df


@cache
def capacity_table() -> dict[str, pd.Series]:
    """Dated installed capacity (MW) of wind and PV, used only to normalise errors (never as a model input)."""
    with open(BENCH_DIR / "capacity.toml", "rb") as f:
        raw = tomllib.load(f)
    out = {}
    for var in ("wind", "pv"):
        rows = raw[var]
        out[var] = pd.Series([r["mw"] for r in rows],
                             index=pd.to_datetime([r["date"] for r in rows]).tz_localize(TZ)).sort_index()
    return out


def installed_capacity(var: str, times: pd.DatetimeIndex) -> pd.Series:
    """Capacity at each time, linear between the dated values and flat beyond them."""
    table = capacity_table()[var]
    x = table.index.tz_convert("UTC").as_unit("s").asi8.astype(float)
    t = pd.DatetimeIndex(times).tz_convert("UTC").as_unit("s").asi8.astype(float)
    return pd.Series(np.interp(t, x, table.to_numpy(float)), index=times)
