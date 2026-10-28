"""Canonical tables of the benchmark and their checksums.

``plfb build`` turns the raw downloads into three tables under ``<data>/bench``:

- ``actuals.parquet``: one row per quarter-hour (``start_utc``) with the targets ``wind``, ``pv``, ``demand``,
  ``price`` (MW, PLN/MWh), the imbalance price ``cen`` (evaluation only) and ``price_available_utc``;
- ``plan.parquet``: the PSE day-ahead plan (PKD) for ``wind``, ``pv``, ``demand`` with ``available_utc``;
- ``nwp.parquet``: hourly ECMWF IFS forecasts per run and point with ``available_utc``;
- ``nwp15.parquet``: derived 15-minute NWP features per run (``plfb.asof.nwp_quarter_hours``).

The checksums are SHA-256 digests of a canonical text form (fixed column order, rounded values) per table and month,
so that a re-download can be compared with the published benchmark data month by month; PSE republishes some
reports after corrections, and a mismatch tells you which months changed.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from plfb import nwp as nwp_mod
from plfb.config import BENCH_DIR, QH, spec
from plfb.pse_data import raw_dir

log = logging.getLogger(__name__)

TABLES = ("actuals", "plan", "nwp")
ACTUAL_COLS = ["wind", "pv", "demand", "price", "cen"]
PLAN_COLS = ["wind", "pv", "demand"]


def bench_dir(root: Path) -> Path:
    return root / "bench"


def _pub_by_start(starts: pd.Series, business_date: pd.Series, pub: pd.DataFrame) -> pd.Series:
    m = dict(zip(pub["business_date"], pub["publication_utc"], strict=True))
    out = business_date.map(m)
    if out.isna().any():
        raise ValueError(f"{int(out.isna().sum())} rows without a publication time")
    return pd.to_datetime(out, utc=True)


def build_actuals(root: Path) -> pd.DataFrame:
    raw = raw_dir(root)
    sys_ = pd.read_parquet(raw / "his-wlk-cal.parquet").set_index("start_utc")
    price = pd.read_parquet(raw / "csdac-pln.parquet").set_index("start_utc")
    cen = pd.read_parquet(raw / "crb-rozl.parquet").set_index("start_utc")
    pub = pd.read_parquet(raw / "csdac-pln_publication.parquet")
    df = pd.DataFrame({
        "wind": sys_["wi"], "pv": sys_["pv"], "demand": sys_["demand"],
        "price": price["csdac_pln"], "cen": cen["cen_cost"],
    })
    df.index.name = "start_utc"
    df = df.sort_index()
    bd = price["business_date"].reindex(df.index)
    df["price_available_utc"] = _pub_by_start(df.index.to_series(), bd, pub).to_numpy()
    df = df.reset_index()
    df["start_utc"] = df["start_utc"].dt.as_unit("ns")
    df["price_available_utc"] = df["price_available_utc"].dt.as_unit("ns")
    return df


def build_plan(root: Path) -> pd.DataFrame:
    raw = raw_dir(root)
    plan = pd.read_parquet(raw / "pdgopkd.parquet")
    pub = pd.read_parquet(raw / "pdgopkd_publication.parquet")
    df = pd.DataFrame({
        "start_utc": plan["start_utc"].dt.as_unit("ns"),
        "wind": plan["gen_wi"], "pv": plan["gen_fv"], "demand": plan["kse_pow_dem"],
    })
    df["available_utc"] = _pub_by_start(plan["start_utc"], plan["business_date"], pub).dt.as_unit("ns").to_numpy()
    return df.sort_values("start_utc").reset_index(drop=True)


def build_nwp(root: Path) -> pd.DataFrame:
    df = nwp_mod.load_runs(root)
    empty = df.groupby("init_utc")["wind_speed_100m"].apply(lambda v: v.isna().mean() > 0.5)
    if empty.any():
        log.warning("dropping %d runs without values: %s", int(empty.sum()), list(empty[empty].index))
        df = df[~df["init_utc"].isin(empty[empty].index)]
    df["init_utc"] = pd.to_datetime(df["init_utc"], utc=True).dt.as_unit("ns")
    df["valid_utc"] = pd.to_datetime(df["valid_utc"], utc=True).dt.as_unit("ns")
    df.insert(1, "available_utc", df["init_utc"] + spec().nwp_delay)
    return df.sort_values(["init_utc", "point", "valid_utc"]).reset_index(drop=True)


def _canonical_months(table: str, df: pd.DataFrame) -> dict[str, str]:
    if table == "nwp":
        key = df["init_utc"]
        cols = ["init_utc", "point", "valid_utc", *nwp_mod.VARIABLES]
        digits = 2
    else:
        key = df["start_utc"]
        cols = list(df.columns)
        digits = 3
    out = {}
    for month, part in df[cols].groupby(key.dt.strftime("%Y-%m"), sort=True):
        text = part.copy()
        for c in cols:
            if pd.api.types.is_datetime64_any_dtype(text[c]):
                text[c] = text[c].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            elif pd.api.types.is_float_dtype(text[c]):
                vals = np.round(text[c].astype("float64"), digits)
                text[c] = vals.map(lambda v: "" if np.isnan(v) else f"{v:.{digits}f}")
        body = text.to_csv(index=False, lineterminator="\n")
        out[month] = hashlib.sha256(body.encode()).hexdigest()
    return out


def checksums(root: Path) -> dict[str, dict[str, str]]:
    return {t: _canonical_months(t, load(root, t)) for t in TABLES}


def build(root: Path, write_checksums: bool = True) -> None:
    out = bench_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in (("actuals", build_actuals), ("plan", build_plan), ("nwp", build_nwp)):
        df = fn(root)
        df.to_parquet(out / f"{name}.parquet", index=False)
        log.info("%s: %d rows", name, len(df))
        if name == "nwp":
            from plfb.asof import nwp_quarter_hours
            q = nwp_quarter_hours(df)
            q.to_parquet(out / "nwp15.parquet", index=False)
            log.info("nwp15: %d rows (15-minute features per run)", len(q))
    load.cache_clear()
    if write_checksums:
        sums = checksums(root)
        text = json.dumps(sums, indent=1, sort_keys=True) + "\n"
        (BENCH_DIR / "checksums.json").write_text(text, encoding="utf-8", newline="\n")
        log.info("checksums written for %s", ", ".join(f"{k} ({len(v)} months)" for k, v in sums.items()))


def verify(root: Path, path: Path) -> bool:
    ref = json.loads(path.read_text(encoding="utf-8"))
    mine = checksums(root)
    ok = True
    for table, months in ref.items():
        bad = [m for m, h in months.items() if mine.get(table, {}).get(m) != h]
        missing = [m for m in months if m not in mine.get(table, {})]
        if bad:
            ok = False
            log.warning("%s: %d of %d months differ: %s%s", table, len(bad), len(months), ", ".join(bad[:12]),
                        " (missing: " + ", ".join(missing) + ")" if missing else "")
        else:
            log.info("%s: all %d months match", table, len(months))
    return ok


_cache: dict[tuple[str, str], pd.DataFrame] = {}


def load(root: Path, table: str) -> pd.DataFrame:
    key = (str(Path(root).resolve()), table)
    if key not in _cache:
        path = bench_dir(root) / f"{table}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `plfb pse`, `plfb nwp` and `plfb build` first")
        _cache[key] = pd.read_parquet(path)
    return _cache[key]


load.cache_clear = _cache.clear  # type: ignore[attr-defined]


def grid(start, end) -> pd.DatetimeIndex:
    """Quarter-hour starts (UTC) from ``start`` (inclusive) to ``end`` (exclusive)."""
    return pd.date_range(start, end, freq=QH, inclusive="left", tz="UTC")
