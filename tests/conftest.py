from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plfb import nwp as nwp_mod
from plfb.asof import Store, load_store, nwp_quarter_hours
from plfb.config import TZ, points, spec

SYN_START = date(2025, 9, 1)
SYN_DAYS = 40


def synthetic_tables(start: date = SYN_START, n_days: int = SYN_DAYS, seed: int = 1):
    """Small random actuals, plan and NWP with realistic publication rules (no download needed)."""
    rng = np.random.default_rng(seed)
    lo = pd.Timestamp(datetime(start.year, start.month, start.day)).tz_localize(TZ).tz_convert("UTC")
    hi = pd.Timestamp(datetime.combine(start + timedelta(days=n_days), datetime.min.time())).tz_localize(TZ)
    q = pd.date_range(lo, hi.tz_convert("UTC"), freq="15min", inclusive="left")
    n = len(q)
    bd = q.tz_convert(TZ).date
    pub_price = {d: pd.Timestamp(datetime.combine(d - timedelta(days=1), datetime.min.time())).tz_localize(TZ)
                 .tz_convert("UTC") + pd.Timedelta(hours=13, minutes=int(rng.integers(0, 120)))
                 for d in sorted(set(bd))}
    pub_plan = {d: v + pd.Timedelta(hours=2, minutes=int(rng.integers(0, 300))) for d, v in pub_price.items()}
    actuals = pd.DataFrame({
        "start_utc": q,
        "wind": np.abs(np.cumsum(rng.normal(0, 80, n))) + 500,
        "pv": np.clip(np.sin(np.linspace(0, n_days * 2 * np.pi, n)) * 8000, 0, None) + rng.normal(0, 50, n).clip(0),
        "demand": 18000 + 3000 * np.sin(np.linspace(0, n_days * 2 * np.pi, n)) + rng.normal(0, 200, n),
        "price": rng.normal(450, 150, n),
        "cen": rng.normal(450, 250, n),
        "price_available_utc": [pub_price[d] for d in bd],
    })
    actuals["start_utc"] = actuals["start_utc"].dt.as_unit("ns")
    actuals["price_available_utc"] = pd.to_datetime(actuals["price_available_utc"], utc=True).dt.as_unit("ns")
    plan = pd.DataFrame({
        "start_utc": actuals["start_utc"],
        "wind": actuals["wind"] + rng.normal(0, 300, n),
        "pv": actuals["pv"] * 0.95,
        "demand": actuals["demand"] + rng.normal(0, 300, n),
        "available_utc": pd.to_datetime([pub_plan[d] for d in bd], utc=True).as_unit("ns"),
    })
    rows = []
    pts = list(points().index)
    for k in range(-2, n_days + 1):
        init = pd.Timestamp(lo.normalize() + pd.Timedelta(days=k))
        valid = pd.date_range(init, periods=48, freq="h")
        for p in pts:
            d = {"init_utc": init, "point": p, "valid_utc": valid}
            for v in nwp_mod.VARIABLES:
                d[v] = rng.uniform(0, 15, 48) if "radiation" not in v else rng.uniform(0, 600, 48)
            rows.append(pd.DataFrame(d))
    nwp = pd.concat(rows, ignore_index=True)
    nwp.insert(1, "available_utc", nwp["init_utc"] + spec().nwp_delay)
    return actuals, plan, nwp


@pytest.fixture(scope="session")
def syn_tables():
    return synthetic_tables()


@pytest.fixture(scope="session")
def syn_store(syn_tables):
    actuals, plan, nwp = syn_tables
    return Store.from_tables(actuals, plan, nwp_quarter_hours(nwp))


@pytest.fixture(scope="session")
def real_store():
    root = Path(os.environ.get("PLFB_DATA", "data"))
    if not (root / "bench" / "nwp15.parquet").exists():
        pytest.skip("benchmark data not built (run `plfb pse`, `plfb nwp`, `plfb build`)")
    return load_store(root)
