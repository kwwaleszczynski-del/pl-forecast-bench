"""PSE reports used by the benchmark, downloaded with pse-opendata into ``<data>/raw/pse``.

- ``his-wlk-cal``: actual demand, wind and PV (the targets), 15 minutes;
- ``csdac-pln``: SDAC day-ahead price for Poland (the price target), 15 minutes (hourly product until 2025-09-30);
- ``pdgopkd``: the PSE day-ahead coordinated plan (PKD): planned demand, wind and PV, the official reference;
- ``crb-rozl``: imbalance price CEN, only for the imbalance-cost metric;
- publication timestamps of ``csdac-pln`` and ``pdgopkd`` per business day, which set when they become visible.

Information obtained from www.pse.pl / api.raporty.pse.pl (PSE S.A. reports). The benchmark does not redistribute
these data; ``plfb build`` derives its tables from them and ``plfb verify`` checks them against published checksums.

Usage:
    python -m plfb pse --end 2026-10-01
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import pse_opendata as pse

from plfb.config import spec

log = logging.getLogger(__name__)

REPORTS = {
    "his-wlk-cal": ["demand", "wi", "pv"],
    "csdac-pln": ["csdac_pln"],
    "pdgopkd": ["kse_pow_dem", "gen_wi", "gen_fv"],
    "crb-rozl": ["cen_cost"],
}
PUBLICATION_REPORTS = ("csdac-pln", "pdgopkd")


def raw_dir(root: Path) -> Path:
    return root / "raw" / "pse"


def _month_starts(start: date, end: date) -> list[tuple[date, date]]:
    out = []
    m = date(start.year, start.month, 1)
    while m <= end:
        nxt = date(m.year + (m.month == 12), m.month % 12 + 1, 1)
        out.append((max(m, start), min(end, nxt - pd.Timedelta(days=1).to_pytimedelta())))
        m = nxt
    return out


def publication_times(client: pse.Client, report: str, start: date, end: date) -> pd.DataFrame:
    """Latest publication timestamp (UTC) of each business day of a report."""
    rows = []
    for lo, hi in _month_starts(start, end):
        recs = client.records(report, filter=f"business_date ge '{lo:%Y-%m-%d}' and business_date le '{hi:%Y-%m-%d}'",
                              select=["business_date", "publication_ts_utc"])
        rows += [(r["business_date"], r["publication_ts_utc"]) for r in recs]
    df = pd.DataFrame(rows, columns=["business_date", "publication_utc"])
    df["publication_utc"] = pd.to_datetime(df["publication_utc"]).dt.tz_localize("UTC")
    n_versions = df.groupby("business_date")["publication_utc"].nunique()
    if (n_versions > 1).any():
        log.warning("%s: %d days with rows of different publication times", report, int((n_versions > 1).sum()))
    out = df.groupby("business_date", as_index=False)["publication_utc"].max()
    out["business_date"] = pd.to_datetime(out["business_date"]).dt.date
    return out


def download(root: Path, end: date = date(2026, 10, 1)) -> None:
    start = spec().data_start
    out = raw_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    client = pse.Client(cache_dir=root / "raw" / "pse-cache")
    for report, fields in REPORTS.items():
        log.info("%s %s..%s", report, start, end)
        df = client.get(report, start, end, fields=fields)
        df.to_parquet(out / f"{report}.parquet", index=False)
        log.info("  %d rows, %s..%s", len(df), df["start_utc"].min(), df["start_utc"].max())
    for report in PUBLICATION_REPORTS:
        pub = publication_times(client, report, start, end)
        pub.to_parquet(out / f"{report}_publication.parquet", index=False)
        log.info("%s publication times: %d days", report, len(pub))
