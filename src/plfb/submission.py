"""Submission format.

A submission is one or more CSV or Parquet files with one row per forecast:

| column | meaning |
|---|---|
| ``track`` | ``day_ahead`` or ``intraday`` |
| ``variable`` | ``wind``, ``pv``, ``demand`` or ``price`` |
| ``origin_utc`` | issue time of the forecast (UTC, ISO 8601 in CSV, e.g. ``2025-10-01T08:00:00Z``) |
| ``target_utc`` | start of the 15-minute interval forecast (UTC) |
| ``point`` | point forecast (MW or PLN/MWh) |
| ``q10`` .. ``q90`` | optional: the 0.1, 0.2, ..., 0.9 quantiles; all nine or none |

Rows must match the task grid of the split exactly (``plfb.tasks``); a task may be left out entirely.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from plfb import tasks
from plfb.config import spec

KEYS = ["track", "variable", "origin_utc", "target_utc"]
QCOLS = [f"q{int(round(q * 100))}" for q in spec().quantiles]


class SubmissionError(ValueError):
    pass


def read(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        p = Path(p)
        if p.suffix == ".parquet":
            df = pd.read_parquet(p)
        elif p.suffix in (".csv", ".gz"):
            df = pd.read_csv(p)
        else:
            raise SubmissionError(f"{p}: use .parquet or .csv")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    missing = [c for c in [*KEYS, "point"] if c not in df.columns]
    if missing:
        raise SubmissionError(f"missing columns {missing}")
    for c in ("origin_utc", "target_utc"):
        df[c] = pd.to_datetime(df[c], utc=True).dt.as_unit("ns")
    has_q = [c for c in QCOLS if c in df.columns]
    if has_q and len(has_q) != len(QCOLS):
        raise SubmissionError(f"quantile columns must be all of {QCOLS} or none; found {has_q}")
    return df


def check(df: pd.DataFrame, split: str, allow_missing: bool = False) -> dict[tuple[str, str], pd.DataFrame]:
    """Split a submission by task and check each task against the grid of ``split``.

    Returns {(track, variable): frame aligned with ``tasks.rows(track, split)``}. With ``allow_missing`` (only for
    references), rows may be absent or NaN and are then excluded from that model's scores.
    """
    out = {}
    valid_tasks = spec().tasks()
    for (track, var), part in df.groupby(["track", "variable"], sort=False):
        if (track, var) not in valid_tasks:
            raise SubmissionError(f"unknown task {track}/{var}")
        grid = tasks.rows(track, split)[["origin_utc", "target_utc", "lead_minutes", "delivery_day"]]
        if part.duplicated(["origin_utc", "target_utc"]).any():
            raise SubmissionError(f"{track}/{var}: duplicated rows")
        merged = grid.merge(part.drop(columns=["track", "variable"]), on=["origin_utc", "target_utc"], how="left",
                            indicator=True)
        extra = len(part) - int((merged["_merge"] == "both").sum())
        if extra:
            raise SubmissionError(f"{track}/{var}: {extra} rows are not in the {split} grid (wrong origin or target)")
        absent = merged["point"].isna()
        if absent.any() and not allow_missing:
            raise SubmissionError(f"{track}/{var}: {int(absent.sum())} of {len(grid)} rows missing or NaN")
        if not np.isfinite(merged.loc[~absent, "point"].to_numpy(float)).all():
            raise SubmissionError(f"{track}/{var}: non-finite point forecasts")
        if QCOLS[0] in merged.columns:
            qv = merged.loc[~absent, QCOLS].to_numpy(float)
            if not np.isfinite(qv).all():
                raise SubmissionError(f"{track}/{var}: non-finite quantiles")
            if (np.diff(qv, axis=1) < -1e-9).any():
                raise SubmissionError(f"{track}/{var}: quantiles are not increasing")
        out[(track, var)] = merged.drop(columns="_merge")
    return out


def write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    for c in ["point", *[q for q in QCOLS if q in df.columns]]:
        df[c] = df[c].astype("float32")
    df.to_parquet(path, index=False, compression="zstd")
