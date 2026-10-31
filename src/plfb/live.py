"""Monthly live track: the CPU reference models scored on every month after the frozen test year.

This is not part of benchmark v1, whose test year (October 2025 to September 2026) stays fixed. Each new month M is
forecast with the same tasks, as-of rules, features and metrics. The models are refitted once, at local midnight on
the first day of M, on the ``TRAIN_MONTHS`` months before it: a rolling window rather than v1's expanding one, so that
the monthly job downloads about 13 months of weather (about 3,300 Open-Meteo units) instead of the whole history.
Foundation models need a GPU and are not part of the live track.

A month is scored from day ``LAG_DAYS`` of the next month on, when PSE has republished the actuals. Imbalance prices
(CEN) can still be republished up to about 120 days after the end of a month (the longest republication pse-opendata
has observed for crb-rozl), so the imbalance cost of the last four months is preliminary.

Usage:
    python -m plfb live                    # the latest month that can be scored
    python -m plfb live --month 2026-10
"""
from __future__ import annotations

import json
import logging
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from plfb import dataset, evaluate, features, nwp, pse_data, submission, tasks
from plfb.asof import Store, _ns, load_store
from plfb.baselines import models as M
from plfb.baselines.run import RESID_DAYS, _predict
from plfb.config import spec

log = logging.getLogger(__name__)

MODELS = ("persistence", "naive_day", "naive_week", "pse_plan", "ridge", "gbm")
TRAIN_MONTHS = 12
LAG_DAYS = 5          # score month M from day 5 of M+1
WARMUP_DAYS = 16      # history before the training window for lags and the 14-day calibration ratio
KEEP = ("rows", "rows_expected", "mae", "rmse", "nmae", "bias", "crps", "coverage80", "imbalance_cost_pln_mwh")


def month_start(m: str) -> date:
    return date.fromisoformat(f"{m}-01")


def add_months(d: date, k: int) -> date:
    y, mo = divmod(d.month - 1 + k, 12)
    return date(d.year + y, mo + 1, 1)


def default_month(today: date) -> str:
    """The latest month whose actuals are final enough: the previous one from day ``LAG_DAYS`` on."""
    back = 1 if today.day >= LAG_DAYS else 2
    return add_months(date(today.year, today.month, 1), -back).strftime("%Y-%m")


def month_days(m: str) -> list[date]:
    lo = month_start(m)
    return tasks.days(lo, add_months(lo, 1) - timedelta(days=1))


def windows(m: str, train_months: int = TRAIN_MONTHS) -> dict[str, date]:
    lo = month_start(m)
    train = add_months(lo, -train_months)
    return {"train_start": train, "data_start": train - timedelta(days=WARMUP_DAYS), "month_start": lo,
            "month_end": add_months(lo, 1) - timedelta(days=1)}


def check_month(m: str, today: date) -> None:
    lo = month_start(m)
    first_live = add_months(spec().splits["test"][1].replace(day=1), 1)
    if lo < first_live:
        raise ValueError(f"{m} is inside benchmark v1; the live track starts with {first_live:%Y-%m}")
    if today < add_months(lo, 1) + timedelta(days=LAG_DAYS - 1):
        raise ValueError(f"{m} can be scored from {add_months(lo, 1) + timedelta(days=LAG_DAYS - 1)} on")


def download(root: Path, w: dict[str, date]) -> None:
    end = w["month_end"] + timedelta(days=1)          # intraday targets of the last evening run past midnight
    pse_data.download(root, end=end, start=w["data_start"])
    nwp.download(root, w["data_start"] - timedelta(days=1), end)
    dataset.build(root, write_checksums=False)


def forecast_month(store: Store, m: str, w: dict[str, date], models=MODELS) -> dict[str, pd.DataFrame]:
    """Forecasts of every task for the days of month ``m``, refitted at its first midnight on the rolling window."""
    levels = spec().quantiles
    T = int(_ns([tasks.local_midnight(w["month_start"])])[0])
    train_from = int(_ns([tasks.local_midnight(w["train_start"])])[0])
    out: dict[str, list[pd.DataFrame]] = {name: [] for name in models}
    for track, var in spec().tasks():
        rows = tasks.rows(track, tasks.days(w["train_start"], w["month_end"]))
        t0 = time.monotonic()
        X = features.build(store, rows, track, var)
        y = features.target(store, rows, var)
        yav = features.target_available(store, rows, var)
        groups = M.group_key(rows, track)
        origin = _ns(rows["origin_utc"])
        target = _ns(rows["target_utc"])
        day = rows["delivery_day"].to_numpy()
        sub = (pd.DatetimeIndex(rows["origin_utc"]).hour % 2 == 0) if track == "intraday" else np.ones(len(rows), bool)
        Xr = M.ridge_design(X)
        pred = (day >= w["month_start"]) & (day <= w["month_end"])
        train = (yav <= T) & np.isfinite(y) & (origin >= train_from)
        recent = train & (target >= T - RESID_DAYS * 86400 * 10**9)
        assert origin[train].max() < T, "a training row was issued after the refit time"
        for name in models:
            if name == "pse_plan" and var == "price":
                continue
            point, q = _predict(name, var, track, store, rows, X, Xr, y, train, recent, pred, sub, groups, levels)
            point, q = M.postprocess(var, point, q)
            frame = rows.loc[pred, ["origin_utc", "target_utc"]].copy()
            frame.insert(0, "variable", var)
            frame.insert(0, "track", track)
            frame["point"] = point
            if q is not None:
                for j, c in enumerate(submission.QCOLS):
                    frame[c] = q[:, j]
            out[name].append(frame)
        log.info("%s/%s: %d training rows, %d forecast (%.0f s)", track, var, int(train.sum()), int(pred.sum()),
                 time.monotonic() - t0)
    return {name: pd.concat(v, ignore_index=True) for name, v in out.items() if v}


def summarise(scores: dict) -> dict:
    out = {}
    for task, s in scores["tasks"].items():
        e = {k: s[k] for k in KEEP if k in s}
        if "reference" in s:
            r = s["reference"]
            e.update(ref=r["name"], skill=r["skill_mae"], skill_ci=r["skill_ci95"])
        if "battery" in s:
            e["battery_share"] = s["battery"]["share_of_perfect"]
        out[task] = e
    return out


def run(root: Path, docs: Path, month: str | None = None, train_months: int = TRAIN_MONTHS,
        download_data: bool = True, today: date | None = None, check: bool = True) -> Path:
    """Forecast and score one month; ``check=False`` allows months inside v1, only to test the pipeline."""
    today = today or datetime.now(UTC).date()
    month = month or default_month(today)
    if check:
        check_month(month, today)
    w = windows(month, train_months)
    log.info("live month %s: training %s..%s", month, w["train_start"], w["month_start"] - timedelta(days=1))
    if download_data:
        download(root, w)
    dataset.load.cache_clear()
    store = load_store(root)
    forecasts = forecast_month(store, month, w)
    days = month_days(month)
    result = {
        "month": month, "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "benchmark_version": spec().raw["benchmark"]["version"],
        "train_window": [w["train_start"].isoformat(), (w["month_start"] - timedelta(days=1)).isoformat()],
        "refit": "once, at local midnight on the first day of the month, on the previous "
                 f"{train_months} months",
        "models": {},
    }
    for name, df in forecasts.items():
        sc = evaluate.score(root, df, name, days, allow_missing=(name == "pse_plan"), label=month, store=store)
        result["models"][name] = summarise(sc)
    out = docs / "live" / f"{month}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    build_index(docs)
    log.info("live scores for %s -> %s", month, out)
    return out


def build_index(docs: Path) -> dict:
    """``docs/live/index.json``: every scored month, newest last, for the site."""
    folder = docs / "live"
    months = []
    for p in sorted(folder.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9].json")):
        months.append(json.loads(p.read_text(encoding="utf-8")))
    index = {
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "models": list(MODELS), "train_months": TRAIN_MONTHS, "months": months,
    }
    folder.mkdir(parents=True, exist_ok=True)
    text = json.dumps(index, indent=1) + "\n"
    (folder / "index.json").write_text(text, encoding="utf-8", newline="\n")
    return index

