"""Run the reference models over the validation and test periods with quarterly refits.

At each refit time T (local midnight on the first day of a quarter) a model is trained on every row of its task from
the start of the training period whose label was public at T, and then forecasts the rows of the quarter that starts
at T, each with its own as-of features. Validation uses one refit (1 July 2025), the test year four.

Usage:
    python -m plfb baselines --splits val test --jobs 2
"""
from __future__ import annotations

import logging
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from plfb import features, submission, tasks
from plfb.asof import Store, _ns, load_store
from plfb.baselines import models as M
from plfb.config import spec

log = logging.getLogger(__name__)

POINT_ONLY = ("pse_plan",)
ALL_MODELS = ("persistence", "naive_day", "naive_week", "pse_plan", "ridge", "gbm")
RESID_DAYS = 90


def refit_days(split: str) -> list[date]:
    lo, hi = spec().splits[split]
    out = [lo]
    m = date(lo.year, lo.month, 1)
    while True:
        m = date(m.year + (m.month > 9), (m.month + 2) % 12 + 1, 1)
        if m > hi:
            return out
        out.append(m)


def _ns_local_midnight(d: date) -> int:
    return int(_ns([tasks.local_midnight(d)])[0])


def run_task(store: Store, track: str, var: str, splits: list[str], models: list[str]) -> dict[str, pd.DataFrame]:
    levels = spec().quantiles
    lo = spec().splits["train"][0]
    hi = max(spec().splits[s][1] for s in splits)
    rows = tasks.rows(track, tasks.days(lo, hi))
    t0 = time.monotonic()
    X = features.build(store, rows, track, var)
    y = features.target(store, rows, var)
    yav = features.target_available(store, rows, var)
    log.info("%s/%s: %d rows, %d features (%.0f s)", track, var, len(rows), X.shape[1], time.monotonic() - t0)
    groups = M.group_key(rows, track)
    origin_ns = _ns(rows["origin_utc"])
    target_ns = _ns(rows["target_utc"])
    day = rows["delivery_day"].to_numpy()
    sub = (pd.DatetimeIndex(rows["origin_utc"]).hour % 2 == 0) if track == "intraday" else np.ones(len(rows), bool)
    Xr = M.ridge_design(X) if {"ridge", "gbm"} & set(models) else None

    out: dict[str, list[pd.DataFrame]] = {m: [] for m in models}
    for split in splits:
        refits = refit_days(split)
        ends = [*refits[1:], spec().splits[split][1] + pd.Timedelta(days=1).to_pytimedelta()]
        for start, end in zip(refits, ends, strict=True):
            T = _ns_local_midnight(start)
            pred = (day >= start) & (day < end)
            train = (yav <= T) & np.isfinite(y)
            recent = train & (target_ns >= T - RESID_DAYS * 86400 * 10**9)
            assert origin_ns[train].max() < T, "a training row was issued after the refit time"
            log.info("  refit %s: %d training rows, %d to forecast", start, int(train.sum()), int(pred.sum()))
            for m in models:
                if m == "pse_plan" and var == "price":
                    continue
                point, q = _predict(m, var, track, store, rows, X, Xr, y, train, recent, pred, sub, groups, levels)
                point, q = M.postprocess(var, point, q)
                frame = rows.loc[pred, ["origin_utc", "target_utc"]].copy()
                frame.insert(0, "variable", var)
                frame.insert(0, "track", track)
                frame["point"] = point
                if q is not None:
                    for j, c in enumerate(submission.QCOLS):
                        frame[c] = q[:, j]
                frame["split"] = split
                out[m].append(frame)
    return {m: pd.concat(v, ignore_index=True) for m, v in out.items() if v}


def _predict(m, var, track, store, rows, X, Xr, y, train, recent, pred, sub, groups, levels):
    n = len(levels)
    if m in M.NAIVE:
        p_all = M.naive_point(X, m, var)
        table = M.residual_quantiles((y - p_all)[recent], groups[recent], levels)
        return p_all[pred], M.apply_quantiles(p_all[pred], groups[pred], table, n)
    if m == "pse_plan":
        if track == "day_ahead":
            # reference only: published after the gate, read without the origin guard
            t = _ns(rows.loc[pred, "target_utc"])
            pos = (t - store.t0) // (15 * 60 * 10**9)
            return store.values[f"plan_{var}"][pos], None
        return X.loc[pred, "plan"].to_numpy(), None
    if m == "ridge":
        fit = M.fit_ridge(Xr[train], y[train])
        p_all = np.full(len(y), np.nan)
        idx = train | pred
        p_all[idx] = fit.predict(Xr[idx])
        table = M.residual_quantiles((y - p_all)[recent], groups[recent], levels)
        return p_all[pred], M.apply_quantiles(p_all[pred], groups[pred], table, n)
    if m == "gbm":
        tr = train & sub
        if not M.GBM_FROM_RIDGE:
            mean = M.fit_gbm(X[tr], y[tr]).predict(X[pred])
            return mean, np.column_stack([M.fit_gbm(X[tr], y[tr], a).predict(X[pred]) for a in levels])
        # boosting from the ridge prediction: trees learn the non-linear residual, the linear base extrapolates
        # (seasons outside the training window, a growing PV fleet)
        base = M.fit_ridge(Xr[tr], y[tr])
        b_tr, b_pr = base.predict(Xr[tr]), base.predict(Xr[pred])
        mean = b_pr + M.fit_gbm(X[tr], y[tr], init_score=b_tr).predict(X[pred])
        q = np.column_stack([b_pr + M.fit_gbm(X[tr], y[tr], a, init_score=b_tr).predict(X[pred]) for a in levels])
        return mean, q
    raise ValueError(f"unknown model {m}")


def run_all(root: Path, out_dir: Path, models: list[str] | None = None, splits: list[str] = ("test",),
            jobs: int = 2) -> None:
    models = list(models or ALL_MODELS)
    store = load_store(root)
    results: dict[str, list[pd.DataFrame]] = {m: [] for m in models}
    for track, var in spec().tasks():
        res = run_task(store, track, var, list(splits), models)
        for m, df in res.items():
            results[m].append(df)
    for m, frames in results.items():
        df = pd.concat(frames, ignore_index=True)
        for split, part in df.groupby("split"):
            path = out_dir / m / f"{split}.parquet"
            submission.write(part.drop(columns="split"), path)
            log.info("%s %s: %d rows -> %s", m, split, len(part), path)
