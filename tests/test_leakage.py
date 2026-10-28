"""No information from the future: features and forecasts must not change when everything that was not public at the
origin is replaced by noise, and the building blocks must respect publication times."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from plfb import features, tasks
from plfb.asof import _ns
from plfb.baselines import models as M
from plfb.config import TZ, spec

from conftest import SYN_DAYS, SYN_START

TASKS = spec().tasks()


def _origins_rows(track: str, ds: list[date], n: int, seed: int) -> list[pd.DataFrame]:
    rows = tasks.rows(track, ds)
    origins = rows["origin_utc"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    pick = rng.choice(origins, size=min(n, len(origins)), replace=False)
    return [rows[rows["origin_utc"] == o].reset_index(drop=True) for o in pick]


def _assert_same(a: pd.DataFrame, b: pd.DataFrame, what: str) -> None:
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-9, obj=what)


@pytest.mark.parametrize("track,var", TASKS)
def test_features_ignore_the_future_synthetic(syn_store, track, var):
    ds = tasks.days(SYN_START + timedelta(days=16), SYN_START + timedelta(days=SYN_DAYS - 2))
    for k, rows in enumerate(_origins_rows(track, ds, 6, seed=3)):
        o = rows["origin_utc"].iloc[0]
        clean = features.build(syn_store, rows, track, var)
        dirty = features.build(syn_store.poisoned(o, seed=k), rows, track, var)
        _assert_same(clean, dirty, f"{track}/{var} at {o}")
        assert clean.notna().any().mean() > 0.9, "features are mostly empty, the test would prove nothing"


@pytest.mark.parametrize("track,var", TASKS)
def test_features_ignore_the_future_real(real_store, track, var):
    lo, hi = spec().splits["test"]
    ds = tasks.days(lo, hi)
    for k, rows in enumerate(_origins_rows(track, ds, 8, seed=11)):
        o = rows["origin_utc"].iloc[0]
        clean = features.build(real_store, rows, track, var)
        dirty = features.build(real_store.poisoned(o, seed=k), rows, track, var)
        _assert_same(clean, dirty, f"{track}/{var} at {o}")


def test_poisoning_changes_something(syn_store):
    """The test above is only meaningful if poisoning really alters the data after the origin."""
    o = pd.Timestamp("2025-09-20T08:00:00Z")
    p = syn_store.poisoned(o)
    t = syn_store.t0 + np.arange(len(syn_store.values["wind"])) * 15 * 60 * 10**9
    late = syn_store.avail["wind"] > _ns([o])[0]
    assert late.any() and (~late).any()
    assert not np.allclose(p.values["wind"][late], syn_store.values["wind"][late])
    assert np.array_equal(p.values["wind"][~late], syn_store.values["wind"][~late])
    assert t[late].min() > _ns([o])[0] - 2 * 3600 * 10**9


def test_lookup_hides_unpublished_values(syn_store):
    o = pd.Timestamp("2025-09-20T08:00:00Z")
    on = _ns([o])[0]
    t = syn_store.t0 + np.arange(len(syn_store.values["wind"])) * 15 * 60 * 10**9
    for name in ("wind", "pv", "demand", "price", "plan_wind"):
        got = syn_store.lookup(name, t, np.full(len(t), on))
        late = syn_store.avail[name] > on
        assert np.isnan(got[late]).all(), name
        assert np.array_equal(got[~late], syn_store.values[name][~late]), name


def test_observation_lag(syn_store):
    """An actual is public one observation lag after its interval ends, not earlier."""
    o = pd.Timestamp("2025-09-20T08:00:00Z")
    last = syn_store.last_time("wind", [o])[0]
    lag = spec().obs_lag.value
    assert last + 15 * 60 * 10**9 + lag <= _ns([o])[0]
    assert np.isfinite(syn_store.lookup("wind", [last], [o])).all()
    assert np.isnan(syn_store.lookup("wind", [last + 15 * 60 * 10**9], [o])).all()


def test_day_ahead_uses_the_previous_morning_run(syn_store):
    for d in tasks.days(SYN_START + timedelta(days=3), SYN_START + timedelta(days=10)):
        o = tasks.day_ahead_origin(d)
        k = syn_store.run_for([o])[0]
        init = pd.Timestamp(syn_store.nwp_init[k], tz="UTC")
        assert syn_store.nwp_avail[k] <= _ns([o])[0]
        assert init == pd.Timestamp(d - timedelta(days=1)).tz_localize("UTC"), (d, init)


def test_day_ahead_origin_before_gate_closure():
    gate = 12 * 60
    for d in tasks.days(date(2025, 10, 20), date(2025, 11, 5)) + tasks.days(date(2026, 3, 25), date(2026, 4, 2)):
        o = tasks.day_ahead_origin(d).tz_convert(TZ)
        assert o.date() == d - timedelta(days=1)
        assert o.hour * 60 + o.minute < gate


def test_day_ahead_plan_is_published_after_the_origin(real_store):
    """Why the PSE plan is only a reference day-ahead: it is never public at the benchmark's issue time."""
    lo, hi = spec().splits["test"]
    rows = tasks.rows("day_ahead", tasks.days(lo, hi))
    got = real_store.lookup("plan_wind", rows["target_utc"], rows["origin_utc"])
    assert np.isnan(got).all()


def test_refit_trains_only_on_public_labels(syn_store):
    """A model refitted at T is identical whether or not the data after T are noise."""
    track, var = "intraday", "wind"
    rows = tasks.rows(track, tasks.days(SYN_START + timedelta(days=15), SYN_START + timedelta(days=30)))
    T = tasks.local_midnight(SYN_START + timedelta(days=25)).tz_convert("UTC")
    preds = []
    for store in (syn_store, syn_store.poisoned(T, seed=5)):
        X = features.build(store, rows, track, var)
        y = features.target(store, rows, var)
        yav = features.target_available(store, rows, var)
        train = (yav <= _ns([T])[0]) & np.isfinite(y)
        Xr = M.ridge_design(X)
        fit = M.fit_ridge(Xr[train], y[train])
        preds.append(fit.predict(Xr[train].iloc[:500]))
        assert rows.loc[train, "origin_utc"].max() < T
    np.testing.assert_allclose(preds[0], preds[1])


@pytest.mark.parametrize("track,var", TASKS)
def test_foundation_contexts_ignore_the_future(syn_store, track, var):
    """The foundation models get their context and NWP covariates through the same guarded store."""
    from plfb.baselines import foundation as F
    ds = tasks.days(SYN_START + timedelta(days=30), SYN_START + timedelta(days=SYN_DAYS - 2))
    hourly = track == "day_ahead"
    step = F.NS_H if hourly else F.NS_QH
    length = 48 if hourly else 192
    for k, rows in enumerate(_origins_rows(track, ds, 4, seed=7)):
        o = pd.DatetimeIndex(rows["origin_utc"].iloc[:1])
        res = []
        for store in (syn_store, syn_store.poisoned(o[0], seed=k)):
            ctx, last = F.contexts(store, var, o, length, step)
            fut = last[:, None] + step * np.arange(1, 30)[None, :]
            past = last[:, None] - step * np.arange(length - 1, -1, -1)[None, :]
            res.append((ctx, last, F._covariates(store, var, o, past), F._covariates(store, var, o, fut)))
        for a, b in zip(res[0], res[1], strict=True):
            np.testing.assert_array_equal(a, b)
        assert np.isfinite(res[0][0]).mean() > 0.9
        # the newest context value is public at the origin (for the price it lies after the origin: D-1 is
        # published on D-2) and the next one is not
        pos = (res[0][1] - syn_store.t0) // F.NS_QH
        assert (syn_store.avail[var][pos] <= _ns(o)[0]).all()
        if not hourly:
            assert (syn_store.avail[var][pos + 1] > _ns(o)[0]).all()
