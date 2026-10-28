from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from plfb import metrics, submission, tasks
from plfb.config import installed_capacity, spec
from plfb.evaluate import bootstrap_skill


def test_point_metrics():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    f = np.array([1.0, 3.0, 1.0, 4.0])
    assert metrics.mae(y, f) == pytest.approx(0.75)
    assert metrics.rmse(y, f) == pytest.approx(np.sqrt(5 / 4))
    assert metrics.nmae(y, f, 10.0) == pytest.approx(7.5)


def test_pinball_and_crps_of_a_point_forecast_equal_mae():
    """With every quantile equal to the point forecast and levels symmetric around 0.5, 2 * mean pinball = MAE."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=500)
    f = rng.normal(size=500)
    levels = spec().quantiles
    q = np.repeat(f[:, None], len(levels), axis=1)
    assert metrics.crps_quantile(y, q, levels) == pytest.approx(metrics.mae(y, f), rel=1e-12)


def test_crps_close_to_closed_form_for_a_normal_forecast():
    from scipy.stats import norm
    rng = np.random.default_rng(1)
    y = rng.normal(size=20000)
    levels = np.arange(1, 100) / 100
    q = np.repeat(norm.ppf(levels)[None, :], len(y), axis=0)
    exact = np.mean(y * (2 * norm.cdf(y) - 1) + 2 * norm.pdf(y) - 1 / np.sqrt(np.pi))
    assert metrics.crps_quantile(y, q, levels) == pytest.approx(exact, rel=0.02)


def test_pinball_penalises_the_right_side():
    levels = [0.9]
    assert metrics.pinball([10.0], np.array([[0.0]]), levels) == pytest.approx(9.0)
    assert metrics.pinball([0.0], np.array([[10.0]]), levels) == pytest.approx(1.0)


def test_imbalance_cost_signs():
    y, f = np.array([110.0]), np.array([100.0])
    # producer long (+10 MW) while the system price is below the day-ahead price: it loses
    per_mwh, _ = metrics.imbalance_cost(y, f, price=np.array([500.0]), cen=np.array([300.0]), sign=1)
    assert per_mwh > 0
    # a supplier with demand above its purchase pays the higher imbalance price
    per_mwh, _ = metrics.imbalance_cost(y, f, price=np.array([500.0]), cen=np.array([700.0]), sign=-1)
    assert per_mwh > 0
    assert metrics.imbalance_cost(f, f, np.array([500.0]), np.array([0.0]), 1)[0] == 0


def test_battery_profit_perfect_foresight_is_the_bound():
    rng = np.random.default_rng(2)
    days = np.repeat(["a", "b"], 96)
    actual = rng.normal(400, 200, 192)
    perfect = metrics.battery_profit(days, actual, actual)
    noisy = metrics.battery_profit(days, actual + rng.normal(0, 150, 192), actual)
    assert perfect["share_of_perfect"] == pytest.approx(100.0)
    assert noisy["pnl_pln"] <= perfect["pnl_pln"] + 1e-6


def test_bootstrap_skill_interval_contains_the_estimate():
    rng = np.random.default_rng(3)
    days = np.repeat(np.arange(100), 24)
    em, er = rng.normal(0, 1, 2400), rng.normal(0, 1.3, 2400)
    lo, hi = bootstrap_skill(em, er, days)
    est = 1 - np.abs(em).sum() / np.abs(er).sum()
    assert lo < est < hi


def test_capacity_interpolation():
    t = pd.DatetimeIndex(["2025-10-01", "2026-10-01", "2030-01-01"]).tz_localize("UTC")
    cap = installed_capacity("wind", t).to_numpy()
    assert 11000 < cap[0] < 11300 and cap[2] == pytest.approx(11794)


def _grid_submission(track: str, var: str, ds: list[date]) -> pd.DataFrame:
    g = tasks.rows(track, ds)
    df = g[["origin_utc", "target_utc"]].copy()
    df.insert(0, "variable", var)
    df.insert(0, "track", track)
    df["point"] = 1.0
    for j, c in enumerate(submission.QCOLS):
        df[c] = float(j)
    return df


@pytest.fixture
def val_days(monkeypatch):
    ds = [date(2025, 7, 1), date(2025, 7, 2)]
    monkeypatch.setattr(tasks, "split_days", lambda split: ds)
    return ds


def test_submission_accepts_the_grid(val_days):
    df = _grid_submission("day_ahead", "wind", val_days)
    out = submission.check(df, "val")
    assert len(out[("day_ahead", "wind")]) == len(df)


def test_submission_rejects_a_late_origin(val_days):
    """A forecast 'issued' after the official origin would see more data; the grid check refuses it."""
    df = _grid_submission("day_ahead", "wind", val_days)
    df.loc[:95, "origin_utc"] = df.loc[:95, "origin_utc"] + pd.Timedelta(hours=4)
    with pytest.raises(submission.SubmissionError, match="not in the val grid"):
        submission.check(df, "val")


def test_submission_rejects_missing_rows_and_crossing_quantiles(val_days):
    df = _grid_submission("intraday", "pv", val_days)
    with pytest.raises(submission.SubmissionError, match="missing"):
        submission.check(df.iloc[10:], "val")
    bad = df.copy()
    bad["q90"] = -5.0
    with pytest.raises(submission.SubmissionError, match="increasing"):
        submission.check(bad, "val")
    assert submission.check(df.iloc[10:], "val", allow_missing=True)


def test_submission_rejects_price_intraday(val_days):
    df = _grid_submission("day_ahead", "price", val_days)
    df["track"] = "intraday"
    with pytest.raises(submission.SubmissionError, match="unknown task"):
        submission.check(df, "val")


def test_task_grids():
    da = tasks.rows("day_ahead", "test")
    assert da.groupby("delivery_day").size().isin([92, 96, 100]).all()
    assert len(da) == 365 * 96
    idr = tasks.rows("intraday", "test")
    assert idr["lead_minutes"].between(60, 345).all()
    assert idr["origin_utc"].nunique() == 8760
