"""Scores of point and quantile forecasts and two value metrics.

Point: MAE, RMSE and nMAE (wind and PV: |error| divided by the installed capacity at that time; demand: by the mean
actual demand; price: by the mean absolute price).
Probabilistic: pinball loss averaged over the nine deciles and CRPS approximated by twice that average (the decile
discretisation of CRPS = 2 * integral of the pinball loss over the quantile levels); coverage and width of the
10-90% interval.
Value:
- imbalance cost (day-ahead wind, PV, demand): the forecast is traded day-ahead at the SDAC price p and the deviation
  is settled at the single imbalance price CEN c; the cost against a perfect forecast is sign * (y - f) * (p - c) per
  interval (sign +1 for a producer, -1 for a supplier), reported in PLN per MWh of actual energy. This is a
  simplified settlement (no fees, no portfolio netting, a national volume taken as one balancing position) and the
  current settlement rules should be verified;
- battery profit (day-ahead price): the 1 MW / 2 MWh MILP schedule of pl-bess-paper-trading is optimised on the
  forecast prices and settled at the actual ones, compared with perfect foresight.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from plfb.config import spec

DT_H = 0.25


def mae(y, f) -> float:
    return float(np.mean(np.abs(y - f)))


def rmse(y, f) -> float:
    return float(np.sqrt(np.mean((y - f) ** 2)))


def nmae(y, f, norm) -> float:
    """Mean of |error| / norm (norm: scalar or one value per row), in percent."""
    return float(np.mean(np.abs(y - f) / norm) * 100)


def pinball(y, q, levels) -> float:
    """Mean pinball loss over rows and quantile levels; q has one column per level."""
    y = np.asarray(y, float)[:, None]
    a = np.asarray(levels, float)[None, :]
    d = y - q
    return float(np.mean(np.maximum(a * d, (a - 1) * d)))


def crps_quantile(y, q, levels) -> float:
    return 2.0 * pinball(y, q, levels)


def coverage(y, lo, hi) -> float:
    return float(np.mean((y >= lo) & (y <= hi)) * 100)


def imbalance_cost(y, f, price, cen, sign: int) -> tuple[float, float]:
    """(PLN per MWh of actual energy, total million PLN)."""
    cost = sign * (y - f) * (price - cen) * DT_H
    energy = np.sum(np.abs(y)) * DT_H
    return float(np.sum(cost) / energy), float(np.sum(cost) / 1e6)


@lru_cache(maxsize=4096)
def _perfect(prices: tuple[float, ...]) -> float:
    from bess_paper.optimizer import optimize, settle
    return settle(optimize(np.array(prices), DT_H), np.array(prices), DT_H)["pnl_pln"]


def battery_profit(days, forecast, actual) -> dict:
    """Battery P&L over delivery days; ``days`` labels each row with its delivery day."""
    from bess_paper.config import BATTERY
    from bess_paper.optimizer import optimize, settle
    days = np.asarray(days)
    total = perfect = 0.0
    n_days = 0
    for d in np.unique(days):
        m = days == d
        f, a = np.asarray(forecast)[m], np.asarray(actual)[m]
        total += settle(optimize(f, DT_H), a, DT_H)["pnl_pln"]
        perfect += _perfect(tuple(np.round(a, 4)))
        n_days += 1
    per_mw_year = total / BATTERY.power_mw / n_days * 365
    return {"pnl_pln": total, "perfect_pln": perfect, "share_of_perfect": total / perfect * 100,
            "pln_per_mw_year": per_mw_year, "days": n_days}


def quantile_levels() -> list[float]:
    return spec().quantiles
