"""Reference models. Each one is refitted at the start of every quarter on the labels public at that moment.

- ``persistence``: the latest public value, held flat;
- ``naive_day``: the same quarter-hour on the newest day that is fully public (D-2 for day-ahead wind, PV, demand;
  D-1 for the price; 24 h earlier intraday);
- ``naive_week``: the same quarter-hour a week earlier;
- ``ridge``: linear regression (ridge) on the NWP, calendar and lag features;
- ``gbm``: gradient boosting (LightGBM) for the mean and for each quantile;
- ``pse_plan``: the PSE day-ahead coordinated plan (PKD). Intraday it is as of the origin; day-ahead it is published
  after the SDAC gate closure (13:30-20:30 Polish time on D-1), so it has more information than the benchmark allows
  and is shown as a reference, not ranked.

Naive models and ridge get quantiles from their empirical errors of the 90 days before the refit, by hour of day
(day-ahead) or by lead hour (intraday).
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NAIVE = ("persistence", "naive_day", "naive_week")
GBM_PARAMS = dict(num_leaves=31, min_child_samples=50, reg_lambda=1.0, subsample=1.0, colsample_bytree=1.0,
                  random_state=0, deterministic=True, force_row_wise=True, verbose=-1)
GBM_ITER = {"mean": (300, 0.08), "quantile": (200, 0.1)}       # (trees, learning rate)
N_JOBS = 2
GBM_FROM_RIDGE = False       # boosting from the ridge prediction lost on 6 of 7 validation tasks (README)


# a lag that is not public yet (e.g. the 25th hour of the autumn DST day falls on D itself) falls back to the next one
NAIVE_COLUMNS = {
    "price": {"persistence": ["p_last"], "naive_day": ["p_lag1", "p_lag2"], "naive_week": ["p_lag7"]},
    "other": {"persistence": ["last"], "naive_day": ["lag_day", "lag_week"], "naive_week": ["lag_week"]},
}


def naive_point(feats: pd.DataFrame, model: str, var: str) -> np.ndarray:
    """Persistence (price: the last quarter-hour of D-1, held flat over D) and the seasonal naive forecasts."""
    cols = NAIVE_COLUMNS["price" if var == "price" else "other"][model]
    out = feats[cols[0]].to_numpy().copy()
    for c in cols[1:]:
        out = np.where(np.isnan(out), feats[c].to_numpy(), out)
    return out


def group_key(rows: pd.DataFrame, track: str) -> np.ndarray:
    if track == "intraday":
        return (rows["lead_minutes"].to_numpy() // 60).astype(int)
    return pd.DatetimeIndex(rows["target_utc"]).tz_convert("Europe/Warsaw").hour.to_numpy()


def residual_quantiles(resid: np.ndarray, groups: np.ndarray, levels: list[float]) -> dict[int, np.ndarray]:
    out = {}
    ok = np.isfinite(resid)
    for g in np.unique(groups):
        r = resid[ok & (groups == g)]
        out[int(g)] = np.quantile(r, levels) if len(r) >= 20 else np.quantile(resid[ok], levels)
    return out


def apply_quantiles(point: np.ndarray, groups: np.ndarray, table: dict[int, np.ndarray], n: int) -> np.ndarray:
    q = np.full((len(point), n), np.nan)
    for g, vals in table.items():
        m = groups == g
        q[m] = point[m, None] + vals[None, :]
    return q


def fit_ridge(X: pd.DataFrame, y: np.ndarray):
    model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                          Ridge(alpha=10.0))
    model.fit(X, y)
    return model


def ridge_design(feats: pd.DataFrame) -> pd.DataFrame:
    X = feats.copy()
    hour = (X["qod"] // 4).astype(int)
    for h in range(24):
        X[f"h{h}"] = (hour == h).astype(float)
    for d in range(7):
        X[f"d{d}"] = (X["dow"] == d).astype(float)
    return X.drop(columns=["qod", "dow"])


def fit_gbm(X: pd.DataFrame, y: np.ndarray, quantile: float | None = None,
            init_score: np.ndarray | None = None) -> lgb.LGBMRegressor:
    """LightGBM for the mean or one quantile; with ``init_score`` it boosts from that base (add it back when
    predicting: ``predict`` returns only the trees' part)."""
    if quantile is None:
        it, lr = GBM_ITER["mean"]
        m = lgb.LGBMRegressor(objective="regression", n_estimators=it, learning_rate=lr, n_jobs=N_JOBS, **GBM_PARAMS)
    else:
        it, lr = GBM_ITER["quantile"]
        m = lgb.LGBMRegressor(objective="quantile", alpha=quantile, n_estimators=it, learning_rate=lr, n_jobs=N_JOBS,
                              **GBM_PARAMS)
    m.fit(X, y, init_score=init_score)
    return m


def postprocess(var: str, point: np.ndarray, q: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
    if var in ("wind", "pv"):
        point = np.clip(point, 0, None)
        if q is not None:
            q = np.clip(q, 0, None)
    if q is not None:
        q = np.sort(q, axis=1)
    return point, q
