"""As-of features for the reference models. Every value is read through ``Store`` lookups guarded by the origin."""
from __future__ import annotations

import numpy as np
import pandas as pd
from bess_paper.calendar_pl import is_holiday

from plfb import solar
from plfb.asof import NS_QH, Store, _ns
from plfb.config import TZ, points

DAY = 96
NS_DAY = DAY * NS_QH
POINT_IDS = list(points().index)
NWP_AGG = ["wind_idx", "ws100_w", "ws10_w", "pv_idx", "ghi_w", "cs_w", "kt_w", "dif_w", "cloud_pv", "t2m_pop"]
NWP_POINT = {
    "wind": [f"ws100_{p}" for p in POINT_IDS],
    "pv": [f"ghi_{p}" for p in POINT_IDS],
    "demand": [f"t2m_{p}" for p in POINT_IDS],
    "price": [],
}
NWP_MAIN = {"wind": "wind_idx", "pv": "pv_idx", "demand": "t2m_pop"}


def calendar(rows: pd.DataFrame) -> pd.DataFrame:
    t = pd.DatetimeIndex(rows["target_utc"])
    loc = t.tz_convert(TZ)
    day = loc.normalize()
    hol = np.array([is_holiday(d) for d in loc.date])
    doy = loc.day_of_year.to_numpy()
    mid = t + pd.Timedelta(minutes=7.5)
    return pd.DataFrame({
        "qod": (loc.hour * 4 + loc.minute // 15).to_numpy(),
        "dow": loc.day_of_week.to_numpy(),
        "holiday": hol.astype(float),
        "offday": (hol | (loc.day_of_week.to_numpy() >= 5)).astype(float),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "cosz": solar.cos_zenith(mid, 52.0, 19.0),
        "day_ns": day.tz_convert("UTC").as_unit("ns").asi8,
    }, index=rows.index)


def _lookup_array(store: Store, name: str, arr: np.ndarray, times: np.ndarray, origins: np.ndarray) -> np.ndarray:
    """Like ``Store.lookup`` for an array derived from a stored series (same grid, same availability).

    Valid only for backward-looking derivations (the value at a quarter-hour uses that quarter-hour and earlier ones),
    whose inputs are all public when the quarter-hour itself is.
    """
    pos = (times - store.t0) // NS_QH
    ok = (pos >= 0) & (pos < len(arr))
    posc = np.where(ok, pos, 0)
    return np.where(ok & (store.avail[name][posc] <= origins), arr[posc], np.nan)


def _rolling(store: Store, name: str, window: int, how: str = "mean") -> np.ndarray:
    s = pd.Series(store.values[name])
    r = s.rolling(window, min_periods=window // 2)
    return getattr(r, how)().to_numpy()


def _dayahead_nwp_index(store: Store, var: str) -> np.ndarray:
    """The variable's NWP index at every quarter-hour from the 00 UTC run of the previous UTC day."""
    t = store.t0 + NS_QH * np.arange(len(store.values["wind"]))
    init = (t // NS_DAY) * NS_DAY - NS_DAY
    return store.nwp_at(init, t, [NWP_MAIN[var]])[:, 0]


def recent_ratio(store: Store, var: str, times: np.ndarray, origins: np.ndarray, days: int = 14) -> np.ndarray:
    """Actual / NWP index over the trailing ``days`` up to ``times`` (a calibration of the fleet size)."""
    idx = _dayahead_nwp_index(store, var)
    act = store.values[var]
    both = np.isfinite(idx) & np.isfinite(act)
    a = pd.Series(np.where(both, act, 0.0)).rolling(days * DAY, min_periods=DAY).sum().to_numpy()
    b = pd.Series(np.where(both, idx, 0.0)).rolling(days * DAY, min_periods=DAY).sum().to_numpy()
    ratio = np.where(b > 1e-6, a / np.where(b > 1e-6, b, 1.0), np.nan)
    return _lookup_array(store, var, ratio, times, origins)


def _obs_block(store: Store, var: str, rows: pd.DataFrame) -> dict[str, np.ndarray]:
    o = _ns(rows["origin_utc"])
    t = _ns(rows["target_utc"])
    last = store.last_time(var, o)
    lag = store.avail[var][0] - (store.t0 + NS_QH)          # observation lag in ns
    k = np.ceil((t + NS_QH + lag - o) / NS_DAY).astype(np.int64)   # newest whole-day lag that is public
    out = {
        "last": store.lookup(var, last, o),
        "last_1h": _lookup_array(store, var, _rolling(store, var, 4), last, o),
        "last_trend": store.lookup(var, last, o) - store.lookup(var, last - 4 * NS_QH, o),
        "lag_day": store.lookup(var, t - k * NS_DAY, o),
        "lag_week": store.lookup(var, t - 7 * NS_DAY, o),
        "mean_day": _lookup_array(store, var, _rolling(store, var, DAY), last, o),
        "mean_week": _lookup_array(store, var, _rolling(store, var, 7 * DAY), last, o),
    }
    if var in ("wind", "pv"):
        out["ratio14"] = recent_ratio(store, var, last, o)
    return out


def _nwp_block(store: Store, var: str, rows: pd.DataFrame) -> pd.DataFrame:
    cols = NWP_AGG + NWP_POINT[var]
    df = store.nwp(rows["origin_utc"], rows["target_utc"], cols)
    df.index = rows.index
    if var in ("wind", "pv"):
        df["idx_scaled"] = df[NWP_MAIN[var]]          # multiplied by ratio14 later
    return df


def build(store: Store, rows: pd.DataFrame, track: str, var: str) -> pd.DataFrame:
    """Feature matrix for forecast rows (``origin_utc``, ``target_utc``, ``lead_minutes``) of one task."""
    cal = calendar(rows)
    o = _ns(rows["origin_utc"])
    t = _ns(rows["target_utc"])
    feats = [cal.drop(columns="day_ns")]
    if var == "price":
        # all prices of D-1 are public on D-1 morning (published on D-2)
        prev_end = cal["day_ns"].to_numpy() - NS_QH
        p = {
            "p_last": store.lookup("price", prev_end, o),
            "p_lag1": store.lookup("price", t - NS_DAY, o),
            "p_lag2": store.lookup("price", t - 2 * NS_DAY, o),
            "p_lag7": store.lookup("price", t - 7 * NS_DAY, o),
            "p_mean1": _lookup_array(store, "price", _rolling(store, "price", DAY), prev_end, o),
            "p_max1": _lookup_array(store, "price", _rolling(store, "price", DAY, "max"), prev_end, o),
            "p_min1": _lookup_array(store, "price", _rolling(store, "price", DAY, "min"), prev_end, o),
            "p_mean7": _lookup_array(store, "price", _rolling(store, "price", 7 * DAY), prev_end, o),
            "demand_lag_week": store.lookup("demand", t - 7 * NS_DAY, o),
        }
        for v in ("wind", "pv"):
            last = store.last_time(v, o)
            p[f"ratio14_{v}"] = recent_ratio(store, v, last, o)
        feats.append(pd.DataFrame(p, index=rows.index))
        nw = store.nwp(rows["origin_utc"], rows["target_utc"], NWP_AGG)
        nw.index = rows.index
        nw["wind_mw_est"] = nw["wind_idx"] * feats[-1]["ratio14_wind"]
        nw["pv_mw_est"] = nw["pv_idx"] * feats[-1]["ratio14_pv"]
        nw["resid_est"] = feats[-1]["demand_lag_week"] - nw["wind_mw_est"] - nw["pv_mw_est"]
        day = cal["day_ns"]
        for c in ("wind_mw_est", "pv_mw_est", "resid_est", "t2m_pop"):
            nw[f"{c}_daymean"] = nw[c].groupby([o, day.to_numpy()]).transform("mean")
        nw["resid_rank"] = nw["resid_est"].groupby([o, day.to_numpy()]).rank(pct=True)
        feats.append(nw)
        return pd.concat(feats, axis=1)

    ob = _obs_block(store, var, rows)
    feats.append(pd.DataFrame(ob, index=rows.index))
    nw = _nwp_block(store, var, rows)
    if var in ("wind", "pv"):
        nw["idx_scaled"] = nw["idx_scaled"] * ob["ratio14"]
    feats.append(nw)
    if track == "intraday":
        last = store.last_time(var, o)
        main = NWP_MAIN[var]
        nwl = store.nwp(rows["origin_utc"], pd.DatetimeIndex(last, tz="UTC"), [main])[main].to_numpy()
        plan_t = store.lookup(f"plan_{var}", t, o)
        plan_l = store.lookup(f"plan_{var}", last, o)
        extra = {
            "lead": rows["lead_minutes"].to_numpy(float),
            "nwp_last": nwl,
            "nwp_delta": nw[main].to_numpy() - nwl,
            "plan": plan_t,
            "plan_last": plan_l,
            "plan_err_last": ob["last"] - plan_l,
            "plan_delta": plan_t - plan_l,
        }
        if var in ("wind", "pv"):
            extra["nwp_err_last"] = ob["last"] - nwl * ob["ratio14"]
        feats.append(pd.DataFrame(extra, index=rows.index))
    else:
        nw_day = nw[NWP_MAIN[var]].groupby([o, cal["day_ns"].to_numpy()]).transform("mean")
        feats.append(pd.DataFrame({"nwp_daymean": nw_day.to_numpy()}, index=rows.index))
    return pd.concat(feats, axis=1)


def target(store: Store, rows: pd.DataFrame, var: str) -> np.ndarray:
    """Actual values at the targets (no origin guard: these are the labels)."""
    t = _ns(rows["target_utc"])
    pos = (t - store.t0) // NS_QH
    ok = (pos >= 0) & (pos < len(store.values[var]))
    return np.where(ok, store.values[var][np.where(ok, pos, 0)], np.nan)


def target_available(store: Store, rows: pd.DataFrame, var: str) -> np.ndarray:
    """When each label becomes public (ns); a model refitted at time T may only train on labels public by T."""
    t = _ns(rows["target_utc"])
    pos = np.clip((t - store.t0) // NS_QH, 0, len(store.values[var]) - 1)
    return store.avail[var][pos]
