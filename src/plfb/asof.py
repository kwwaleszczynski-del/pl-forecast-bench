"""The only door from the data to the models: every value comes with the time it became public.

``Store`` holds the actuals, the PSE plan and the NWP runs together with their availability times and answers
lookups *as of* an origin: a value whose availability is later than the origin comes back as NaN, whatever the caller
asked for. Feature code therefore cannot read the future by mistake; ``tests/test_leakage.py`` checks it end to end
by replacing every not-yet-public value with noise and requiring identical features and forecasts.

Availability (``benchmark/spec.toml``):
- actual wind, PV, demand: interval end + ``observation_lag_minutes``;
- day-ahead price and PSE plan: the publication time of their business day;
- an NWP run: initialisation + ``nwp_delay_hours``; the forecast at origin o uses the newest 00 UTC run that is
  available at o.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from plfb import solar
from plfb.config import QH, points, spec

OBS = ("wind", "pv", "demand", "price")
PLAN = ("wind", "pv", "demand")
NS_QH = QH.value


def _ns(x) -> np.ndarray:
    """int64 nanoseconds since the epoch (UTC) of timestamps of any unit."""
    idx = pd.DatetimeIndex(pd.to_datetime(x, utc=True)).as_unit("ns")
    return idx.asi8


def power_curve(v: np.ndarray) -> np.ndarray:
    """Generic normalised wind turbine power curve: cubic from 3 to 12 m/s, rated to 25 m/s."""
    v = np.asarray(v, float)
    p = (v ** 3 - 27.0) / (12.0 ** 3 - 27.0)
    return np.where(v < 3, 0.0, np.where(v < 12, p, np.where(v < 25, 1.0, 0.0)))


def nwp_quarter_hours(nwp: pd.DataFrame) -> pd.DataFrame:
    """Hourly NWP per run and point -> 15-minute features per run (rows) with per-point and weighted columns.

    Instantaneous variables are interpolated linearly to the middle of each quarter-hour; irradiance (mean of the
    preceding hour) is spread within its hour with the shape of the clear-sky irradiance, keeping the hourly mean.
    """
    pts = points()
    out = []
    for init, run in nwp.groupby("init_utc", sort=True):
        avail = run["available_utc"].iloc[0]
        valid = pd.DatetimeIndex(run["valid_utc"].unique()).sort_values()
        q = pd.date_range(valid[0], valid[-1], freq=QH, inclusive="left")
        mid = _ns(q) + NS_QH // 2
        cols: dict[str, np.ndarray] = {}
        cs_w = np.zeros(len(q))
        ghi_w = np.zeros(len(q))
        for pid, g in run.groupby("point"):
            g = g.set_index("valid_utc").sort_index()
            x = _ns(g.index)
            def lin(c, g=g, x=x, mid=mid):
                return np.interp(mid, x, g[c].to_numpy(float), left=np.nan, right=np.nan)
            ws100, ws10, t2m = lin("wind_speed_100m"), lin("wind_speed_10m"), lin("temperature_2m")
            cloud = lin("cloud_cover")
            lat, lon = pts.loc[pid, "lat"], pts.loc[pid, "lon"]
            cosz = solar.cos_zenith(pd.DatetimeIndex(mid, tz="UTC"), lat, lon)
            cs = solar.clear_sky_ghi(cosz)
            # hour label of each quarter: radiation at hh:00 is the mean of (hh-1, hh]
            label = (q + QH).ceil("h")
            ghi_h = g["shortwave_radiation"].reindex(label).to_numpy(float)
            dif_h = g["diffuse_radiation"].reindex(label).to_numpy(float)
            cs_h = pd.Series(cs).groupby(_ns(label)).transform("mean").to_numpy()
            shape = np.where(cs_h > 1.0, cs / np.where(cs_h > 1.0, cs_h, 1.0), 1.0)
            ghi = np.clip(ghi_h * shape, 0, None)
            dif = np.clip(dif_h * shape, 0, None)
            w = pts.loc[pid]
            cols[f"ws100_{pid}"] = ws100
            cols[f"ghi_{pid}"] = ghi
            cols[f"t2m_{pid}"] = t2m
            cols[f"cloud_{pid}"] = cloud
            # aggregates
            cols["wind_idx"] = cols.get("wind_idx", 0) + w.w_wind * power_curve(ws100)
            cols["ws100_w"] = cols.get("ws100_w", 0) + w.w_wind * ws100
            cols["ws10_w"] = cols.get("ws10_w", 0) + w.w_wind * ws10
            tcell = t2m + 0.03 * ghi
            cols["pv_idx"] = cols.get("pv_idx", 0) + w.w_pv * ghi / 1000 * (1 - 0.004 * (tcell - 25))
            cols["dif_w"] = cols.get("dif_w", 0) + w.w_pv * dif
            cols["cloud_pv"] = cols.get("cloud_pv", 0) + w.w_pv * cloud
            cols["t2m_pop"] = cols.get("t2m_pop", 0) + w.w_pop * t2m
            ghi_w += w.w_pv * ghi
            cs_w += w.w_pv * cs
        cols["ghi_w"] = ghi_w
        cols["cs_w"] = cs_w
        cols["kt_w"] = np.where(cs_w > 20, ghi_w / np.where(cs_w > 20, cs_w, 1), np.nan)
        df = pd.DataFrame(cols, index=q)
        df.index.name = "valid_utc"
        df.insert(0, "available_utc", avail)
        df.insert(0, "init_utc", init)
        out.append(df.reset_index())
    res = pd.concat(out, ignore_index=True)
    float_cols = res.columns[res.dtypes == "float64"]
    res[float_cols] = res[float_cols].astype("float32")
    return res


@dataclass
class Store:
    t0: int                           # ns of the first quarter-hour
    values: dict[str, np.ndarray]     # quarter-hour series on one regular grid
    avail: dict[str, np.ndarray]      # their availability (ns)
    nwp_init: np.ndarray              # ns of each run's initialisation, sorted
    nwp_avail: np.ndarray             # ns of each run's availability
    nwp_valid0: np.ndarray            # ns of each run's first quarter-hour
    nwp_cols: list[str]
    nwp_values: np.ndarray            # (runs, quarter-hours per run, columns)

    @classmethod
    def from_tables(cls, actuals: pd.DataFrame, plan: pd.DataFrame, nwp15: pd.DataFrame) -> Store:
        a = actuals.sort_values("start_utc")
        start = _ns(a["start_utc"])
        if not (np.diff(start) == NS_QH).all():
            raise ValueError("actuals are not on a regular 15-minute grid")
        lag = spec().obs_lag.value
        values, avail = {}, {}
        for c in ("wind", "pv", "demand"):
            values[c] = a[c].to_numpy(float)
            avail[c] = start + NS_QH + lag
        values["price"] = a["price"].to_numpy(float)
        avail["price"] = _ns(a["price_available_utc"])
        p = plan.set_index(_ns(plan["start_utc"])).reindex(start)
        for c in PLAN:
            values[f"plan_{c}"] = p[c].to_numpy(float)
            avail[f"plan_{c}"] = np.where(p["available_utc"].isna(), np.iinfo(np.int64).max,
                                          _ns(p["available_utc"].fillna(pd.Timestamp("2262-01-01", tz="UTC"))))
        runs = nwp15.sort_values(["init_utc", "valid_utc"])
        cols = [c for c in runs.columns if c not in ("init_utc", "available_utc", "valid_utc")]
        init_ns = _ns(runs["init_utc"])
        inits, first_idx, sizes = np.unique(init_ns, return_index=True, return_counts=True)
        # runs differ in length (a few were fetched with a longer horizon to bridge gaps of the archive)
        arr = np.full((len(inits), int(sizes.max()), len(cols)), np.nan, dtype=np.float32)
        vals = runs[cols].to_numpy(np.float32)
        step = np.arange(len(runs)) - np.repeat(first_idx, sizes)
        arr[np.repeat(np.arange(len(inits)), sizes), step] = vals
        valid = _ns(runs["valid_utc"])
        first = valid[first_idx]
        av = _ns(runs["available_utc"])[first_idx]
        return cls(int(start[0]), values, avail, inits, av, first, cols, arr)

    # --- guarded lookups --------------------------------------------------------------------------------------------

    def lookup(self, name: str, times, origins) -> np.ndarray:
        """Value of series ``name`` at quarter-hours ``times`` as known at ``origins`` (NaN if not yet public)."""
        t, o = _ns(times), _ns(origins)
        pos = (t - self.t0) // NS_QH
        ok = (pos >= 0) & (pos < len(self.values[name])) & ((t - self.t0) % NS_QH == 0)
        posc = np.where(ok, pos, 0)
        out = np.where(ok, self.values[name][posc], np.nan)
        return np.where(ok & (self.avail[name][posc] <= o), out, np.nan)

    def last_time(self, name: str, origins) -> np.ndarray:
        """Start (ns) of the latest quarter-hour of an observation series that is public at each origin."""
        if name not in ("wind", "pv", "demand"):
            raise ValueError("last_time is defined for the real-time series only")
        o = _ns(origins)
        last_end = o - spec().obs_lag.value
        return (last_end // NS_QH) * NS_QH - NS_QH

    def run_for(self, origins) -> np.ndarray:
        """Index of the newest NWP run available at each origin (-1 if none)."""
        o = _ns(origins)
        k = np.searchsorted(self.nwp_avail, o, side="right") - 1
        return k

    def nwp(self, origins, times, cols: list[str] | None = None) -> pd.DataFrame:
        """NWP features at ``times`` from the newest run available at each origin."""
        cols = cols or self.nwp_cols
        ci = [self.nwp_cols.index(c) for c in cols]
        k = self.run_for(origins)
        o, t = _ns(origins), _ns(times)
        kc = np.where(k >= 0, k, 0)
        step = (t - self.nwp_valid0[kc]) // NS_QH
        ok = (k >= 0) & (step >= 0) & (step < self.nwp_values.shape[1]) & (self.nwp_avail[kc] <= o)
        out = self.nwp_values[kc[:, None], np.where(ok, step, 0)[:, None], np.array(ci)[None, :]].astype(float)
        out[~ok] = np.nan
        return pd.DataFrame(out, columns=cols)

    def nwp_at(self, init_ns: np.ndarray, times, cols: list[str]) -> np.ndarray:
        """NWP of explicitly chosen runs (by initialisation) at ``times``; used only for past quarter-hours."""
        ci = [self.nwp_cols.index(c) for c in cols]
        k = np.searchsorted(self.nwp_init, init_ns)
        kc = np.clip(k, 0, len(self.nwp_init) - 1)
        t = _ns(times)
        step = (t - self.nwp_valid0[kc]) // NS_QH
        ok = (self.nwp_init[kc] == init_ns) & (step >= 0) & (step < self.nwp_values.shape[1])
        out = self.nwp_values[kc[:, None], np.where(ok, step, 0)[:, None], np.array(ci)[None, :]].astype(float)
        out[~ok] = np.nan
        return out

    # --- for the leakage tests ----------------------------------------------------------------------------------------

    def poisoned(self, origin, seed: int = 0) -> Store:
        """A copy in which everything not yet public at ``origin`` is replaced by noise."""
        rng = np.random.default_rng(seed)
        o = int(_ns([origin])[0])
        values = {}
        for k, v in self.values.items():
            v = v.copy()
            late = self.avail[k] > o
            v[late] = rng.normal(5000, 3000, late.sum())
            values[k] = v
        arr = self.nwp_values.copy()
        late = self.nwp_avail > o
        arr[late] = rng.normal(10, 50, arr[late].shape).astype(np.float32)
        return replace(self, values=values, nwp_values=arr)


def load_store(root) -> Store:
    from plfb import dataset
    actuals = dataset.load(root, "actuals")
    plan = dataset.load(root, "plan")
    return Store.from_tables(actuals, plan, dataset.load(root, "nwp15"))
