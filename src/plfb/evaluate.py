"""Score a submission against the actuals and the reference forecasts.

For every task in the file: point and probabilistic scores, breakdowns by lead hour (intraday) or hour of day
(day-ahead) and by month, the value metric, and the MAE skill against a reference on the rows both have, with a 95%
confidence interval from a bootstrap over delivery days. The reference is the PSE plan for wind, PV and demand and
the weekly naive forecast for the price.

Usage:
    python -m plfb evaluate --forecast my_forecasts.parquet --name my-model --out submissions/my-model --meta meta.json
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from plfb import __version__, metrics, submission
from plfb.asof import NS_QH, Store, _ns, load_store
from plfb.config import BENCH_DIR, TZ, installed_capacity, spec

log = logging.getLogger(__name__)

N_BOOT = 1000


def _truth(store: Store, frame: pd.DataFrame, name: str) -> np.ndarray:
    t = _ns(frame["target_utc"])
    pos = (t - store.t0) // NS_QH
    return store.values[name][pos]


def _cen(root: Path, frame: pd.DataFrame) -> np.ndarray:
    from plfb import dataset
    a = dataset.load(root, "actuals").set_index("start_utc")["cen"]
    return a.reindex(pd.DatetimeIndex(frame["target_utc"])).to_numpy(float)


def reference(store: Store, track: str, var: str, frame: pd.DataFrame) -> tuple[str, np.ndarray]:
    t = _ns(frame["target_utc"])
    if var == "price":
        return "naive_week", store.lookup("price", t - 7 * 96 * NS_QH, _ns(frame["origin_utc"]))
    if track == "day_ahead":
        pos = (t - store.t0) // NS_QH
        return "pse_plan", store.values[f"plan_{var}"][pos]
    return "pse_plan", store.lookup(f"plan_{var}", t, _ns(frame["origin_utc"]))


def _normaliser(var: str, frame: pd.DataFrame, y: np.ndarray):
    kind = spec().variable(var)["normaliser"]
    if kind == "installed_capacity":
        return installed_capacity(var, pd.DatetimeIndex(frame["target_utc"])).to_numpy()
    if kind == "mean_actual":
        return float(np.mean(y))
    return float(np.mean(np.abs(y)))


def bootstrap_skill(err_m: np.ndarray, err_r: np.ndarray, days: np.ndarray, seed: int = 0) -> list[float]:
    """95% interval of 1 - MAE_model / MAE_ref, resampling whole delivery days."""
    df = pd.DataFrame({"d": days, "m": np.abs(err_m), "r": np.abs(err_r)})
    g = df.groupby("d").agg(m=("m", "sum"), r=("r", "sum"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (N_BOOT, len(g)))
    m, r = g["m"].to_numpy()[idx].sum(1), g["r"].to_numpy()[idx].sum(1)
    s = 1 - m / r
    return [float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))]


def score_task(root: Path, store: Store, track: str, var: str, frame: pd.DataFrame) -> dict:
    levels = spec().quantiles
    ok = frame["point"].notna().to_numpy().copy()
    f_all = frame["point"].to_numpy(float)
    y_all = _truth(store, frame, var)
    ok &= np.isfinite(y_all)
    fr = frame[ok].reset_index(drop=True)
    y, f = y_all[ok], f_all[ok]
    norm = _normaliser(var, fr, y)
    res: dict = {
        "rows": int(ok.sum()), "rows_expected": int(len(frame)), "complete": bool(ok.all()),
        "mae": metrics.mae(y, f), "rmse": metrics.rmse(y, f), "nmae": metrics.nmae(y, f, norm),
        "bias": float(np.mean(f - y)),
    }
    has_q = submission.QCOLS[0] in fr.columns and fr[submission.QCOLS].notna().all().all()
    if has_q:
        q = fr[submission.QCOLS].to_numpy(float)
        res["pinball"] = metrics.pinball(y, q, levels)
        res["crps"] = metrics.crps_quantile(y, q, levels)
        res["coverage80"] = metrics.coverage(y, q[:, 0], q[:, -1])
        res["width80"] = float(np.mean(q[:, -1] - q[:, 0]))

    # breakdowns
    if track == "intraday":
        key = (fr["lead_minutes"] // 60).astype(int)
        res["by_lead_hour"] = {int(k): metrics.mae(y[key == k], f[key == k]) for k in sorted(key.unique())}
    else:
        hour = pd.DatetimeIndex(fr["target_utc"]).tz_convert(TZ).hour
        res["by_hour"] = {int(h): metrics.mae(y[hour == h], f[hour == h]) for h in range(24)}
    month = pd.DatetimeIndex(fr["target_utc"]).tz_convert(TZ).strftime("%Y-%m")
    res["by_month"] = {m: metrics.mae(y[month == m], f[month == m]) for m in sorted(set(month))}

    # skill against the reference, on the rows both have
    ref_name, r_all = reference(store, track, var, fr)
    both = np.isfinite(r_all)
    days = fr["delivery_day"].astype(str).to_numpy()
    if both.any():
        mae_m, mae_r = metrics.mae(y[both], f[both]), metrics.mae(y[both], r_all[both])
        res["reference"] = {
            "name": ref_name, "rows": int(both.sum()), "mae_model": mae_m, "mae_reference": mae_r,
            "skill_mae": 1 - mae_m / mae_r,
            "skill_ci95": bootstrap_skill(y[both] - f[both], y[both] - r_all[both], days[both]),
        }

    # value
    if track == "day_ahead" and var in ("wind", "pv", "demand"):
        price = _truth(store, fr, "price")
        cen = _cen(root, fr)
        sign = spec().variable(var)["sign"]
        good = np.isfinite(price) & np.isfinite(cen)
        per_mwh, total = metrics.imbalance_cost(y[good], f[good], price[good], cen[good], sign)
        res["imbalance_cost_pln_mwh"] = per_mwh
        res["imbalance_cost_mln_pln"] = total
    if track == "day_ahead" and var == "price":
        res["battery"] = metrics.battery_profit(days, f, y)
    return res


def checksums_digest() -> str | None:
    """SHA-256 of the benchmark checksums in canonical JSON form (independent of line endings)."""
    path = BENCH_DIR / "checksums.json"
    if not path.exists():
        return None
    canon = json.dumps(json.loads(path.read_text(encoding="utf-8")), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()


def score(root: Path, df: pd.DataFrame, name: str, split: str, allow_missing: bool = False) -> dict:
    by_task = submission.check(df, split, allow_missing=allow_missing)
    store = load_store(root)
    out = {
        "name": name, "split": split, "scored_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "benchmark_version": spec().raw["benchmark"]["version"], "plfb_version": __version__,
        "data_checksums_sha256": checksums_digest(),
        "tasks": {},
    }
    for (track, var), frame in sorted(by_task.items()):
        out["tasks"][f"{track}/{var}"] = score_task(root, store, track, var, frame)
        t = out["tasks"][f"{track}/{var}"]
        log.info("%s %s/%s: MAE %.1f nMAE %.2f%%%s", name, track, var, t["mae"], t["nmae"],
                 f" skill vs {t['reference']['name']} {t['reference']['skill_mae']:+.1%}" if "reference" in t else "")
    return out


def cli(root: Path, paths: list[Path], name: str, split: str, out: Path | None, meta: Path | None) -> None:
    df = submission.read(paths)
    allow = False
    if meta is not None:
        allow = bool(json.loads(meta.read_text(encoding="utf-8")).get("reference", False))
    scores = score(root, df, name, split, allow_missing=allow)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        text = json.dumps(scores, indent=1) + "\n"
        (out / f"scores_{split}.json").write_text(text, encoding="utf-8", newline="\n")
        if meta is not None and meta.resolve() != (out / "meta.json").resolve():
            shutil.copyfile(meta, out / "meta.json")
        log.info("wrote %s", out / f"scores_{split}.json")
