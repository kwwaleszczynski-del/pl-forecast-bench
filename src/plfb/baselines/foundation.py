"""Zero-shot time series foundation models (no fine-tuning), with the same as-of contexts as the other models.

Models (all Apache-2.0; licences checked on the model cards on 2026-10-28):
- ``chronos_bolt``: amazon/chronos-bolt-small, univariate;
- ``chronos2``: amazon/chronos-2, univariate and with NWP covariates (``chronos2_nwp``);
- ``timesfm``: google/timesfm-2.5-200m-pytorch, univariate;
- ``toto``: Datadog/Toto-Open-Base-1.0, univariate, quantiles from 100 samples.
Not used: Moirai 1.x/2.0 (CC BY-NC 4.0, non-commercial) and TiRex (NXAI Community License, not an open licence).

Context: the actual series up to the newest value public at the origin (read through the guarded ``Store``).
Intraday: 15-minute context of 2048 quarter-hours, forecast to the last target. Day-ahead: hourly means (context
1024 hours), because the horizon from the last public value to the end of D is 38-40 hours, longer than some models
forecast natively at 15 minutes; the hourly forecast is interpolated to quarter-hours at the interval mid-points.
The price context ends with the last quarter-hour of D-1 (published on D-2).

Usage:
    python -m plfb foundation --models chronos_bolt chronos2 --device cuda
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from plfb import submission, tasks
from plfb.asof import NS_QH, Store, _ns, load_store
from plfb.config import spec

log = logging.getLogger(__name__)

MODELS = ("chronos_bolt", "chronos2", "chronos2_nwp", "timesfm", "toto")
CTX_ID = 2048
TOTO_SAMPLES = 100
CTX_DA_H = 1024
NS_H = 4 * NS_QH
COVARIATES = {"wind": ["wind_idx"], "pv": ["pv_idx", "cs_w"], "demand": ["t2m_pop", "pv_idx"],
              "price": ["wind_idx", "pv_idx", "t2m_pop"]}


# --- contexts ---------------------------------------------------------------------------------------------------------

def _last_public(store: Store, var: str, origin_ns: np.ndarray) -> np.ndarray:
    if var == "price":
        # prices are published a day ahead: the newest public quarter-hour is the end of the newest published day,
        # which lies after the origin; scan back from two days after the origin
        av = store.avail["price"]
        out = np.empty_like(origin_ns)
        for i, o in enumerate(origin_ns):
            pos = min((o + 2 * 96 * NS_QH - store.t0) // NS_QH, len(av) - 1)
            while pos >= 0 and av[pos] > o:
                pos -= 1
            out[i] = store.t0 + pos * NS_QH
        return out
    return store.last_time(var, origin_ns)


def contexts(store: Store, var: str, origins: pd.DatetimeIndex, length: int,
             step_ns: int) -> tuple[np.ndarray, np.ndarray]:
    """(len(origins), length) context matrix ending at the newest public value, and that time per origin.

    With ``step_ns`` of one hour the context holds hourly means of whole public hours.
    """
    o = _ns(origins)
    last = _last_public(store, var, o)
    if step_ns == NS_QH:
        times = last[:, None] - NS_QH * np.arange(length - 1, -1, -1)[None, :]
        ctx = store.lookup(var, times.ravel(), np.repeat(o, length)).reshape(len(o), length)
        return ctx, last
    # hourly: the last whole hour that ends at or before the end of the newest public quarter-hour
    end = ((last + NS_QH) // NS_H) * NS_H
    qtimes = end[:, None] - NS_QH * np.arange(4 * length, 0, -1)[None, :]
    q = store.lookup(var, qtimes.ravel(), np.repeat(o, 4 * length)).reshape(len(o), length, 4)
    with np.errstate(all="ignore"):
        ctx = np.nanmean(q, axis=2)
    return ctx, end - NS_H


def _fill(ctx: np.ndarray) -> np.ndarray:
    """Forward-fill gaps (models take a few NaNs badly); leading NaNs become the first valid value."""
    df = pd.DataFrame(ctx.T).ffill().bfill()
    return df.to_numpy().T


def _covariates(store: Store, var: str, origins: pd.DatetimeIndex, times_ns: np.ndarray) -> np.ndarray:
    """NWP covariates at ``times_ns`` (origins x steps): past steps from the day-ahead run of their own day, future
    steps from the newest run public at the origin."""
    o = _ns(origins)
    n, k = times_ns.shape
    cols = COVARIATES[var]
    flat_t = times_ns.ravel()
    flat_o = np.repeat(o, k)
    fut = store.nwp(pd.DatetimeIndex(flat_o, tz="UTC"), pd.DatetimeIndex(flat_t, tz="UTC"), cols).to_numpy()
    init = (flat_t // (96 * NS_QH)) * 96 * NS_QH - 96 * NS_QH
    past = store.nwp_at(init, flat_t, cols)
    past_ok = (init + spec().nwp_delay.value) <= flat_o
    out = np.where(past_ok[:, None] & (flat_t < flat_o)[:, None], past, fut)
    return out.reshape(n, k, len(cols))


# --- models -----------------------------------------------------------------------------------------------------------

class Forecaster:
    """Wraps one model: ``predict(contexts, horizon, covariates)`` -> (point, quantiles[n, horizon, 9])."""

    def __init__(self, name: str, device: str):
        import torch
        self.name, self.device, self.levels = name, device, spec().quantiles
        torch.set_num_threads(2)
        if name == "chronos_bolt":
            from chronos import BaseChronosPipeline
            self.pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map=device,
                                                            torch_dtype=torch.float32)
        elif name.startswith("chronos2"):
            from chronos import Chronos2Pipeline
            self.pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device)
        elif name == "timesfm":
            import timesfm
            # torch.compile needs Triton, which is not available on Windows
            self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch",
                                                                        torch_compile=False)
            self.model.compile(timesfm.ForecastConfig(max_context=2048, max_horizon=256, normalize_inputs=True,
                                                      per_core_batch_size=64, use_continuous_quantile_head=True,
                                                      fix_quantile_crossing=True))
        elif name == "toto":
            from huggingface_hub import snapshot_download
            from toto.inference.forecaster import TotoForecaster
            from toto.model.toto import Toto
            # Toto.from_pretrained does not match the keyword arguments of huggingface_hub 1.x/2.x; load the
            # downloaded folder (weights and config.json) directly
            folder = snapshot_download("Datadog/Toto-Open-Base-1.0")
            model = Toto.load_from_checkpoint(folder, "cpu", False).to(device)
            model.eval()
            self.toto = TotoForecaster(model.model)
        else:
            raise ValueError(name)

    def predict(self, ctx: np.ndarray, horizon: int, step_ns: int, cov_past=None, cov_future=None,
                batch: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        # sized for a 6 GB card: larger batches spill into shared memory and run slower
        batch = batch or {"toto": 16, "timesfm": 64}.get(self.name, 256)
        points, quants = [], []
        for i in range(0, len(ctx), batch):
            sl = slice(i, i + batch)
            p, q = self._predict(ctx[sl], horizon, step_ns,
                                 None if cov_past is None else cov_past[sl],
                                 None if cov_future is None else cov_future[sl])
            points.append(p)
            quants.append(q)
        return np.concatenate(points), np.concatenate(quants)

    def _predict(self, ctx, horizon, step_ns, cov_past, cov_future):
        import torch
        levels = self.levels
        if self.name == "chronos_bolt":
            q, mean = self.pipe.predict_quantiles(torch.tensor(ctx, dtype=torch.float32), prediction_length=horizon,
                                                  quantile_levels=levels)
            return mean.cpu().numpy(), q.cpu().numpy()
        if self.name.startswith("chronos2"):
            inputs = []
            for j in range(len(ctx)):
                item = {"target": ctx[j]}
                if cov_past is not None:
                    item["past_covariates"] = {f"c{k}": cov_past[j, :, k] for k in range(cov_past.shape[2])}
                    item["future_covariates"] = {f"c{k}": cov_future[j, :, k] for k in range(cov_future.shape[2])}
                inputs.append(item)
            q, mean = self.pipe.predict_quantiles(inputs, prediction_length=horizon, quantile_levels=levels)
            q = np.stack([x.squeeze(0).float().cpu().numpy() for x in q])
            mean = np.stack([x.squeeze(0).float().cpu().numpy() for x in mean])
            return mean, q
        if self.name == "timesfm":
            point, qf = self.model.forecast(horizon=horizon, inputs=[c for c in ctx])
            # quantile head: [mean, q10, ..., q90]
            return np.asarray(point), np.asarray(qf)[:, :, 1:10]
        if self.name == "toto":
            from toto.data.util.dataset import MaskedTimeseries
            x = torch.tensor(ctx, dtype=torch.float32, device=self.device).unsqueeze(1)    # (batch, 1 variate, T)
            n, _, T = x.shape
            ts = torch.zeros((n, 1, T), dtype=torch.int64, device=self.device)
            inputs = MaskedTimeseries(series=x, padding_mask=torch.ones_like(x, dtype=torch.bool),
                                      id_mask=torch.zeros((n, 1, T), dtype=torch.int64, device=self.device),
                                      timestamp_seconds=ts,
                                      time_interval_seconds=torch.full((n, 1), int(step_ns // 10**9), dtype=torch.int64,
                                                                       device=self.device))
            with torch.inference_mode():
                fc = self.toto.forecast(inputs, prediction_length=horizon, num_samples=TOTO_SAMPLES,
                                        samples_per_batch=TOTO_SAMPLES)
            samples = fc.samples.squeeze(1).cpu().numpy()          # (batch, horizon, samples)
            q = np.quantile(samples, levels, axis=-1).transpose(1, 2, 0)
            return samples.mean(-1), q
        raise ValueError(self.name)


# --- running a task ---------------------------------------------------------------------------------------------------

def _to_targets(pred_times: np.ndarray, pred: np.ndarray, target_ns: np.ndarray, hourly: bool) -> np.ndarray:
    """Values at the target quarter-hours: direct for 15-minute steps, interpolated at mid-points for hourly ones."""
    if not hourly:
        idx = (target_ns - pred_times[0]) // NS_QH
        return pred[idx]
    mids_h = pred_times + NS_H // 2
    mids_q = target_ns + NS_QH // 2
    return np.interp(mids_q, mids_h, pred)


def run_task(fc: Forecaster, store: Store, track: str, var: str, split: str | list, use_cov: bool) -> pd.DataFrame:
    rows = tasks.rows(track, split)
    origins = pd.DatetimeIndex(rows["origin_utc"].drop_duplicates())
    hourly = track == "day_ahead"
    step = NS_H if hourly else NS_QH
    length = CTX_DA_H if hourly else CTX_ID
    ctx, last = contexts(store, var, origins, length, step)
    ctx = _fill(ctx)
    tmax = rows.groupby("origin_utc")["target_utc"].max()
    horizon = int(((_ns(tmax.to_numpy()) - last) // step).max())
    cov_p = cov_f = None
    if use_cov:
        past_t = last[:, None] - step * np.arange(length - 1, -1, -1)[None, :]
        fut_t = last[:, None] + step * np.arange(1, horizon + 1)[None, :]
        cov_p = _fill_cov(_covariates(store, var, origins, past_t))
        cov_f = _fill_cov(_covariates(store, var, origins, fut_t))
    t0 = time.monotonic()
    point, q = fc.predict(ctx, horizon, step, cov_p, cov_f)
    log.info("%s %s/%s %s: %d origins, horizon %d, %.0f s", fc.name, track, var, split, len(origins), horizon,
             time.monotonic() - t0)
    out = []
    o_ns = _ns(origins)
    for i, (o, part) in enumerate(rows.groupby("origin_utc", sort=True)):
        assert _ns([o])[0] == o_ns[i]
        pt = last[i] + step * np.arange(1, horizon + 1)
        tns = _ns(part["target_utc"])
        f = part[["origin_utc", "target_utc"]].copy()
        f["point"] = _to_targets(pt, point[i], tns, hourly)
        for j, c in enumerate(submission.QCOLS):
            f[c] = _to_targets(pt, q[i, :, j], tns, hourly)
        out.append(f)
    df = pd.concat(out, ignore_index=True)
    if var in ("wind", "pv"):
        df[["point", *submission.QCOLS]] = df[["point", *submission.QCOLS]].clip(lower=0)
    df[submission.QCOLS] = np.sort(df[submission.QCOLS].to_numpy(), axis=1)
    df.insert(0, "variable", var)
    df.insert(0, "track", track)
    return df


def _fill_cov(c: np.ndarray) -> np.ndarray:
    n, k, m = c.shape
    flat = c.transpose(0, 2, 1).reshape(n * m, k)
    flat = pd.DataFrame(flat.T).ffill().bfill().fillna(0.0).to_numpy().T
    return flat.reshape(n, m, k).transpose(0, 2, 1)


def run(root: Path, out_dir: Path, models: list[str] | None = None, device: str = "cpu",
        splits: list[str] = ("test",)) -> None:
    # model weights are cached next to the data unless HF_HOME says otherwise
    os.environ.setdefault("HF_HOME", str(Path(root).resolve() / "hf-cache"))
    store = load_store(root)
    for name in models or MODELS:
        fc = Forecaster(name, device)
        use_cov = name.endswith("_nwp")
        for split in splits:
            frames = [run_task(fc, store, t, v, split, use_cov) for t, v in spec().tasks()]
            path = out_dir / name / f"{split}.parquet"
            submission.write(pd.concat(frames, ignore_index=True), path)
            log.info("%s %s -> %s", name, split, path)
