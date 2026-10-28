"""Solar geometry and clear-sky irradiance (NOAA solar position, Haurwitz clear-sky model)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def cos_zenith(times: pd.DatetimeIndex, lat: float, lon: float) -> np.ndarray:
    """Cosine of the solar zenith angle at UTC ``times`` (NOAA general solar position equations)."""
    t = pd.DatetimeIndex(times).tz_convert("UTC")
    doy = t.day_of_year.to_numpy()
    hour = (t.hour + t.minute / 60 + t.second / 3600).to_numpy()
    g = 2 * np.pi / 365 * (doy - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                       - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    tst = hour * 60 + eqtime + 4 * lon
    ha = np.deg2rad(tst / 4 - 180)
    phi = np.deg2rad(lat)
    return np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(ha)


def clear_sky_ghi(cosz: np.ndarray) -> np.ndarray:
    """Haurwitz clear-sky global horizontal irradiance (W/m2)."""
    c = np.clip(cosz, 1e-6, None)
    return np.where(cosz > 0, 1098.0 * c * np.exp(-0.059 / c), 0.0)
