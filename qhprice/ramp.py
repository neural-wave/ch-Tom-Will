"""L1 · Ramp rule: the intra-hour "sawtooth" as a one-parameter formula (no training).

Inside each hour the quarter-hour price ramps from the previous hour's level towards the next
hour's level, about twice as steeply as smooth interpolation (see docs/RESEARCH.md, F1):

    deviation_q = k * s * [-3, -1, 1, 3] / 8,   s = (H(h+1) - H(h-1)) / 2

At the day edges the missing neighbour is padded with the hour's own price (the next day is
unknown at prediction time). deviations_to_prices then restores the exact hourly mean.

Run from the repo root:
    python run_evaluation.py models.ramp:RampRule
    python run_evaluation.py models.ramp:RampRule --param k=1.8
"""
import numpy as np
import pandas as pd

from qhprice.data import DEVIATION, HOUR_KEYS, HOURLY
from qhprice.base_model import BaseModel

RAMP_TEMPLATE = np.array([-3.0, -1.0, 1.0, 3.0]) / 8.0
QUARTER_KEYS = HOUR_KEYS + ["quarter_index"]


def ramp_slope(hourly: pd.DataFrame) -> pd.Series:
    """Neighbour slope s = (H(h+1) - H(h-1)) / 2 per row of `hourly` (sorted by HOUR_KEYS).

    Missing neighbours (hour 1, hour 24) are padded with the hour's own price.
    """
    h = hourly.sort_values(HOUR_KEYS).reset_index(drop=True)
    g = h.groupby(["delivery_date", "market_id"])[HOURLY]
    prev = g.shift(1).fillna(h[HOURLY])
    nxt = g.shift(-1).fillna(h[HOURLY])
    return (nxt - prev) / 2.0


def ramp_deviations(hourly: pd.DataFrame, k: float) -> np.ndarray:
    """Ramp-rule deviations, shape (n_hours, 4), rows in HOUR_KEYS order."""
    return k * ramp_slope(hourly).to_numpy()[:, None] * RAMP_TEMPLATE[None, :]


def to_quarter_frame(hourly: pd.DataFrame, deviations: np.ndarray) -> pd.DataFrame:
    """(n_hours, 4) deviation array -> long frame with QUARTER_KEYS + predicted_deviation."""
    keys = hourly[HOUR_KEYS].sort_values(HOUR_KEYS).reset_index(drop=True)
    out = keys.loc[keys.index.repeat(4)].reset_index(drop=True)
    out["quarter_index"] = np.tile([1, 2, 3, 4], len(keys))
    out[DEVIATION] = deviations.reshape(-1)
    return out[QUARTER_KEYS + [DEVIATION]]


class RampRule(BaseModel):
    """deviation = k * neighbour slope * [-3, -1, 1, 3] / 8. One parameter, nothing learned."""

    MODEL_NAME = "ramp_rule"
    PREDICTS_DEVIATIONS = True

    def __init__(self, k: float = 2.0):
        super().__init__()
        self.k = float(k)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | None = None,
            history: pd.DataFrame | None = None) -> "RampRule":
        return self

    def predict(self, X: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
        return to_quarter_frame(X, ramp_deviations(X, self.k))
