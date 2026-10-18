"""The submitted model: XGBCrossZone with one fixed shrink factor on its predicted deviations.

XGBCrossZone (qhprice/xgb_model.py) predicts, for every hour, how far each of the four quarters
lies from the hourly price: a ramp rule gives a starting point and four XGBoost regressors (one
per quarter) learn the correction from features of the day's hourly curve, the other zones and
earlier days' quarter prices.

On days after its training period the model draws the intra-hour pattern slightly too large
(the pattern weakened over the year). The deviations are therefore multiplied by one factor,
found once at training time:
  1. hold out the last `holdout_days` training days,
  2. train XGBCrossZone on the earlier days and predict the held-out days,
  3. keep the factor in SCALES with the lowest official MAE on them,
  4. train XGBCrossZone again on all training days.
After training the factor is fixed. Inference never re-estimates it.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from qhprice.data import DEVIATION, HOUR_KEYS, HOURLY, QUARTER_KEYS, TARGET
from qhprice.xgb_model import XGBCrossZone

SCALES = np.round(np.arange(0.5, 1.201, 0.025), 3)
SEED = 42


class ShrunkXGBCrossZone:
    def __init__(self, holdout_days: int = 28, seed: int = SEED):
        self.holdout_days = int(holdout_days)
        self.seed = int(seed)
        self.scale: float | None = None
        self.info: dict = {}
        self.model: XGBCrossZone | None = None

    def fit(self, train: pd.DataFrame) -> "ShrunkXGBCrossZone":
        """train: quarter-hour prices (training-data format). Features only use earlier days."""
        hourly = train.groupby(HOUR_KEYS, as_index=False)[TARGET].mean().rename(columns={TARGET: HOURLY})
        days = sorted(train["delivery_date"].unique())
        holdout = days[-self.holdout_days:]
        inner = ~train["delivery_date"].isin(holdout)
        inner_h = ~hourly["delivery_date"].isin(holdout)

        probe = XGBCrossZone(seed=self.seed).fit(hourly[inner_h], train[inner], train[inner])
        # History = all training days: a held-out day's lag features then see the held-out days
        # before it, as at inference. No feature of a day uses that day or later days
        # (tests/test_no_lookahead.py), so predicting all held-out days in one call equals
        # predicting them one by one.
        dev = probe.predict(hourly[~inner_h], train)
        maes = self._mae_per_scale(dev, train[~inner], hourly[~inner_h])
        self.scale = float(maes.idxmin())
        self.info = {"holdout_start": holdout[0], "holdout_end": holdout[-1],
                     "holdout_mae_unscaled": round(float(maes.loc[1.0]), 4),
                     "holdout_mae_scaled": round(float(maes.min()), 4), "scale": self.scale,
                     "train_start": days[0], "train_end": days[-1], "train_days": len(days)}

        self.model = XGBCrossZone(seed=self.seed).fit(hourly, train, train)
        return self

    def predict_deviations(self, hourly: pd.DataFrame, history: pd.DataFrame | None) -> pd.DataFrame:
        """Deviations from the hourly price for every quarter of `hourly` (one or more days)."""
        out = self.model.predict(hourly, history)
        out[DEVIATION] = self.scale * out[DEVIATION]
        return out

    @staticmethod
    def _mae_per_scale(dev: pd.DataFrame, truth: pd.DataFrame, hourly: pd.DataFrame) -> pd.Series:
        d = dev.merge(truth[QUARTER_KEYS + [TARGET]], on=QUARTER_KEYS).merge(hourly, on=HOUR_KEYS)
        centred = d[DEVIATION] - d.groupby(HOUR_KEYS)[DEVIATION].transform("mean")
        actual = d[TARGET] - d[HOURLY]
        res = {}
        for s in SCALES:
            err = (s * centred - actual).abs()
            res[float(s)] = err.groupby([d["delivery_date"], d["market_id"]]).mean().groupby(level=1).mean().mean()
        return pd.Series(res)

    def save(self, folder: str | Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, folder / "model.joblib")
        (folder / "model_info.json").write_text(json.dumps(self.info, indent=2))

    @staticmethod
    def load(folder: str | Path) -> "ShrunkXGBCrossZone":
        return joblib.load(Path(folder) / "model.joblib")
