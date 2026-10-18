"""L2 / L3 · XGBoost on intra-hour deviations.

XGBPlain   (L2): 4 regressors (one per quarter) predict the deviation from the hourly price.
XGBHybrid  (L3): the ramp rule (models/ramp.py) gives a prior; 4 regressors learn only the residual.
                 prediction = ramp prior + residual. With few/no trees it falls back to the ramp rule.

Both use the shared features.py (calendar, hourly curve, day context; no history features),
objective reg:absoluteerror (the metric is MAE), fixed seeds and thread count. Output is
deviations; the runner turns them into prices with deviations_to_prices (hourly mean exact).

Run from the repo root:
    python run_evaluation.py models.xgb_model:XGBPlain
    python run_evaluation.py models.xgb_model:XGBHybrid
    python run_evaluation.py models.xgb_model:XGBHybrid --param k=1.8 --param n_estimators=800
"""
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from qhprice.data import HOUR_KEYS
from qhprice.features import FeatureExtractor, make_targets
from qhprice.base_model import BaseModel
from qhprice.ramp import ramp_deviations, ramp_slope, to_quarter_frame

SEED = 42
FEATURE_GROUPS = ["calendar", "hourly_curve", "day_context"]
EXCLUDED = ["zone"]  # categorical dtype; zone_onehot carries the same information
ZONES = ["ZONE_0", "ZONE_1", "ZONE_2", "ZONE_3"]  # the four zones in every official file
TARGET_COLUMNS = [f"target_q{q}_dev" for q in (1, 2, 3, 4)]

DEFAULT_XGB = {
    "n_estimators": 600,
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}
INT_PARAMS = {"n_estimators", "max_depth"}


def _cast(params: dict) -> dict:
    """run_evaluation --param passes strings; convert to numeric type where appropriate."""
    res = {}
    for k, v in params.items():
        if k in INT_PARAMS:
            res[k] = int(float(v))
        else:
            try:
                res[k] = float(v)
            except (ValueError, TypeError):
                res[k] = v
    return res


class XGBPlain(BaseModel):
    """k is only used by XGBHybrid (ramp prior strength); XGBPlain ignores it.

    No history argument yet: hour 1 has no previous hour (NaN). History features come in Phase 4.
    """

    MODEL_NAME = "xgb_plain"
    PREDICTS_DEVIATIONS = True
    USES_RAMP_PRIOR = False

    FEATURE_GROUPS = ["calendar", "hourly_curve", "day_context"]

    def __init__(self, k: float = 2.0, seed: int = SEED, n_jobs: int = 4, **xgb_params):
        super().__init__()
        self.k = float(k)
        self.seed = int(seed)
        self.n_jobs = int(n_jobs)
        self.xgb_params = {**DEFAULT_XGB, **_cast(xgb_params)}
        groups_features = [f for g in self.FEATURE_GROUPS for f in FeatureExtractor.available()[g]]
        self.extractor = FeatureExtractor(features=[f for f in groups_features if f not in EXCLUDED])
        self.models: list[XGBRegressor] = []

    # -- features
    def _features(self, hourly: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
        hist = history if history is not None else pd.DataFrame(columns=["delivery_date", "market_id", "hour_index", "quarter_index", "price_eur_mwh"])
        X = self.extractor.transform(hourly, history=hist)  # sorted by HOUR_KEYS
        X["ramp_slope"] = ramp_slope(hourly).to_numpy()
        return X

    def _prior(self, hourly: pd.DataFrame) -> np.ndarray:
        if not self.USES_RAMP_PRIOR:
            return np.zeros((len(hourly), 4))
        return ramp_deviations(hourly, self.k)

    def _regressor(self) -> XGBRegressor:
        params = {
            "objective": "reg:absoluteerror",
            "tree_method": "hist",
            "random_state": self.seed,
            "n_jobs": self.n_jobs,
            **self.xgb_params,
        }
        return XGBRegressor(**params)

    # -- BaseModel contract
    def fit(self, X: pd.DataFrame, y: pd.DataFrame,
            history: pd.DataFrame | None = None) -> "XGBPlain":
        """X: hourly train prices (to_hourly(train)); y: quarter-hour train prices."""
        hourly = X.sort_values(HOUR_KEYS).reset_index(drop=True)
        feats = self._features(hourly, history)
        targets = make_targets(y, kind="deviation", layout="hour")
        aligned = feats[HOUR_KEYS].merge(targets, on=HOUR_KEYS, how="left", validate="one_to_one")
        if aligned[TARGET_COLUMNS].isna().any().any():
            raise ValueError("Some training hours have no quarter-hour targets")
        residual = aligned[TARGET_COLUMNS].to_numpy() - self._prior(hourly)
        cols = self.feature_columns
        self.models = [self._regressor().fit(feats[cols], residual[:, q]) for q in range(4)]
        return self

    def predict(self, X: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
        hourly = X.sort_values(HOUR_KEYS).reset_index(drop=True)
        feats = self._features(hourly, history)
        dev = self._prior(hourly) + np.column_stack(
            [m.predict(feats[self.feature_columns]) for m in self.models])
        return to_quarter_frame(hourly, dev)

    # -- introspection (for the report)
    @property
    def feature_columns(self) -> list[str]:
        return self.extractor.columns + ["ramp_slope"]

    def feature_importance(self, top: int = 10) -> pd.Series:
        imp = np.mean([m.feature_importances_ for m in self.models], axis=0)
        return pd.Series(imp, index=self.feature_columns).sort_values(ascending=False).head(top)


class XGBHybrid(XGBPlain):
    MODEL_NAME = "xgb_hybrid"
    USES_RAMP_PRIOR = True


class XGBHistory(XGBHybrid):
    MODEL_NAME = "xgb_history"
    FEATURE_GROUPS = ["calendar", "hourly_curve", "day_context", "history"]


class XGBCrossZone(XGBHistory):
    """L6 · Cross-Zone & Boundary-Coupled Hybrid XGBoost.

    Enriches XGBHistory with:
    1. Boundary coupling: jump_in (H_h - H_{h-1}), jump_out (H_{h+1} - H_h), asymmetry_ratio
    2. Dynamic slope features: abs_slope, extreme ramp indicator
    3. 7-day shape dynamics: past_7d_tilt (q4 - q1), past_7d_bow ((q1+q4)/2 - (q2+q3)/2)
    4. Diurnal peak/trough timing: dist_to_peak, dist_to_trough
    5. Contemporaneous cross-zone spreads and slopes across available zones in the day
    """

    MODEL_NAME = "xgb_cross_zone"

    DEFAULT_XGB = {
        "n_estimators": 800,
        "learning_rate": 0.02,
        "max_depth": 9,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
    }

    def __init__(self, k: float = 2.0, seed: int = SEED, n_jobs: int = 4, **xgb_params):
        super().__init__(k=k, seed=seed, n_jobs=n_jobs, **xgb_params)
        self.user_xgb_params = _cast(xgb_params)
        self.xgb_params = {**self.DEFAULT_XGB, **self.user_xgb_params}
        self._extra_columns: list[str] = []

    def _regressor_for_quarter(self, q: int) -> XGBRegressor:
        """Quarter-specialized parameters, respecting any explicit user/test overrides."""
        is_boundary = q in (0, 3)
        depth = self.user_xgb_params.get("max_depth", 10 if is_boundary else 7)
        n_est = self.user_xgb_params.get("n_estimators", 900 if is_boundary else 700)
        min_child = self.user_xgb_params.get("min_child_weight", 2 if is_boundary else 6)
        lr = self.user_xgb_params.get("learning_rate", 0.02)
        sub = self.user_xgb_params.get("subsample", 0.8)
        col = self.user_xgb_params.get("colsample_bytree", 0.7)

        params = {
            "objective": "reg:absoluteerror",
            "tree_method": "hist",
            "random_state": self.seed + q,
            "n_jobs": self.n_jobs,
            "n_estimators": n_est,
            "max_depth": depth,
            "min_child_weight": min_child,
            "learning_rate": lr,
            "subsample": sub,
            "colsample_bytree": col,
        }
        return XGBRegressor(**params)

    def _features(self, hourly: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
        feats = super()._features(hourly, history)
        h_sorted = hourly.sort_values(HOUR_KEYS).reset_index(drop=True)
        price = h_sorted["hourly_price_eur_mwh"]

        # 1. Midnight-aware boundary coupling
        g = h_sorted.groupby(["delivery_date", "market_id"])["hourly_price_eur_mwh"]
        next_h = g.shift(-1).fillna(price)
        if "prev_hour_price" in feats.columns and feats["prev_hour_price"].notna().any():
            prev_h = feats["prev_hour_price"].fillna(price)
        else:
            prev_h = g.shift(1).fillna(price)

        feats["jump_in"] = (price - prev_h).to_numpy()
        feats["jump_out"] = (next_h - price).to_numpy()
        denom = feats["jump_out"].abs() + feats["jump_in"].abs() + 1e-4
        feats["asymmetry_ratio"] = (feats["jump_out"] - feats["jump_in"]) / denom

        # Curvature acceleration (second discrete derivative) and relative scaling
        feats["jump_accel"] = (feats["jump_out"] - feats["jump_in"]).to_numpy()
        feats["rel_jump_in"] = (feats["jump_in"] / (price.abs() + 10.0)).to_numpy()
        feats["rel_jump_out"] = (feats["jump_out"] / (price.abs() + 10.0)).to_numpy()

        # Update slope using connected midnight boundary
        true_slope = (next_h - prev_h) / 2.0
        feats["ramp_slope"] = true_slope.to_numpy()
        feats["abs_slope"] = feats["ramp_slope"].abs()
        feats["is_extreme_ramp"] = (feats["abs_slope"] > 20.0).astype(float)

        # 2. 7-day rolling tilt and bow shapes
        if "past_7d_q1_dev_mean" in feats.columns and "past_7d_q4_dev_mean" in feats.columns:
            feats["past_7d_tilt"] = feats["past_7d_q4_dev_mean"] - feats["past_7d_q1_dev_mean"]
            q_mid = (feats["past_7d_q2_dev_mean"] + feats["past_7d_q3_dev_mean"]) / 2.0
            q_edge = (feats["past_7d_q1_dev_mean"] + feats["past_7d_q4_dev_mean"]) / 2.0
            feats["past_7d_bow"] = q_edge - q_mid

        # 3. Whole-day timing (peak and trough hours)
        day_peaks = h_sorted.groupby(["delivery_date", "market_id"])["hourly_price_eur_mwh"].idxmax()
        peak_hours = h_sorted.loc[day_peaks, ["delivery_date", "market_id", "hour_index"]].rename(
            columns={"hour_index": "peak_hour"})
        day_troughs = h_sorted.groupby(["delivery_date", "market_id"])["hourly_price_eur_mwh"].idxmin()
        trough_hours = h_sorted.loc[day_troughs, ["delivery_date", "market_id", "hour_index"]].rename(
            columns={"hour_index": "trough_hour"})

        h_meta = h_sorted.merge(peak_hours, on=["delivery_date", "market_id"]).merge(
            trough_hours, on=["delivery_date", "market_id"])
        feats["dist_to_peak"] = (h_meta["hour_index"] - h_meta["peak_hour"]).abs().to_numpy()
        feats["dist_to_trough"] = (h_meta["hour_index"] - h_meta["trough_hour"]).abs().to_numpy()

        # 4. Cross-zone features. Always built for all four zones, looked up by name, so the
        #    columns match training even when a zone is missing from the input; a missing
        #    zone's features are NaN (XGBoost handles NaN) and the means use the zones present.
        h_piv = h_sorted.pivot(index=["delivery_date", "hour_index"], columns="market_id",
                              values="hourly_price_eur_mwh").reindex(columns=ZONES).reset_index()
        s_piv = h_sorted.assign(s=feats["ramp_slope"].to_numpy()).pivot(
            index=["delivery_date", "hour_index"], columns="market_id", values="s").reindex(columns=ZONES).reset_index()
        s_piv.columns = ["delivery_date", "hour_index"] + [f"slope_{z}" for z in ZONES]

        h_piv["cz_mean_price"] = h_piv[ZONES].mean(axis=1)
        h_piv["cz_std_price"] = h_piv[ZONES].std(axis=1).fillna(0)
        s_piv["cz_mean_slope"] = s_piv[[f"slope_{z}" for z in ZONES]].mean(axis=1)
        s_piv["cz_std_slope"] = s_piv[[f"slope_{z}" for z in ZONES]].std(axis=1).fillna(0)

        cz_all = h_piv.merge(s_piv, on=["delivery_date", "hour_index"])
        m = h_sorted.merge(cz_all, on=["delivery_date", "hour_index"], how="left")

        feats["cz_mean_price"] = m["cz_mean_price"].to_numpy()
        feats["cz_std_price"] = m["cz_std_price"].to_numpy()
        feats["cz_mean_slope"] = m["cz_mean_slope"].to_numpy()
        feats["cz_std_slope"] = m["cz_std_slope"].to_numpy()
        feats["spread_to_cz_mean"] = price.to_numpy() - feats["cz_mean_price"]
        feats["slope_diff_to_cz_mean"] = feats["ramp_slope"].to_numpy() - feats["cz_mean_slope"]

        for z in ZONES:
            feats[f"spread_to_{z}"] = price.to_numpy() - m[z].to_numpy()
            feats[f"slope_{z}"] = m[f"slope_{z}"].to_numpy()
            feats[f"slope_diff_to_{z}"] = feats["ramp_slope"].to_numpy() - m[f"slope_{z}"].to_numpy()

        self._extra_columns = [c for c in feats.columns if c not in HOUR_KEYS + ["zone"]
                               and c not in super().feature_columns]
        return feats

    @property
    def feature_columns(self) -> list[str]:
        return super().feature_columns + self._extra_columns

    def fit(self, X: pd.DataFrame, y: pd.DataFrame,
            history: pd.DataFrame | None = None) -> "XGBCrossZone":
        hourly = X.sort_values(HOUR_KEYS).reset_index(drop=True)
        # Strictly use y as training history to prevent cross-block exposure and train robust lag splits
        train_hist = y if (history is None or len(history) > len(y)) else history
        feats = self._features(hourly, train_hist)
        targets = make_targets(y, kind="deviation", layout="hour")
        aligned = feats[HOUR_KEYS].merge(targets, on=HOUR_KEYS, how="left", validate="one_to_one")
        residual = aligned[TARGET_COLUMNS].to_numpy() - self._prior(hourly)
        cols = self.feature_columns
        self.models = [self._regressor_for_quarter(q).fit(feats[cols], residual[:, q]) for q in range(4)]
        return self

    def predict(self, X: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
        hourly = X.sort_values(HOUR_KEYS).reset_index(drop=True)
        feats = self._features(hourly, history)
        dev = self._prior(hourly) + np.column_stack(
            [m.predict(feats[self.feature_columns]) for m in self.models])

        # Area-preserving inter-hour boundary continuity relaxation (gamma = 0.020, a hand-picked value)
        gamma = 0.020
        for (d, z), group in hourly.groupby(["delivery_date", "market_id"]):
            z_idx = group.index.to_numpy()
            if len(z_idx) > 1:
                h_prices = group["hourly_price_eur_mwh"].to_numpy()
                p = h_prices[:, None] + dev[z_idx]
                for h in range(len(z_idx) - 1):
                    gap = p[h + 1, 0] - p[h, 3]
                    adj = gamma * gap
                    dev[z_idx[h], 3] += adj / 2.0
                    dev[z_idx[h], 0] -= adj / 2.0
                    dev[z_idx[h + 1], 0] -= adj / 2.0
                    dev[z_idx[h + 1], 3] += adj / 2.0

        return to_quarter_frame(hourly, dev)


