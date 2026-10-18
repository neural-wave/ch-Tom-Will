"""Shared feature extraction. Pick the features you want; everything is computed in memory.

The raw file data/train_prices_15min.csv is the only source of truth. Features are never saved
as a separate CSV, so every model always gets the same, up-to-date calculations.

Usage:
    import pandas as pd
    from qhprice.data import to_hourly
    from features import FeatureExtractor, make_targets

    df = pd.read_csv("data/train_prices_15min.csv")
    train = df  # quarter-hour prices

    fe = FeatureExtractor(features=["hour_of_day", "delta_prev", "delta_next"], groups=["history"])
    X_train = fe.transform(to_hourly(train), history=df)
    y_train = make_targets(train, kind="deviation")

    # at submission time: fe.transform(test_hourly, history=history_15min)

Rules:
- Features for a day only use that day's hourly prices plus history from strictly earlier days.
  Passing the full dataset as history is therefore safe: later days are never looked at.
- Natural gaps (hour 1 has no earlier hour that day, a lag day is missing) become NaN, with a
  *_missing flag where it matters. Handling NaN and scaling is the model's job.
- Real data errors (missing hours, NaN or duplicate hourly prices) raise an error.
"""
import warnings

import numpy as np
import pandas as pd

from qhprice.data import HOUR_KEYS, HOURLY, QUARTER_KEYS, TARGET, to_hourly

ZONES = ["ZONE_0", "ZONE_1", "ZONE_2", "ZONE_3"]
QUARTERS = [1, 2, 3, 4]
HISTORY_DAYS = 7

GROUPS: dict[str, list[str]] = {
    "calendar": [
        "zone", "zone_onehot", "hour_of_day", "day_of_week", "is_weekend",
        "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    ],
    "hourly_curve": [
        "hour_price", "prev_hour_price", "next_hour_price", "prev_2h_price", "next_2h_price",
        "delta_prev", "delta_next", "curvature", "is_local_peak", "is_local_trough",
        "is_negative_price", "prev_hour_missing", "next_hour_missing",
    ],
    "day_context": [
        "day_mean_price", "hour_minus_day_mean", "day_price_std", "day_price_range",
        "delta_prev_rel", "delta_next_rel",
    ],
    "history": (
        [f"lag_1d_q{q}_dev" for q in QUARTERS]
        + [f"lag_7d_q{q}_dev" for q in QUARTERS]
        + [f"past_7d_q{q}_dev_mean" for q in QUARTERS]
        + [f"past_7d_q{q}_dev_std" for q in QUARTERS]
        + [
            "lag_1d_hour_price", "lag_7d_hour_price", "hour_price_change_1d", "hour_price_change_7d",
            "lag_1d_spread", "past_7d_spread_mean", "history_days_available",
            "lag_1d_missing", "lag_7d_missing",
        ]
    ),
}
ALL_FEATURES = [f for group in GROUPS.values() for f in group]
# Features that expand into several columns
EXPANDS = {"zone_onehot": [f"zone_{z}" for z in ZONES]}


class FeatureExtractor:
    """Computes the chosen features for hourly prices (+ optional history).

    features: list of feature names (see GROUPS or FeatureExtractor.available()).
    groups:   list of group names, as a shortcut for all features in those groups.
    layout:   "hour"    -> one row per (date, zone, hour)
              "quarter" -> one row per (date, zone, hour, quarter), with quarter_index added
    """

    def __init__(self, features: list[str] | None = None, groups: list[str] | None = None,
                 layout: str = "hour"):
        chosen = list(features or [])
        for g in groups or []:
            if g not in GROUPS:
                raise ValueError(f"Unknown group {g!r}. Groups: {list(GROUPS)}")
            chosen += GROUPS[g]
        if not chosen:
            raise ValueError("Choose at least one feature or group")
        targets = [f for f in chosen if f.startswith("target_")]
        if targets:
            raise ValueError(f"Targets can never be features: {targets}")
        unknown = [f for f in chosen if f not in ALL_FEATURES]
        if unknown:
            raise ValueError(f"Unknown features {unknown}. See FeatureExtractor.available()")
        if layout not in ("hour", "quarter"):
            raise ValueError(f'layout must be "hour" or "quarter", got {layout!r}')
        self.features = list(dict.fromkeys(chosen))  # drop duplicates, keep order
        self.layout = layout

    @staticmethod
    def available() -> dict[str, list[str]]:
        return {g: list(f) for g, f in GROUPS.items()}

    @property
    def columns(self) -> list[str]:
        """The output feature columns (zone_onehot expanded), excluding keys."""
        return [c for f in self.features for c in EXPANDS.get(f, [f])]

    def transform(self, hourly: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
        """hourly:  delivery_date, market_id, hour_index, hourly_price_eur_mwh (e.g. to_hourly(val))
        history: quarter-hour prices in the train schema. Only days before each date are used.
        Needed for history features and for prev_hour_price of hour 1 (yesterday's hour 24).
        """
        _validate_hourly(hourly)
        needs_history = any(f in GROUPS["history"] for f in self.features)
        if needs_history and history is None:
            raise ValueError("History features were chosen, so pass history=...")

        out = _compute(hourly, history)[HOUR_KEYS + self.columns]
        if self.layout == "quarter":
            out = out.merge(pd.DataFrame({"quarter_index": QUARTERS}), how="cross")
            out = out[QUARTER_KEYS + self.columns]
            return out.sort_values(QUARTER_KEYS).reset_index(drop=True)
        return out.sort_values(HOUR_KEYS).reset_index(drop=True)


def make_targets(prices: pd.DataFrame, kind: str = "deviation", layout: str = "hour") -> pd.DataFrame:
    """Targets from quarter-hour prices. Never use these as features.

    kind="deviation": quarter price minus the hourly mean (convert predictions back with deviations_to_prices)
    kind="price":     absolute quarter price
    layout="hour":    columns target_q1_<kind> ... target_q4_<kind>, one row per hour
    layout="quarter": one column target_<kind>, one row per quarter
    """
    if kind not in ("deviation", "price"):
        raise ValueError(f'kind must be "deviation" or "price", got {kind!r}')
    if layout not in ("hour", "quarter"):
        raise ValueError(f'layout must be "hour" or "quarter", got {layout!r}')
    suffix = "dev" if kind == "deviation" else "price"

    p = prices[QUARTER_KEYS + [TARGET]].copy()
    value = p[TARGET] - p.groupby(HOUR_KEYS)[TARGET].transform("mean") if kind == "deviation" else p[TARGET]
    p[f"target_{suffix}"] = value
    if layout == "quarter":
        return p[QUARTER_KEYS + [f"target_{suffix}"]].sort_values(QUARTER_KEYS).reset_index(drop=True)
    wide = p.pivot(index=HOUR_KEYS, columns="quarter_index", values=f"target_{suffix}")
    wide.columns = [f"target_q{q}_{suffix}" for q in wide.columns]
    return wide.reset_index().sort_values(HOUR_KEYS).reset_index(drop=True)


# ---------------------------------------------------------------- internals


def _validate_hourly(hourly: pd.DataFrame) -> None:
    missing_cols = [c for c in HOUR_KEYS + [HOURLY] if c not in hourly.columns]
    if missing_cols:
        raise ValueError(f"hourly is missing columns: {missing_cols}")
    if hourly.duplicated(HOUR_KEYS).any():
        raise ValueError(f"hourly has {hourly.duplicated(HOUR_KEYS).sum()} duplicate hours")
    if hourly[HOURLY].isna().any():
        raise ValueError(f"hourly has {hourly[HOURLY].isna().sum()} NaN hourly prices")
    hours = hourly.groupby(["delivery_date", "market_id"])["hour_index"].agg(lambda h: sorted(h) == list(range(1, 25)))
    if not hours.all():
        bad = hours[~hours].index[:3].tolist()
        raise ValueError(f"{(~hours).sum()} day/zone combinations do not have exactly hours 1-24, e.g. {bad}")


def _shift_date(dates: pd.Series, days: int) -> pd.Series:
    return (pd.to_datetime(dates) - pd.Timedelta(days=days)).dt.strftime("%Y-%m-%d")


def _history_table(history: pd.DataFrame) -> pd.DataFrame:
    """One row per (date, zone, hour) in history: hourly mean, q1..q4 deviations and spread."""
    h = history[QUARTER_KEYS + [TARGET]].copy()
    h["mean"] = h.groupby(HOUR_KEYS)[TARGET].transform("mean")
    h["dev"] = h[TARGET] - h["mean"]
    wide = h.pivot(index=HOUR_KEYS, columns="quarter_index", values="dev")
    wide.columns = [f"q{q}" for q in wide.columns]
    stats = h.groupby(HOUR_KEYS)[TARGET].agg(hour_mean="mean", spread=lambda s: s.max() - s.min())
    return wide.join(stats).reset_index()


def _lookup(base: pd.DataFrame, table: pd.DataFrame, days_back: int, cols: list[str]) -> pd.DataFrame:
    """Values from `table` for the same zone and hour, `days_back` days before each row's date."""
    key = base[HOUR_KEYS].copy()
    key["delivery_date"] = _shift_date(key["delivery_date"], days_back)
    return key.merge(table[HOUR_KEYS + cols], on=HOUR_KEYS, how="left")[cols].set_axis(base.index)


def _compute(hourly: pd.DataFrame, history: pd.DataFrame | None) -> pd.DataFrame:
    df = hourly[HOUR_KEYS + [HOURLY]].sort_values(HOUR_KEYS).reset_index(drop=True)
    price = df[HOURLY]
    df["hour_price"] = price
    day = pd.to_datetime(df["delivery_date"])

    # Calendar and zone
    df["zone"] = pd.Categorical(df["market_id"], categories=ZONES)
    for z in ZONES:
        df[f"zone_{z}"] = (df["market_id"] == z).astype(int)
    df["hour_of_day"] = df["hour_index"] - 1
    df["day_of_week"] = day.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Hourly curve: neighbours within the same day; hour 1-2 may look back into yesterday via history
    g = df.groupby(["delivery_date", "market_id"])[HOURLY]
    df["prev_hour_price"], df["next_hour_price"] = g.shift(1), g.shift(-1)
    df["prev_2h_price"], df["next_2h_price"] = g.shift(2), g.shift(-2)
    if history is not None and len(history):
        hist_hourly = to_hourly(history)
        for col, hour, lag in [("prev_hour_price", 1, 24), ("prev_2h_price", 1, 23), ("prev_2h_price", 2, 24)]:
            rows = df["hour_index"] == hour
            key = df.loc[rows, ["delivery_date", "market_id"]].copy()
            key["delivery_date"] = _shift_date(key["delivery_date"], 1)
            key["hour_index"] = lag
            found = key.merge(hist_hourly, on=HOUR_KEYS, how="left")[HOURLY].to_numpy()
            df.loc[rows, col] = found
    df["delta_prev"] = price - df["prev_hour_price"]
    df["delta_next"] = df["next_hour_price"] - price
    df["curvature"] = df["delta_next"] - df["delta_prev"]
    both = df["prev_hour_price"].notna() & df["next_hour_price"].notna()
    df["is_local_peak"] = ((price > df["prev_hour_price"]) & (price > df["next_hour_price"])).astype(float).where(both)
    df["is_local_trough"] = ((price < df["prev_hour_price"]) & (price < df["next_hour_price"])).astype(float).where(both)
    df["is_negative_price"] = (price < 0).astype(int)
    df["prev_hour_missing"] = df["prev_hour_price"].isna().astype(int)
    df["next_hour_missing"] = df["next_hour_price"].isna().astype(int)

    # Whole-day context
    gd = df.groupby(["delivery_date", "market_id"])[HOURLY]
    df["day_mean_price"] = gd.transform("mean")
    df["hour_minus_day_mean"] = price - df["day_mean_price"]
    df["day_price_std"] = gd.transform("std")
    df["day_price_range"] = gd.transform("max") - gd.transform("min")
    std = df["day_price_std"].replace(0, np.nan)
    df["delta_prev_rel"] = df["delta_prev"] / std
    df["delta_next_rel"] = df["delta_next"] / std

    # History: same zone and hour on earlier days only
    hist_cols = GROUPS["history"]
    if history is None or not len(history):
        for c in hist_cols:
            df[c] = np.nan
        df["history_days_available"] = 0
        df["lag_1d_missing"] = df["lag_7d_missing"] = 1
        return df

    table = _history_table(history)
    qcols = [f"q{q}" for q in QUARTERS]
    lag1 = _lookup(df, table, 1, qcols + ["hour_mean", "spread"])
    lag7 = _lookup(df, table, 7, qcols + ["hour_mean"])
    for q in QUARTERS:
        df[f"lag_1d_q{q}_dev"] = lag1[f"q{q}"]
        df[f"lag_7d_q{q}_dev"] = lag7[f"q{q}"]
    df["lag_1d_hour_price"] = lag1["hour_mean"]
    df["lag_7d_hour_price"] = lag7["hour_mean"]
    df["hour_price_change_1d"] = price - df["lag_1d_hour_price"]
    df["hour_price_change_7d"] = price - df["lag_7d_hour_price"]
    df["lag_1d_spread"] = lag1["spread"]
    df["lag_1d_missing"] = df["lag_1d_hour_price"].isna().astype(int)
    df["lag_7d_missing"] = df["lag_7d_hour_price"].isna().astype(int)

    window = [_lookup(df, table, d, qcols + ["spread"]) for d in range(1, HISTORY_DAYS + 1)]
    stacked = {c: np.column_stack([w[c].to_numpy() for w in window]) for c in qcols + ["spread"]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN rows -> NaN, as intended
        for q in QUARTERS:
            vals = stacked[f"q{q}"]
            df[f"past_7d_q{q}_dev_mean"] = np.nanmean(vals, axis=1)
            df[f"past_7d_q{q}_dev_std"] = np.where(
                (~np.isnan(vals)).sum(axis=1) >= 2, np.nanstd(vals, axis=1, ddof=1), np.nan)
        df["past_7d_spread_mean"] = np.nanmean(stacked["spread"], axis=1)
    df["history_days_available"] = (~np.isnan(stacked["q1"])).sum(axis=1)
    return df
