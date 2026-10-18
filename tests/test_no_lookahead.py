"""No feature of a day may use that day's quarter prices or any later day's.

Scrambles the quarters of one day inside each hour (the hourly prices, i.e. the model input, stay
identical) and checks that no feature of that day or of earlier days changes.
"""
from pathlib import Path

import numpy as np
import pytest

from qhprice.data import HOUR_KEYS, read_prices, to_hourly
from qhprice.xgb_model import XGBCrossZone

DATA = Path(__file__).resolve().parent.parent / "data" / "train_prices_15min.csv"


@pytest.mark.skipif(not DATA.exists(), reason="needs data/train_prices_15min.csv")
@pytest.mark.parametrize("scrambled_day", ["2026-03-03", "2026-03-08"])
def test_features_never_see_the_day_itself_or_later(scrambled_day):
    df = read_prices(DATA)
    days = sorted(d for d in df["delivery_date"].unique() if "2026-03-01" <= d <= "2026-03-10")
    df = df[df["delivery_date"] <= days[-1]]
    mod = df.copy()
    sel = mod["delivery_date"] == scrambled_day
    mod.loc[sel, "price_eur_mwh"] = mod[sel].groupby(HOUR_KEYS)["price_eur_mwh"].transform(
        lambda s: s.sample(frac=1, random_state=1).to_numpy())
    hourly = to_hourly(df[df["delivery_date"].isin(days)]).sort_values(HOUR_KEYS).reset_index(drop=True)
    hourly_mod = to_hourly(mod[mod["delivery_date"].isin(days)]).sort_values(HOUR_KEYS).reset_index(drop=True)
    assert np.allclose(hourly["hourly_price_eur_mwh"], hourly_mod["hourly_price_eur_mwh"])  # same input

    model = XGBCrossZone()
    a, b = model._features(hourly, df), model._features(hourly, mod)
    cols = [c for c in a.columns if c not in ("delivery_date", "market_id", "hour_index", "zone")]
    changed = ~np.isclose(a[cols].to_numpy(float), b[cols].to_numpy(float), equal_nan=True)
    changed_days = set(a.loc[changed.any(axis=1), "delivery_date"])
    assert all(d > scrambled_day for d in changed_days), sorted(changed_days)
