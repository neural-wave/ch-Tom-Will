"""Turn predicted deviations into prices, check the output against the challenge rules, score it."""
import numpy as np
import pandas as pd

from qhprice.data import DEVIATION, HOUR_KEYS, HOURLY, PRED, QUARTER_KEYS, TARGET

TOLERANCE = 1e-4  # max |mean of the four quarters - hourly price|, EUR/MWh (challenge section 9)


def deviations_to_prices(hourly: pd.DataFrame, deviations: pd.DataFrame) -> pd.DataFrame:
    """Centre each hour's four deviations so they sum to zero and add them to the hourly price.

    This makes every hour's mean equal the given hourly price by construction.
    """
    d = deviations[QUARTER_KEYS + [DEVIATION]].merge(
        hourly[HOUR_KEYS + [HOURLY]], on=HOUR_KEYS, how="left", validate="many_to_one")
    if d[HOURLY].isna().any():
        raise ValueError(f"{int(d[HOURLY].isna().sum())} predicted rows have no hourly price")
    centred = d[DEVIATION] - d.groupby(HOUR_KEYS)[DEVIATION].transform("mean")
    d[PRED] = d[HOURLY] + centred
    return d.sort_values(QUARTER_KEYS)[QUARTER_KEYS + [PRED]].reset_index(drop=True)


def check_output(pred: pd.DataFrame, hourly: pd.DataFrame) -> list[str]:
    """The jury's compliance checks (section 9). Returns a list of problems; empty means compliant."""
    problems = []
    missing = [c for c in QUARTER_KEYS + [PRED] if c not in pred.columns]
    if missing:
        return [f"missing columns {missing}"]
    if pred.duplicated(QUARTER_KEYS).any():
        problems.append(f"{int(pred.duplicated(QUARTER_KEYS).sum())} duplicated rows")
    values = pd.to_numeric(pred[PRED], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        problems.append("non-finite or non-numeric predicted prices")
    if not pred["quarter_index"].isin([1, 2, 3, 4]).all():
        problems.append("quarter_index outside 1-4")
    counts = pred.groupby(HOUR_KEYS).size()
    if (counts != 4).any():
        problems.append(f"{int((counts != 4).sum())} hours without exactly four quarters")
    given = hourly.set_index(HOUR_KEYS)[HOURLY]
    missing_hours = given.index.difference(counts.index)
    if len(missing_hours):
        problems.append(f"{len(missing_hours)} hours of the input have no prediction")
    gap = (pred.groupby(HOUR_KEYS)[PRED].mean() - given).abs().dropna()
    if (gap > TOLERANCE).any():
        problems.append(f"{int((gap > TOLERANCE).sum())} hours whose mean differs from the hourly price "
                        f"by more than {TOLERANCE} (max {gap.max():.2e})")
    return problems


def official_mae(pred: pd.DataFrame, truth: pd.DataFrame) -> tuple[float, dict]:
    """Section 10: MAE per day and zone -> mean over days per zone -> simple mean over zones."""
    d = pred.merge(truth[QUARTER_KEYS + [TARGET]], on=QUARTER_KEYS)
    err = (d[PRED] - d[TARGET]).abs()
    per_zone = err.groupby([d["delivery_date"], d["market_id"]]).mean().groupby(level=1).mean()
    return float(per_zone.mean()), {z: float(v) for z, v in per_zone.items()}
