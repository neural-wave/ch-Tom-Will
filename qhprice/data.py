"""Column names and the hourly aggregation shared by the whole pipeline."""
import pandas as pd

HOUR_KEYS = ["delivery_date", "market_id", "hour_index"]
QUARTER_KEYS = HOUR_KEYS + ["quarter_index"]
TARGET = "price_eur_mwh"                      # quarter-hour price (training data and history)
HOURLY = "hourly_price_eur_mwh"               # hourly price (test_hourly.csv)
PRED = "predicted_price_eur_mwh"              # output column
DEVIATION = "predicted_deviation"             # model output: quarter price minus hourly price
ZONES = ["ZONE_0", "ZONE_1", "ZONE_2", "ZONE_3"]


def read_prices(path) -> pd.DataFrame:
    """Read a CSV with delivery_date and market_id kept as strings."""
    return pd.read_csv(path, dtype={"delivery_date": str, "market_id": str})


def to_hourly(prices: pd.DataFrame) -> pd.DataFrame:
    """Quarter-hour prices -> hourly prices in test_hourly.csv format (mean of the four quarters)."""
    hourly = prices.groupby(HOUR_KEYS, as_index=False)[TARGET].mean()
    return hourly.rename(columns={TARGET: HOURLY})
