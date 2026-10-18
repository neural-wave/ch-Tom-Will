"""The interface every model implements, so run_evaluation.py can train and score any of them.

    fit(X, y, history) -> self
        X:       hourly train input, the output of to_hourly(train)
        y:       the train frame with quarter-hour prices, for models that need targets
        history: the full price data. Use it for lag features on the train rows; for each
                 row only earlier days may be used (FeatureExtractor already does this).

    predict(X, history) -> DataFrame
        X:       hourly prices for ONE day (all zones), like test_hourly.csv
        history: quarter-hour prices for every day strictly before that day,
                 like history_15min.csv at submission time
        Returns delivery_date, market_id, hour_index, quarter_index and either
        predicted_price_eur_mwh, or predicted_deviation if PREDICTS_DEVIATIONS = True.

The runner, not the model, builds history for each day from the full data, so every model
sees exactly what it would see at submission time and nothing from the future.
"""
import pandas as pd


class BaseModel:
    # Set to True in a subclass whose predict returns predicted_deviation
    PREDICTS_DEVIATIONS = False

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, history: pd.DataFrame) -> "BaseModel":
        raise NotImplementedError("Subclasses should implement this method.")

    def predict(self, X: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Subclasses should implement this method.")
