"""Inference entry point: predict the 96 quarter-hour prices per zone for every date in test_hourly.csv.

    python predict.py --test test_hourly.csv --history history_15min.csv --output predictions.csv

Uses the trained model in artifacts/ (no training, no network). Each date is predicted with only
the history from strictly earlier dates. The output is checked against the challenge rules before
it is written (four finite prices per hour, mean equal to the hourly price within 1e-4 EUR/MWh).
"""
import argparse
import sys
import time

import pandas as pd

from qhprice.data import HOUR_KEYS, HOURLY, QUARTER_KEYS, read_prices
from qhprice.model import ShrunkXGBCrossZone
from qhprice.output import check_output, deviations_to_prices


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", required=True, help="test_hourly.csv")
    parser.add_argument("--history", default=None, help="history_15min.csv (optional)")
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--model", default="artifacts", help="folder with the trained model")
    args = parser.parse_args()

    t = time.time()
    model = ShrunkXGBCrossZone.load(args.model)
    test = read_prices(args.test)[HOUR_KEYS + [HOURLY]]
    history = read_prices(args.history) if args.history else None

    parts = []
    for date, day in test.groupby("delivery_date"):
        hist = history[history["delivery_date"] < date] if history is not None else None
        parts.append(deviations_to_prices(day, model.predict_deviations(day, hist)))
    pred = pd.concat(parts, ignore_index=True).sort_values(QUARTER_KEYS).reset_index(drop=True)

    problems = check_output(pred, test)
    if problems:
        print("Output is NOT compliant, nothing written:\n  - " + "\n  - ".join(problems), file=sys.stderr)
        return 1
    pred.to_csv(args.output, index=False)
    print(f"Wrote {len(pred)} rows for {test['delivery_date'].nunique()} date(s) to {args.output} "
          f"in {time.time() - t:.1f} s (shrink factor {model.scale}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
