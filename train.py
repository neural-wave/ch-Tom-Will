"""Train the model from scratch on the official training set and save it to artifacts/.

    python train.py --data data/train_prices_15min.csv

Writes artifacts/model.joblib (the trained model) and artifacts/model_info.json (training period,
the fixed shrink factor and its held-out MAE). predict.py only needs artifacts/.
"""
import argparse
import time

from qhprice.data import read_prices
from qhprice.model import ShrunkXGBCrossZone


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/train_prices_15min.csv", help="official training file")
    parser.add_argument("--out", default="artifacts", help="folder for the trained model")
    args = parser.parse_args()

    t = time.time()
    train = read_prices(args.data)
    print(f"Training on {train['delivery_date'].nunique()} days "
          f"({train['delivery_date'].min()} to {train['delivery_date'].max()})", flush=True)
    model = ShrunkXGBCrossZone().fit(train)
    model.save(args.out)
    print(f"Done in {time.time() - t:.0f} s. Shrink factor {model.scale} "
          f"(held-out MAE {model.info['holdout_mae_unscaled']} -> {model.info['holdout_mae_scaled']}). "
          f"Saved to {args.out}/")


if __name__ == "__main__":
    main()
