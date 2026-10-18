"""End-to-end: predict.py with the trained artifacts gives compliant output, also with odd inputs."""
import subprocess
import sys
from pathlib import Path

import pytest

from qhprice.data import read_prices, to_hourly
from qhprice.output import check_output

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "train_prices_15min.csv"
needs = pytest.mark.skipif(not (DATA.exists() and (ROOT / "artifacts" / "model.joblib").exists()),
                           reason="needs data/ and trained artifacts/")


def _run(tmp_path, test, history):
    test_csv, out = tmp_path / "test_hourly.csv", tmp_path / "pred.csv"
    test.to_csv(test_csv, index=False)
    cmd = [sys.executable, "predict.py", "--test", str(test_csv), "--output", str(out)]
    if history is not None:
        history.to_csv(tmp_path / "history.csv", index=False)
        cmd += ["--history", str(tmp_path / "history.csv")]
    subprocess.run(cmd, cwd=ROOT, check=True)
    return read_prices(out)


@needs
@pytest.mark.parametrize("case", ["normal", "no_history", "missing_zone", "two_dates"])
def test_output_is_compliant(tmp_path, case):
    df = read_prices(DATA)
    dates = ["2026-09-29", "2026-09-30"] if case == "two_dates" else ["2026-09-30"]
    test = to_hourly(df[df["delivery_date"].isin(dates)])
    if case == "missing_zone":
        test = test[test["market_id"] != "ZONE_2"]
    history = None if case == "no_history" else df[df["delivery_date"] < dates[0]]
    pred = _run(tmp_path, test, history)
    assert check_output(pred, test) == []
    assert len(pred) == len(test) * 4
