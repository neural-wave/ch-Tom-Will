# Methodology

## 1. Task and output representation

For each day and zone we receive 24 hourly prices and must return 96 quarter-hour prices whose
hourly means equal the given prices (within 1e-4 EUR/MWh). The model never predicts prices
directly. It predicts, for every hour, the **deviation** of each quarter from the hourly price.
The four deviations are centred to sum to zero and added to the hourly price
(`qhprice/output.py: deviations_to_prices`), so the hourly-mean rule holds by construction for any
model output. The model only has to learn the shape inside the hour.

## 2. Model

**XGBCrossZone + one fixed shrink factor** (`qhprice/model.py`).

1. **Ramp prior.** If the price rises from the previous hour to the next, the quarters inside the
   hour most likely rise too. The prior is `k · s · [-3, -1, 1, 3] / 8` with `s = (H[h+1] - H[h-1]) / 2`
   and `k = 2` (`qhprice/ramp.py`). This rule alone explains most of the intra-hour shape
   (about 90 % of the variance of the quarter deviations is this linear tilt).
2. **Four XGBoost regressors**, one per quarter, learn the correction to the prior with an L1
   (absolute error) objective, matching the MAE metric. One model serves all four zones; the zone is
   an input feature. Settings: 800 trees, learning rate 0.02, depth 9, min child weight 3,
   subsample 0.8, column sample 0.7, seed 42.
3. **Smoothing at hour boundaries.** The last quarter of an hour and the first of the next are
   pulled 2 % towards each other (gamma = 0.02, hand-picked), changing no hourly mean.
4. **Fixed shrink factor.** All deviations are multiplied by one factor, **0.875**, found at training
   time: the model is trained without the last 28 training days (3–30 Sep 2026), predicts them, and
   the factor between 0.5 and 1.2 with the lowest MAE on those days is kept (held-out MAE 5.23 → 5.08).
   The model is then retrained on all days. The factor is part of the frozen model and is never
   re-estimated at inference.

**Why the factor.** The intra-hour pattern weakened during the year (the ramp inside an hour per
EUR/MWh of hourly slope fell from about 0.32 in October 2025 to 0.24 in September 2026). A model
trained on the past therefore draws the pattern slightly too large on later days. Shrinking towards
the flat hourly price corrects this and, more generally, is a conservative choice when predictions
are uncertain. It improved the score in every forward-in-time test (table below), including every
month of the rolling split and a separate backtest in March, June and September.

## 3. Features (86, all from permitted inputs)

| Group | n | Examples | Source |
|---|---|---|---|
| Hourly curve of the zone | 30 | hourly price, previous/next hour, slope, curvature, jumps into and out of the hour, local peak/trough, distance to the day's peak and trough, day mean/spread | the day's hourly prices (test_hourly) |
| Other zones, same hour | 18 | mean and spread of the four zones' prices and slopes, price and slope difference to each zone | the day's hourly prices of all zones (test_hourly) |
| History | 27 | the quarter pattern of the same hour yesterday and a week ago, 7-day mean and spread of each quarter's deviation, 7-day tilt and bow | quarter prices of strictly earlier dates (history_15min) |
| Calendar | 11 | zone, hour (sin/cos), weekday, weekend | date and zone id |

Cross-zone features are built by zone name for all four zones; if a zone is missing from the input
its features are missing (NaN), which XGBoost handles, and prediction still works (tested).

**No look-ahead.** History features of a day only use earlier dates. `tests/test_no_lookahead.py`
scrambles the quarter prices of one day while keeping its hourly prices, and checks that no feature
of that day or any earlier day changes. At inference `predict.py` itself filters the history to dates
strictly before each predicted date.

## 4. Validation

**Rolling forward-in-time split** (main): for each month May, June, July and August 2026, the model
is trained on all days before the month and predicts the month one day at a time, with only earlier
days as history, exactly as in the official run. Scores are the official MAE (per day and zone, then
per zone, then the mean of the four zones). A forward split matters here: on a random-days split
the model trains on days before and after each validation day, which hides the drift described above
and overstates accuracy.

| Model (rolling split) | May | Jun | Jul | Aug | **Overall** |
|---|---|---|---|---|---|
| Flat baseline (hourly price repeated) | 8.070 | 8.405 | 7.619 | 8.773 | **8.215** |
| Linear interpolation between hours | 5.646 | 5.839 | 5.380 | 6.238 | **5.775** |
| LightGBM (shared features) | 4.076 | 4.030 | 3.884 | 4.577 | **4.143** |
| MLP | 3.985 | 3.961 | 3.912 | 4.459 | **4.080** |
| 1D CNN over the day's curve | 4.099 | 4.034 | 3.830 | 4.394 | **4.090** |
| CatBoost on pattern weights | 3.813 | 3.821 | 3.693 | 4.322 | **3.913** |
| XGBCrossZone | 3.836 | 3.743 | 3.746 | 4.292 | **3.906** |
| **XGBCrossZone + fixed shrink factor (submitted)** | 3.819 | 3.710 | 3.691 | 4.272 | **3.874** |

A second forward split (train to 14 Jul, validate 17 Jul – 31 Aug) gave 4.330 without and 4.217 with
the factor. September, the most volatile month in the data, scores around 5 for every model, so
higher errors on volatile days are expected.

## 5. Design choices and what we left out

- **One model, not an ensemble.** Blends of XGBCrossZone with CatBoost and a CNN reached about
  3.83 on the rolling split, roughly 1 % better, at the cost of two more models, two more libraries
  (CatBoost, PyTorch) and a much larger code base. In line with the challenge's emphasis on justified
  complexity we submit the single model.
- **No re-estimation at inference.** The shrink factor could be re-estimated each day from recent
  history and scored slightly better in development, but that would recalibrate a parameter on
  history supplied after the freeze, which the rules forbid. It is fixed at training time.
- Tried and not kept: per-hour or per-pattern shrink factors (overfit), alignment of hours to the
  day's price landmarks instead of the clock (the clock was better), separate models per hour.

## 6. Rules and reproducibility

- Training, validation and the shrink factor use only the official training set
  (1 Oct 2025 – 30 Sep 2026). No data from 1–18 October 2026, no external data, no pretrained models.
- At inference the model uses the day's hourly prices and quarter prices from strictly earlier dates,
  only through the fixed feature rules above. Nothing is retrained or recalibrated.
- Fixed seeds and pinned versions; `train.py` regenerates `artifacts/`. Inference was verified in a
  fresh Python 3.13 environment installed from `requirements.txt`, giving identical predictions.

## 7. Known limitations

- If the intra-hour pattern became stronger again, the fixed factor would shrink slightly too much
  (the factor changes predictions by about 12 %).
- Cross-zone features assume all four zones are present, as the rules state. With a missing zone the
  pipeline still runs, but those features are missing and predictions are somewhat less accurate.
- Errors concentrate in the morning and evening ramps and in sudden single-quarter spikes that the
  hourly curve does not reveal.
