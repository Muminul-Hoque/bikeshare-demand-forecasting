# Capital Bikeshare Demand: Analysis & Forecasting

EDA, regression, and hourly demand forecasting on two years of Capital Bikeshare (Washington, D.C.)
rental data — 17,379 hourly records, Jan 2011–Dec 2012.

## What's here

- **EDA** — distributional analysis, weather/temporal patterns, casual vs. registered user behavior.
- **Regression** — Negative Binomial model (chosen after confirming heavy overdispersion; variance
  is ~174x the mean), with VIF-based multicollinearity checks.
- **Forecasting** — hourly demand prediction with a strict temporal split (train: days 1–20 of each
  month, test: days 21–end), leakage-safe feature engineering, and four models compared head-to-head.

## Key results

**Regression (rate ratios, all p < 0.001 unless noted):**

| Effect | Rate Ratio |
|---|---|
| Temperature +10°C | 1.37 |
| Light rain/snow vs. clear | 0.60 |
| Working day vs. weekend | 0.82 |
| 2012 vs. 2011 | 1.60 |

Casual riders are far more weather-sensitive than registered riders (+91% vs. +29% demand per 10°C)
and strongly avoid working days — consistent with registered trips being commuting and casual trips
being discretionary.

**Forecasting (24 monthly test windows, days 21–end of each month):**

| Model | MAE | RMSE | RMSLE |
|---|---|---|---|
| Historical baseline | 50.57 | 88.69 | 0.57 |
| Random Forest | 43.58 | 72.65 | **0.52** |
| XGBoost | 35.80 | 61.49 | 0.53 |
| MLP (128, 64) | **34.48** | **56.78** | 0.58 |

The MLP wins on absolute error (best in 19/24 windows); Random Forest is more reliable on relative
error at low-volume hours. No lookahead: `casual`/`registered` (direct components of the target) are
excluded from features, and the main engineered feature (`historical_demand_profile`) only uses
calendar-aligned demand from *before* each test window.

## Running it

```
pip install -r requirements.txt
jupyter notebook bikeshare_forecasting.ipynb
```

Dataset: [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)
(Fanaee-T & Gama, 2013). Update the data path near the top of the notebook to point at your local copy.

## Data note

This analysis encodes `season` as 1=Spring, 2=Summer, 3=Fall, 4=Winter, which differs from the UCI
repository's own documentation (1=Winter). That schema is kept consistent throughout — worth knowing
if you're cross-referencing coefficients against the raw UCI docs.
