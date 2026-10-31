# PJM West demand forecast website

## Run it
```
pip install -r requirements.txt
streamlit run app.py
```

## Folder layout (keep it exactly like this)
```
app.py
requirements.txt
models/   xgb_h*.json, xgb_q_h*.json, feature_lists.json   <- direct multi-horizon models + intervals
data/     pjmw_clean_hourly.csv, horizon_backtest_results.csv, interval_coverage.csv
```
Optional, only if you want your original recursive model inside the app: put `forecasting.py`, `xgb_model.pkl`,
`feature_columns.pkl` (and `holiday_dates.pkl` if you have it) next to `app.py`.

## Put it online (free)
1. Push this folder to a GitHub repository.
2. Go to https://share.streamlit.io, choose the repo, set the main file to `app.py`, deploy.

## Notes
- The direct models were trained on data up to 2017-08-03. Backtests starting after that date are genuinely out-of-sample.
- To forecast from today you need history that ends today. Upload it in the app, or retrain with the notebooks.
- The model files must be loaded with the same feature code that trained them (`make_features` in app.py).
  Do not change that function without retraining.
