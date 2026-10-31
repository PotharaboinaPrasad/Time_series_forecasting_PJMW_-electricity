"""PJM West electricity demand forecasting website (Streamlit).

Models available in the app
  * XGBoost, direct multi-horizon (recommended): one model per horizon (1h, 2h, 1 day, 1 week,
    1 month, 1 year), each trained to use only data that is available that far in advance.
    Includes an 80% prediction interval.
  * XGBoost, recursive: your original model (xgb_model.pkl + forecasting.py), if those files are present.
  * SARIMA, ARIMA, Holt-Winters: fitted on the most recent 45 days of the history you provide,
    limited to horizons up to 1 week.
  * Seasonal naive: the "same time last day / week / year" baseline every model has to beat.

Run locally:   streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from pandas.tseries.holiday import USFederalHolidayCalendar
from xgboost import XGBRegressor

# Your original recursive model needs forecasting.py next to this file. It is optional here.
try:
    from forecasting import build_feature_row, build_manual_history, forecast_recursive

    LEGACY_CODE_OK = True
except Exception:  # ImportError, or an error inside forecasting.py
    LEGACY_CODE_OK = False

# --------------------------------------------------------------------------------------
# Paths: everything is relative to this file so the app also works when deployed online.
# --------------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
BUNDLED_HISTORY = DATA_DIR / "pjmw_clean_hourly.csv"
BACKTEST_CSV = DATA_DIR / "horizon_backtest_results.csv"
COVERAGE_CSV = DATA_DIR / "interval_coverage.csv"
OLD_LOCAL_DIR = Path(r"C:\Users\potha\OneDrive\Documents")  # your previous location, used only as a fallback

TRAIN_END = pd.Timestamp("2017-08-03 00:00")  # the direct models were trained on data up to here

DIRECT_HORIZONS = [1, 2, 24, 168, 720, 8760]
HORIZON_OPTIONS = {
    "1 hour": 1,
    "2 hours": 2,
    "1 day (24 hours)": 24,
    "2 days (48 hours)": 48,
    "1 week (168 hours)": 168,
    "1 month (30 days)": 30 * 24,
    "1 year (365 days)": 365 * 24,
}
NAIVE_LAGS = [24, 168, 8736, 8904]  # same time yesterday / last week / ~last year (weekday-aligned)
STAT_WINDOW_DAYS = 45
STAT_MAX_STEPS = 168

M_DIRECT = "XGBoost, direct multi-horizon (recommended)"
M_LEGACY = "XGBoost, recursive (your original model)"
M_SARIMA = "SARIMA"
M_ARIMA = "ARIMA"
M_HW = "Holt-Winters"
M_NAIVE = "Seasonal naive baseline"


# --------------------------------------------------------------------------------------
# Feature engineering. This MUST match the training notebook (Part 2, make_features).
# Every lag / rolling feature is built from data at least `m` hours older than the target.
# --------------------------------------------------------------------------------------
def make_features(y: pd.Series, m: int) -> pd.DataFrame:
    X = pd.DataFrame(index=y.index)
    lags = {m, m + 24, m + 168} | {k for k in [1, 2, 3, 24, 48, 168, 336] if k >= m}
    if m >= 168:
        lags |= {k for k in [8736, 8760] if k >= m}
    for k in sorted(lags):
        X[f"lag_{k}"] = y.shift(k)
    base = y.shift(m)
    for w in [24, 168, 720]:
        X[f"roll_mean_{w}"] = base.rolling(w).mean()
    X["roll_std_24"] = base.rolling(24).std()
    X["roll_max_24"] = base.rolling(24).max()
    X["roll_min_24"] = base.rolling(24).min()
    i = y.index
    X["hour"], X["dow"], X["month"], X["dayofyear"] = i.hour, i.dayofweek, i.month, i.dayofyear
    X["weekofyear"] = i.isocalendar().week.astype(int).values
    X["is_weekend"] = (i.dayofweek >= 5).astype(int)
    hol = USFederalHolidayCalendar().holidays(i.min(), i.max())
    X["is_holiday"] = i.normalize().isin(hol).astype(int)
    X["hour_sin"], X["hour_cos"] = np.sin(2 * np.pi * i.hour / 24), np.cos(2 * np.pi * i.hour / 24)
    X["month_sin"], X["month_cos"] = np.sin(2 * np.pi * i.month / 12), np.cos(2 * np.pi * i.month / 12)
    return X


def min_history_hours(m: int, feats: list[str]) -> int:
    """Hours of consecutive history needed before the forecast origin."""
    max_lag = max(int(f.split("_")[1]) for f in feats if f.startswith("lag_"))
    return max(max_lag, m + 720) + 1


def pick_direct_horizon(steps: int) -> int:
    """Smallest trained model that is valid `steps` hours ahead (a model with horizon m is valid up to m hours)."""
    return next(m for m in DIRECT_HORIZONS if m >= steps)


# --------------------------------------------------------------------------------------
# Loading artifacts
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading forecasting models...")
def load_direct_models() -> dict:
    path = MODELS_DIR / "feature_lists.json"
    if not path.is_file():
        return {}
    feature_lists = json.loads(path.read_text())
    out = {}
    for m in DIRECT_HORIZONS:
        point_path = MODELS_DIR / f"xgb_h{m}.json"
        if not point_path.is_file() or str(m) not in feature_lists:
            continue
        point = XGBRegressor()
        point.load_model(str(point_path))
        quant = None
        quant_path = MODELS_DIR / f"xgb_q_h{m}.json"
        if quant_path.is_file():
            quant = XGBRegressor()
            quant.load_model(str(quant_path))
        out[m] = {"point": point, "quant": quant, "feats": feature_lists[str(m)]}
    return out


def _find_artifact(name: str) -> Path | None:
    for folder in (BASE_DIR, MODELS_DIR, OLD_LOCAL_DIR):
        candidate = folder / name
        if candidate.is_file():
            return candidate
    return None


@st.cache_resource(show_spinner="Loading your original model...")
def load_legacy():
    """Your original recursive XGBoost. Returns None if its files or forecasting.py are missing."""
    if not LEGACY_CODE_OK:
        return None
    import joblib

    model_path, cols_path = _find_artifact("xgb_model.pkl"), _find_artifact("feature_columns.pkl")
    if model_path is None or cols_path is None:
        return None
    model = joblib.load(model_path)
    feature_columns = list(joblib.load(cols_path))
    if not feature_columns or len(feature_columns) != len(set(feature_columns)):
        raise ValueError("feature_columns.pkl must contain unique feature names.")

    holiday_path = _find_artifact("holiday_dates.pkl")
    if holiday_path is not None:
        holiday_dates = {pd.Timestamp(day).date() for day in joblib.load(holiday_path)}
    else:
        try:
            import holidays

            holiday_dates = set(holidays.US(years=range(2000, 2037)).keys())
        except ImportError:
            holiday_dates = set()
    return {"model": model, "feature_columns": feature_columns, "holiday_dates": holiday_dates}


@st.cache_data(show_spinner="Loading bundled PJM West history...")
def load_bundled_history() -> pd.Series | None:
    if not BUNDLED_HISTORY.is_file():
        return None
    df = pd.read_csv(BUNDLED_HISTORY, index_col=0, parse_dates=True)
    s = df.iloc[:, 0].astype(float).asfreq("h")
    s.name = "demand"
    return s


@st.cache_data
def load_accuracy_tables():
    bt = pd.read_csv(BACKTEST_CSV).set_index("hours_ahead") if BACKTEST_CSV.is_file() else None
    cov = pd.read_csv(COVERAGE_CSV) if COVERAGE_CSV.is_file() else None
    return bt, cov


# --------------------------------------------------------------------------------------
# Reading and cleaning an uploaded history
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_table(file_name: str, content: bytes) -> pd.DataFrame:
    import io

    buffer = io.BytesIO(content)
    if file_name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(buffer)
    return pd.read_csv(buffer)


def clean_history(timestamps: pd.Series, demands: pd.Series) -> tuple[pd.Series, list[str]]:
    """Validate, then sort, average duplicate timestamps, reindex to a strict hourly grid, fill short gaps."""
    notes: list[str] = []
    if timestamps.isna().any():
        raise ValueError("The selected timestamp column contains invalid or empty values.")
    if timestamps.dt.tz is not None:
        raise ValueError("Use timezone-naive timestamps in the history file.")
    if demands.isna().any() or not np.isfinite(demands.to_numpy(dtype=float)).all():
        raise ValueError("The selected demand column must contain finite numeric values.")
    if (demands < 0).any():
        raise ValueError("Demand values cannot be negative.")
    if not timestamps.dt.minute.eq(0).all() or not timestamps.dt.second.eq(0).all():
        raise ValueError("All timestamps must be aligned to the start of an hour.")

    s = pd.Series(demands.to_numpy(dtype=float), index=pd.DatetimeIndex(timestamps)).sort_index()
    n_dup = int(s.index.duplicated().sum())
    if n_dup:
        s = s.groupby(level=0).mean()
        notes.append(f"{n_dup} duplicate timestamp(s) were averaged (e.g. the repeated hour when clocks go back).")
    s = s.asfreq("h")
    n_missing = int(s.isna().sum())
    if n_missing:
        if n_missing > 0.05 * len(s):
            raise ValueError(f"{n_missing:,} of {len(s):,} hourly values are missing. Too many gaps to fill safely.")
        s = s.interpolate(method="time")
        notes.append(f"{n_missing} missing hour(s) were filled by interpolation.")

    # Same outlier rule as the training notebook: a reading far from its 25-hour rolling median is a data error.
    dev = s - s.rolling(25, center=True, min_periods=5).median()
    spread = (dev - dev.median()).abs().median() * 1.4826
    if spread > 0:
        outliers = (dev / spread).abs() > 8
        if outliers.any():
            s = s.mask(outliers).interpolate(method="time")
            notes.append(f"{int(outliers.sum())} extreme reading(s) were treated as data errors and interpolated.")
    s.name = "demand"
    return s, notes


# --------------------------------------------------------------------------------------
# Forecasting
# --------------------------------------------------------------------------------------
def _direct_block(series: pd.Series, m: int, steps: int, models: dict) -> pd.DataFrame:
    bundle = models[m]
    need = min_history_hours(m, bundle["feats"])
    if len(series) < need:
        raise ValueError(
            f"The {m}-hour-ahead model needs at least {need:,} consecutive hours of history "
            f"(about {need / 24:,.0f} days); only {len(series):,} are available before the forecast start."
        )
    hist = series.iloc[-(need + 24):]
    index = pd.date_range(hist.index[-1] + pd.Timedelta(hours=1), periods=steps, freq="h")
    extended = pd.concat([hist, pd.Series(np.nan, index=index)])
    X = make_features(extended, m).loc[index, bundle["feats"]]
    if X.isna().any().any():
        raise ValueError("Not enough clean history to build the model inputs.")
    out = pd.DataFrame({"forecast": bundle["point"].predict(X)}, index=index)
    if bundle["quant"] is not None:
        q = np.sort(bundle["quant"].predict(X), axis=1)
        out["lower_10"], out["upper_90"] = q[:, 0], q[:, 2]
    return out


def direct_forecast(series: pd.Series, steps: int, models: dict) -> pd.DataFrame:
    if steps <= 2:  # hour 1 comes from the 1h model, hour 2 from the 2h model
        return pd.concat([_direct_block(series, k, k, models).iloc[[-1]] for k in range(1, steps + 1)])
    return _direct_block(series, pick_direct_horizon(steps), steps, models)


@st.cache_data(show_spinner=False)
def statistical_forecast(name: str, recent: pd.Series, steps: int) -> np.ndarray:
    import warnings

    warnings.filterwarnings("ignore")
    recent = recent.asfreq("h")
    if name == M_ARIMA:
        from statsmodels.tsa.arima.model import ARIMA

        return np.asarray(ARIMA(recent, order=(2, 1, 2)).fit().forecast(steps))
    if name == M_SARIMA:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        fit = SARIMAX(recent, order=(1, 0, 1), seasonal_order=(1, 1, 1, 24),
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=100)
        return np.asarray(fit.forecast(steps))
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    fit = ExponentialSmoothing(recent, trend=None, seasonal="add", seasonal_periods=24,
                               initialization_method="estimated").fit()
    return np.asarray(fit.forecast(steps))


def naive_forecast(series: pd.Series, steps: int) -> np.ndarray:
    lag = next(k for k in NAIVE_LAGS if k >= steps)
    if len(series) < lag:
        raise ValueError(f"The naive baseline needs at least {lag:,} hours of history.")
    return series.iloc[-lag:].to_numpy()[:steps]


def legacy_forecast(legacy: dict, start: pd.Timestamp, steps: int, history: list[float],
                    first_features: dict | None) -> pd.DataFrame:
    holiday_dates = set(legacy["holiday_dates"])
    try:
        import holidays

        final_year = (start + pd.Timedelta(hours=steps - 1)).year
        holiday_dates.update(holidays.US(years=range(start.year, final_year + 1)).keys())
    except ImportError:
        pass
    frame = forecast_recursive(
        model=legacy["model"], feature_columns=legacy["feature_columns"], start=start, steps=steps,
        history=history, holiday_dates=holiday_dates, first_features=first_features,
    )
    return frame.set_index("timestamp")[["predicted_demand_mw"]].rename(columns={"predicted_demand_mw": "forecast"})


def error_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    err = actual - predicted
    return {"MAE (MW)": err.abs().mean(), "RMSE (MW)": float(np.sqrt((err ** 2).mean())),
            "MAPE (%)": float((err.abs() / actual).mean() * 100)}


# --------------------------------------------------------------------------------------
# UI helpers
# --------------------------------------------------------------------------------------
def make_chart(plot_df: pd.DataFrame, band: pd.DataFrame | None):
    import altair as alt

    plot_df = plot_df.rename_axis("timestamp")
    long = plot_df.reset_index().melt("timestamp", var_name="series", value_name="MW")
    lines = alt.Chart(long).mark_line().encode(
        x=alt.X("timestamp:T", title=None),
        y=alt.Y("MW:Q", title="Demand (MW)", scale=alt.Scale(zero=False)),
        color=alt.Color("series:N", title=None, legend=alt.Legend(orient="bottom")),
        tooltip=["timestamp:T", "series:N", alt.Tooltip("MW:Q", format=",.0f")],
    )
    if band is None:
        return lines.interactive()
    area = alt.Chart(band.rename_axis("timestamp").reset_index()).mark_area(opacity=0.18).encode(
        x="timestamp:T", y="lower:Q", y2="upper:Q")
    return (area + lines).interactive()


def show_results(result: dict) -> None:
    frame, steps = result["frame"], result["steps"]
    for note in result["notes"]:
        st.caption(note)

    last = frame.index[-1]
    model_cols = [c for c in frame.columns if c != "Actual"]
    st.success(f"Forecast for {last:%Y-%m-%d %H:%M} ({result['label']} ahead):  " +
               "   |   ".join(f"{c}: {frame.loc[last, c]:,.0f} MW" for c in model_cols))

    plot_df, band = frame.copy(), result["band"]
    if steps > 720:  # keep the chart light for the 1-year horizon
        plot_df = plot_df.resample("D").mean()
        band = band.resample("D").mean() if band is not None else None
        st.caption("Chart shows daily averages for readability. The downloaded CSV is hourly.")
    st.altair_chart(make_chart(plot_df, band), width="stretch")

    if result["band_note"]:
        st.caption(result["band_note"])
    if "Actual" in frame.columns and frame["Actual"].notna().any():
        valid = frame["Actual"].notna()
        rows = {c: error_metrics(frame.loc[valid, "Actual"], frame.loc[valid, c]) for c in model_cols}
        st.subheader("Backtest accuracy on the hours where actual demand is known")
        st.dataframe(pd.DataFrame(rows).T.style.format("{:,.1f}"), width="stretch")

    csv = frame.rename_axis("timestamp")
    if result["band"] is not None:
        csv = csv.join(result["band"].rename(columns={"lower": "interval_lower_10", "upper": "interval_upper_90"}))
    st.download_button("Download hourly forecast CSV", data=csv.to_csv().encode("utf-8"),
                       file_name="pjmw_demand_forecast.csv", mime="text/csv")


def accuracy_note(steps: int) -> None:
    bt, cov = load_accuracy_tables()
    if bt is None:
        return
    m = pick_direct_horizon(steps)
    if m in bt.index:
        row = bt.loc[m]
        text = (f"Measured on a year of data the direct model never saw, the **{row['horizon']}** model had a typical error "
                f"of **{row['MAPE_pct']:.1f}%** (MAPE) versus {row['naive_MAPE_pct']:.1f}% for the seasonal-naive baseline.")
        if cov is not None and (cov["horizon"] == row["horizon"]).any():
            c = float(cov.loc[cov["horizon"] == row["horizon"], "coverage_80_pct"].iloc[0])
            text += f" Its nominal 80% interval actually contained the real value about **{c:.0f}%** of the time, so treat the band as optimistic."
        st.info(text)


# --------------------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="PJMW Demand Forecast", page_icon="⚡", layout="wide")
    st.title("⚡ PJM West Electricity Demand Forecast")
    st.caption(
        "Hourly demand forecasts from 1 hour to 1 year ahead. The recommended models are trained separately for each "
        "horizon (direct forecasting), so long forecasts do not accumulate error the way one-hour-at-a-time forecasts do."
    )

    direct_models = load_direct_models()
    legacy = load_legacy()
    if not direct_models and legacy is None:
        st.error("No models found. Put the `models/` folder (xgb_h*.json, xgb_q_h*.json, feature_lists.json) next to app.py.")
        st.stop()

    horizon_label = st.selectbox("Forecast horizon", list(HORIZON_OPTIONS))
    steps = HORIZON_OPTIONS[horizon_label]

    sources = ["Bundled PJM West history (ends 2018-08-03)", "Upload my own hourly history (CSV or Excel)"]
    if legacy is not None:
        sources.append("Enter lag values manually (original model only)")
    source = st.radio("Starting demand history", sources)

    series: pd.Series | None = None
    notes: list[str] = []
    manual: dict | None = None

    # ---------------- data source ----------------
    if source.startswith("Bundled"):
        series = load_bundled_history()
        if series is None:
            st.error("data/pjmw_clean_hourly.csv was not found.")
            st.stop()
    elif source.startswith("Upload"):
        uploaded = st.file_uploader("Upload hourly history (timestamp + demand in MW)", type=["csv", "xlsx"],
                                    help="Long forecasts need long history: about 31 days for 1 hour to 1 day, "
                                         "about 1 year for 1 week and 1 month, and about 13 months for 1 year.")
        if uploaded is None:
            st.info("Upload your history file to continue, or switch to the bundled data.")
            st.stop()
        try:
            table = read_table(uploaded.name, uploaded.getvalue())
        except Exception as error:
            st.error(f"Could not read the uploaded file: {error}")
            st.stop()
        if len(table.columns) < 2:
            st.error("The file must contain a timestamp column and a demand column.")
            st.stop()
        columns = list(table.columns)
        ts_default = next((i for i, c in enumerate(columns) if str(c).lower() in {"timestamp", "datetime", "date"}), 0)
        dem_default = next((i for i, c in enumerate(columns) if str(c).lower() in {"demand", "mw", "usage", "load", "pjmw_mw"}),
                           1 if ts_default == 0 else 0)
        c1, c2 = st.columns(2)
        ts_col = c1.selectbox("Timestamp column", columns, index=ts_default)
        dem_col = c2.selectbox("Demand (MW) column", columns, index=dem_default)
        if ts_col == dem_col:
            st.error("Choose different columns for timestamps and demand.")
            st.stop()
        try:
            series, notes = clean_history(pd.to_datetime(table[ts_col], errors="coerce"),
                                          pd.to_numeric(table[dem_col], errors="coerce"))
        except (ValueError, TypeError) as error:
            st.error(str(error))
            st.stop()
    else:
        manual = legacy_manual_inputs(legacy)

    # ---------------- origin (backtest) ----------------
    origin = None
    full_series = series
    if series is not None:
        origin = series.index[-1]
        with st.expander("Backtest: forecast from an earlier date and compare with what really happened"):
            use_backtest = st.checkbox("Use an earlier forecast start")
            if use_backtest:
                lo, hi = series.index[0].date(), series.index[-1].date()
                d_col, h_col = st.columns(2)
                day = d_col.date_input("Last observed date", value=hi, min_value=lo, max_value=hi)
                hour = h_col.slider("Last observed hour", 0, 23, 0)
                origin = min(pd.Timestamp(day) + pd.Timedelta(hours=hour), series.index[-1])
                series = series.loc[:origin]
        st.caption(f"History used: {series.index[0]:%Y-%m-%d %H:%M} to {origin:%Y-%m-%d %H:%M} ({len(series):,} hours). "
                   f"The forecast starts at {origin + pd.Timedelta(hours=1):%Y-%m-%d %H:%M}.")
        age_days = (pd.Timestamp.today().normalize() - origin.normalize()).days
        if age_days > 30:
            st.warning(f"This history ends {age_days:,} days ago, so the forecast describes the period right after "
                       f"{origin:%Y-%m-%d}, not the present. For a current forecast, upload data up to today.")
        if direct_models and use_backtest and origin < TRAIN_END + pd.Timedelta(hours=steps):
            st.warning("Part of this forecast window falls inside the period the direct models were trained on "
                       f"(up to {TRAIN_END:%Y-%m-%d}), so a backtest here looks better than it really is. "
                       "Use a start date after 2017-08-03 for a fair test.")

    # ---------------- model choice ----------------
    options: list[str] = []
    if series is not None:
        if direct_models:
            options.append(M_DIRECT)
        if legacy is not None:
            options.append(M_LEGACY)
        if steps <= STAT_MAX_STEPS:
            options += [M_SARIMA, M_HW, M_ARIMA]
        options.append(M_NAIVE)
        defaults = [M_DIRECT] if M_DIRECT in options else options[:1]
        chosen = st.multiselect("Models to run (pick several to compare)", options, default=defaults)
        if steps > STAT_MAX_STEPS:
            st.caption("ARIMA, SARIMA and Holt-Winters are only offered up to 1 week ahead: they are fitted on 45 days of "
                       "history and their forecasts collapse to a flat seasonal pattern beyond that.")
        if M_DIRECT in chosen:
            accuracy_note(steps)
            if 2 < steps < pick_direct_horizon(steps):
                st.caption(f"{horizon_label} uses the {pick_direct_horizon(steps)}-hour-ahead model, the smallest trained "
                           "model that is valid that far ahead.")
    else:
        chosen = [M_LEGACY]
        st.warning("Only the first forecast hour uses your entered values exactly. Later recursive hours use an "
                   "estimated history. Uploading hourly history gives more reliable multi-hour forecasts.")

    # ---------------- run ----------------
    if st.button("Generate forecast", type="primary"):
        if not chosen:
            st.error("Pick at least one model.")
            st.stop()
        try:
            with st.spinner(f"Forecasting {steps:,} hour(s) ahead..."):
                result = run_models(chosen, steps, horizon_label, series, full_series, origin,
                                    direct_models, legacy, manual, notes)
        except ValueError as error:
            st.error(str(error))
            st.stop()
        st.session_state["result"] = result

    if "result" in st.session_state:
        show_results(st.session_state["result"])


def run_models(chosen, steps, label, series, full_series, origin, direct_models, legacy, manual, notes) -> dict:
    out: dict[str, pd.Series] = {}
    band = None
    band_note = ""

    if series is None:  # manual mode, original model only
        frame = legacy_forecast(legacy, manual["start"], steps, manual["history"], manual["first_features"])
        out[M_LEGACY] = frame["forecast"]
    else:
        index = pd.date_range(origin + pd.Timedelta(hours=1), periods=steps, freq="h")
        if M_DIRECT in chosen:
            f = direct_forecast(series, steps, direct_models)
            out[M_DIRECT] = f["forecast"]
            if "lower_10" in f:
                band = f[["lower_10", "upper_90"]].rename(columns={"lower_10": "lower", "upper_90": "upper"})
                band_note = ("Shaded band: 80% prediction interval from quantile models. The point forecast and band come "
                             "from separate models, so the line can sit off-centre inside the band.")
        if M_LEGACY in chosen:
            if len(series) < 168:
                raise ValueError("The original model needs at least 168 hours of history.")
            f = legacy_forecast(legacy, origin + pd.Timedelta(hours=1), steps, series.iloc[-168:].tolist(), None)
            out[M_LEGACY] = f["forecast"]
        for name in (M_SARIMA, M_HW, M_ARIMA):
            if name in chosen:
                recent = series.iloc[-STAT_WINDOW_DAYS * 24:]
                if len(recent) < 7 * 24:
                    raise ValueError(f"{name} needs at least 7 days of history.")
                out[name] = pd.Series(statistical_forecast(name, recent, steps), index=index)
        if M_NAIVE in chosen:
            out[M_NAIVE] = pd.Series(naive_forecast(series, steps), index=index)

    frame = pd.DataFrame(out)
    frame.index = pd.DatetimeIndex(frame.index)
    if full_series is not None:
        actual = full_series.reindex(frame.index)
        if actual.notna().any():
            frame["Actual"] = actual
    local_notes = list(notes)
    if (frame.drop(columns=["Actual"], errors="ignore") < 0).any().any():
        local_notes.append("A model produced a negative demand value. Output is not clipped.")
    return {"frame": frame, "band": band, "band_note": band_note, "steps": steps, "label": label, "notes": local_notes}


def legacy_manual_inputs(legacy: dict) -> dict:
    """Your original manual lag-entry form (original recursive model only)."""
    date_col, hour_col = st.columns(2)
    forecast_date = date_col.date_input("First forecast date")
    forecast_hour = hour_col.slider("First forecast hour", 0, 23, 12)
    start = pd.Timestamp(forecast_date) + pd.Timedelta(hours=forecast_hour)

    st.subheader("Previous demand values (MW)")
    d = 5500.0
    c1, c2, c3 = st.columns(3)
    lag_1 = c1.number_input("1 hour ago", min_value=0.0, value=d, step=100.0)
    lag_2 = c2.number_input("2 hours ago", min_value=0.0, value=d, step=100.0)
    lag_3 = c3.number_input("3 hours ago", min_value=0.0, value=d, step=100.0)
    c4, c5, c6 = st.columns(3)
    lag_24 = c4.number_input("24 hours ago", min_value=0.0, value=d, step=100.0)
    lag_48 = c5.number_input("48 hours ago", min_value=0.0, value=d, step=100.0)
    lag_168 = c6.number_input("168 hours ago (1 week)", min_value=0.0, value=d, step=100.0)
    c7, c8, c9 = st.columns(3)
    rolling_mean_24 = c7.number_input("Average of last 24 hours", min_value=0.0, value=d, step=100.0)
    rolling_mean_168 = c8.number_input("Average of last 168 hours", min_value=0.0, value=d, step=100.0)
    rolling_std_24 = c9.number_input("Std dev of last 24 hours", min_value=0.0, value=300.0, step=50.0)

    history = build_manual_history(lag_1=lag_1, lag_2=lag_2, lag_3=lag_3, lag_24=lag_24, lag_48=lag_48,
                                   lag_168=lag_168, rolling_mean_168=rolling_mean_168)
    first_features = build_feature_row(start, history, legacy["holiday_dates"])
    first_features.update({"lag_1": lag_1, "lag_2": lag_2, "lag_3": lag_3, "lag_24": lag_24, "lag_48": lag_48,
                           "lag_168": lag_168, "rolling_mean_24": rolling_mean_24,
                           "rolling_mean_168": rolling_mean_168, "rolling_std_24": rolling_std_24})
    return {"start": start, "history": history, "first_features": first_features}


if __name__ == "__main__":
    main()
