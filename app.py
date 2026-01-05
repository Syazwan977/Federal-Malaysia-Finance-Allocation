import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =========================
# Config
# =========================
st.set_page_config(page_title="Malaysia Federal Finance Allocation Dashboard", layout="wide")
st.title("📊 Data Exploration Dashboard: Federal Finance Allocation (1970–2023)")

# =========================
# Data Source (Cloud + Local compatible)
# =========================
st.sidebar.header("1) Data Source")

data_mode = st.sidebar.radio(
    "Choose data source",
    ["Use repo dataset (GitHub/Streamlit Cloud)", "Upload CSV", "Use local path (VS Code only)"],
    index=0
)

REPO_DATA_PATH = Path("data") / "federal_finance.csv"
LOCAL_DATA_PATH = r"C:\Users\User\OneDrive - National Defence University of Malaysia\Documents\BIG DATA\federal_finance_year_de (1).csv"

@st.cache_data
def load_data(path_or_buffer):
    df = pd.read_csv(path_or_buffer)

    # basic cleaning
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna(subset=["year", "value"])
    df["year"] = df["year"].astype(int)

    # normalize text columns
    for c in ["category", "function"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower()

    return df

if data_mode == "Use repo dataset (GitHub/Streamlit Cloud)":
    if not REPO_DATA_PATH.exists():
        st.error(
            "❌ Dataset not found in Streamlit Cloud.\n\n"
            "✅ Fix your GitHub repo structure to:\n"
            "data/federal_finance.csv\n\n"
            "Open your GitHub repo → create folder `data/` → upload the CSV renamed as `federal_finance.csv`."
        )

        st.write("### Debug: Files in repo root")
        st.code("\n".join(sorted([p.name for p in Path('.').iterdir() if p.is_file()])))
        if Path("data").exists():
            st.write("### Debug: Files in data/ folder")
            st.code("\n".join(sorted([p.name for p in Path('data').iterdir() if p.is_file()])))
        else:
            st.warning("No `data/` folder found in the deployed repo.")

        st.stop()

    df = load_data(REPO_DATA_PATH)
    st.sidebar.success("Loaded: data/federal_finance.csv ✅")

elif data_mode == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload your CSV (same format)", type=["csv"])
    if uploaded is None:
        st.info("Upload your CSV to begin.")
        st.stop()
    df = load_data(uploaded)

else:  # Local path
    try:
        df = load_data(LOCAL_DATA_PATH)
        st.sidebar.success("Loaded local file ✅")
    except FileNotFoundError:
        st.error(
            "❌ Local file not found.\n\n"
            f"Check your path:\n{LOCAL_DATA_PATH}\n\n"
            "Or switch to 'Upload CSV' / 'Use repo dataset'."
        )
        st.stop()

# =========================
# Helper: build sector view
# =========================
def build_wide(df, level="category", include_total=False):
    d = df.copy()

    total_rows = d[(d["category"] == "total") & (d["function"] == "total")]
    if len(total_rows) > 0:
        total_series = total_rows.groupby("year")["value"].sum().sort_index()
    else:
        total_series = None

    if level == "category":
        sector_col = "category"
        if not include_total:
            d = d[d["category"] != "total"]

        cat_tot = d[d["function"] == "total"]
        if len(cat_tot) > 0:
            g = cat_tot.groupby(["year", sector_col])["value"].sum().reset_index()
        else:
            g = d.groupby(["year", sector_col])["value"].sum().reset_index()

    elif level == "function":
        sector_col = "function"
        if not include_total:
            d = d[d["function"] != "total"]
        g = d.groupby(["year", sector_col])["value"].sum().reset_index()

    else:
        raise ValueError("level must be 'category' or 'function'")

    wide = g.pivot(index="year", columns=sector_col, values="value").sort_index()

    if total_series is None:
        total_series = wide.sum(axis=1)

    return wide, total_series

def summary_stats(wide_df):
    stats = pd.DataFrame({
        "mean": wide_df.mean(),
        "median": wide_df.median(),
        "variance": wide_df.var(ddof=1),
        "min": wide_df.min(),
        "min_year": wide_df.idxmin(),
        "max": wide_df.max(),
        "max_year": wide_df.idxmax(),
        "latest": wide_df.iloc[-1],
        "latest_year": wide_df.index.max()
    }).sort_values("latest", ascending=False)
    return stats

def detect_outliers_zscore(series, window=7, z_thresh=2.5):
    s = series.dropna().copy()
    rm = s.rolling(window, min_periods=max(3, window // 2)).mean()
    rs = s.rolling(window, min_periods=max(3, window // 2)).std(ddof=0)
    z = (s - rm) / rs.replace(0, np.nan)

    out = pd.DataFrame({
        "year": s.index,
        "value": s.values,
        "rolling_mean": rm.values,
        "rolling_std": rs.values,
        "zscore": z.values,
        "is_outlier": (np.abs(z) >= z_thresh)
    })
    return out

# =========================
# Sidebar controls
# =========================
st.sidebar.header("2) Dashboard Controls")

level = st.sidebar.selectbox("Sector level", ["category", "function"], index=0)
include_total = st.sidebar.checkbox("Include 'total' as a sector (if applicable)", value=False)

wide, total_series = build_wide(df, level=level, include_total=include_total)

years = wide.index.tolist()
min_year, max_year = int(min(years)), int(max(years))
year_range = st.sidebar.slider("Year range", min_year, max_year, (min_year, max_year))

wide_f = wide.loc[year_range[0]:year_range[1]].copy()
total_f = total_series.loc[year_range[0]:year_range[1]].copy()

sector_list = list(wide_f.columns)
default_sel = sector_list[: min(5, len(sector_list))]
selected_sectors = st.sidebar.multiselect("Select sectors", sector_list, default=default_sel)

if not selected_sectors:
    st.warning("Select at least one sector.")
    st.stop()

wide_sel = wide_f[selected_sectors]

# =========================
# Layout
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1) Summary Statistics",
    "2) Distribution + Outliers",
    "3) Correlations",
    "4) Trends + Prioritization + Benchmarks",
    "5) Forecasting (ARIMA vs LR)"
])

# =========================
# TAB 1
# =========================
with tab1:
    st.subheader("1. Summary Statistics")

    stats = summary_stats(wide_sel)
    colA, colB = st.columns([1.2, 0.8])

    with colA:
        st.write("**Descriptive statistics (by selected sectors):**")
        st.dataframe(stats, use_container_width=True)

    with colB:
        latest_year = wide_sel.index.max()
        latest_values = wide_sel.loc[latest_year].sort_values(ascending=False)

        st.write(f"**Largest allocation (latest year = {latest_year})**")
        st.metric(label=str(latest_values.index[0]), value=float(latest_values.iloc[0]))

        st.write(f"**Smallest allocation (latest year = {latest_year})**")
        st.metric(label=str(latest_values.index[-1]), value=float(latest_values.iloc[-1]))

    st.divider()
    st.subheader("Quick view: Latest year bar chart")
    fig_latest = px.bar(
        x=latest_values.index, y=latest_values.values,
        labels={"x": "Sector", "y": "Allocation"},
        title=f"Allocations by Sector (Latest Year: {latest_year})"
    )
    st.plotly_chart(fig_latest, use_container_width=True)

# =========================
# TAB 2
# =========================
with tab2:
    st.subheader("2. Distribution Analysis")

    c1, c2 = st.columns(2)
    long_df = wide_sel.reset_index().melt(id_vars="year", var_name="sector", value_name="allocation")

    with c1:
        fig_hist = px.histogram(
            long_df, x="allocation", color="sector", barmode="overlay",
            nbins=30, title="Histogram of Allocations (Selected Sectors)"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        fig_box = px.box(
            long_df, x="sector", y="allocation",
            title="Boxplot of Allocations (Outliers visible as points)"
        )
        st.plotly_chart(fig_box, use_container_width=True)

    st.divider()
    st.subheader("Skewness + Simple outlier flags (IQR rule)")

    skew = wide_sel.skew(numeric_only=True)
    outlier_flags = []
    for s in selected_sectors:
        x = wide_sel[s].dropna()
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((x < lo) | (x > hi)).sum())
        outlier_flags.append((s, float(skew.get(s, np.nan)), n_out))

    skew_df = pd.DataFrame(outlier_flags, columns=["sector", "skewness", "iqr_outlier_count"]).sort_values(
        "iqr_outlier_count", ascending=False
    )
    st.dataframe(skew_df, use_container_width=True)

    st.divider()
    st.subheader("7. Outlier Detection (Rolling Z-score)")

    z_window = st.slider("Rolling window (years)", 3, 15, 7)
    z_thresh = st.slider("Z-score threshold", 1.5, 4.0, 2.5, 0.1)

    outlier_tables = []
    for s in selected_sectors:
        out = detect_outliers_zscore(wide_sel[s], window=z_window, z_thresh=z_thresh)
        out["sector"] = s
        outlier_tables.append(out)

    out_all = pd.concat(outlier_tables, ignore_index=True)
    out_only = out_all[out_all["is_outlier"]].sort_values(["sector", "year"])

    c3, c4 = st.columns([1, 1])
    with c3:
        st.write("**Detected outlier years (by sector):**")
        st.dataframe(out_only[["sector", "year", "value", "zscore"]], use_container_width=True)

    with c4:
        sector_for_plot = st.selectbox("Plot outliers for sector", selected_sectors, index=0)
        out_s = out_all[out_all["sector"] == sector_for_plot].set_index("year")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=wide_sel.index, y=wide_sel[sector_for_plot], mode="lines+markers", name="allocation"))
        out_pts = out_s[out_s["is_outlier"]]
        fig.add_trace(go.Scatter(
            x=out_pts.index, y=out_pts["value"], mode="markers",
            name="outliers", marker=dict(size=12, symbol="x")
        ))
        fig.update_layout(title=f"Outliers for {sector_for_plot} (Rolling z-score)")
        st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 3
# =========================
with tab3:
    st.subheader("3. Correlation Analysis")

    st.write("**Pearson correlation among selected sectors (same year):**")
    corr = wide_sel.corr(method="pearson")
    fig_corr = px.imshow(corr, text_auto=True, title="Correlation Heatmap (Pearson)")
    st.plotly_chart(fig_corr, use_container_width=True)

    st.divider()
    st.subheader("Previous year vs current year correlation (Lag-1)")

    lag_rows = []
    for s in selected_sectors:
        cur = wide_sel[s]
        prev = wide_sel[s].shift(1)
        valid = pd.concat([cur, prev], axis=1).dropna()
        r = valid.iloc[:, 0].corr(valid.iloc[:, 1]) if len(valid) >= 3 else np.nan
        lag_rows.append((s, float(r) if pd.notna(r) else np.nan))

    lag_df = pd.DataFrame(lag_rows, columns=["sector", "corr(current, previous_year)"]).sort_values(
        "corr(current, previous_year)", ascending=False
    )
    st.dataframe(lag_df, use_container_width=True)

    st.info("GDP growth correlation needs a separate GDP growth dataset (year + GDP growth).")

# =========================
# TAB 4
# =========================
with tab4:
    st.subheader("4. Trend Analysis (1970–2023)")

    long_tr = wide_sel.reset_index().melt(id_vars="year", var_name="sector", value_name="allocation")
    fig_line = px.line(long_tr, x="year", y="allocation", color="sector", markers=True, title="Allocation Trends")
    st.plotly_chart(fig_line, use_container_width=True)

    st.divider()
    st.subheader("Percentage Change Analysis")

    pct = wide_sel.pct_change() * 100
    long_pct = pct.reset_index().melt(id_vars="year", var_name="sector", value_name="pct_change")
    fig_pct = px.line(long_pct, x="year", y="pct_change", color="sector", markers=True,
                      title="Year-on-Year % Change (Selected Sectors)")
    st.plotly_chart(fig_pct, use_container_width=True)

    st.divider()
    st.subheader("5. Sector Prioritization (Share of Total)")

    total_use = total_f.reindex(wide_f.index).loc[year_range[0]:year_range[1]].replace(0, np.nan)
    share = wide_sel.divide(total_use, axis=0) * 100
    long_share = share.reset_index().melt(id_vars="year", var_name="sector", value_name="share_pct")

    c5, c6 = st.columns(2)
    with c5:
        fig_share = px.area(long_share, x="year", y="share_pct", color="sector",
                            title="Share of Total Allocation Over Time (%)")
        st.plotly_chart(fig_share, use_container_width=True)

    with c6:
        year_for_pie = st.selectbox("Choose year for pie chart", sorted(wide_sel.index.tolist()),
                                    index=len(wide_sel.index) - 1)
        pie_vals = share.loc[year_for_pie].dropna().sort_values(ascending=False)
        fig_pie = px.pie(values=pie_vals.values, names=pie_vals.index,
                         title=f"Sector Share of Total (%) in {year_for_pie}")
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()
    st.subheader("6. Regional Allocation (if data available)")
    if "state" in df.columns:
        st.success("Detected a 'state' column ✅ Showing state-level analysis.")
        reg = df.copy()
        reg = reg[reg["year"].between(year_range[0], year_range[1])]
        reg_p = reg.groupby(["year", "state"])["value"].sum().reset_index()
        pivot_reg = reg_p.pivot(index="state", columns="year", values="value")
        fig_hm = px.imshow(pivot_reg, aspect="auto", title="State Allocation Heatmap")
        st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.info("No state-level column found in this dataset.")

    st.divider()
    st.subheader("8. Comparative Analysis (ASEAN benchmarking)")
    st.write("Upload ASEAN CSV with columns: `country, year, sector, value`")
    asean_file = st.file_uploader("Upload ASEAN benchmark CSV (optional)", type=["csv"], key="asean")

    if asean_file is not None:
        asean = pd.read_csv(asean_file)
        for c in ["country", "sector"]:
            if c in asean.columns:
                asean[c] = asean[c].astype(str).str.strip().str.lower()
        asean["year"] = pd.to_numeric(asean["year"], errors="coerce").astype("Int64")
        asean["value"] = pd.to_numeric(asean["value"], errors="coerce")
        asean = asean.dropna(subset=["year", "value", "country", "sector"])

        sector_pick = st.selectbox("Choose sector", sorted(asean["sector"].unique()))
        years_common = sorted(set(asean["year"].dropna().astype(int).unique()) & set(wide_sel.index))

        if not years_common:
            st.warning("No overlapping years with your dataset.")
        else:
            yr_pick = st.selectbox("Choose year", years_common, index=len(years_common) - 1)

            comp = asean[(asean["sector"] == sector_pick) & (asean["year"] == yr_pick)].copy()
            comp = comp.sort_values("value", ascending=False)

            my_val = None
            if sector_pick in wide.columns and yr_pick in wide.index:
                my_val = wide.loc[yr_pick, sector_pick]

            if my_val is not None and pd.notna(my_val):
                comp = pd.concat([comp, pd.DataFrame([{
                    "country": "malaysia (from your dataset)",
                    "year": yr_pick,
                    "sector": sector_pick,
                    "value": float(my_val)
                }])], ignore_index=True)

            fig_cmp = px.bar(comp, x="country", y="value", title=f"ASEAN Benchmark: {sector_pick} ({yr_pick})")
            st.plotly_chart(fig_cmp, use_container_width=True)
    else:
        st.info("No ASEAN file uploaded.")

# =========================
# TAB 5: Forecasting (ARIMA vs Linear Regression) + COMPARISON PLOTS ✅
# =========================
with tab5:
    st.subheader("9. Forecasting Allocation Trends (ARIMA vs Linear Regression)")

    st.write(
        "Compare **ARIMA** (time-series) vs **Simple Linear Regression** (trend using year). "
        "Both models forecast the next N years. Optionally evaluate using a holdout (last N years)."
    )

    sector_forecast = st.selectbox("Select sector to forecast", list(wide_sel.columns), key="fc_sector")
    horizon = st.slider("Forecast horizon (years)", 1, 10, 5, key="fc_horizon")

    st.divider()
    st.markdown("### Optional: Holdout evaluation (last N years as test set)")
    use_holdout = st.checkbox("Enable holdout evaluation", value=True, key="use_holdout")
    test_size = st.slider("Test size (years)", 3, 12, 5, key="test_size") if use_holdout else 0

    st.divider()
    st.markdown("### ARIMA settings")
    colp, cold, colq = st.columns(3)
    with colp:
        p = st.number_input("AR (p)", 0, 5, 1, key="arima_p_fc")
    with cold:
        d = st.number_input("Diff (d)", 0, 2, 1, key="arima_d_fc")
    with colq:
        q = st.number_input("MA (q)", 0, 5, 1, key="arima_q_fc")

    series = wide_sel[sector_forecast].dropna()

    if len(series) < 12:
        st.warning("Not enough data points. Please select a wider year range or another sector.")
        st.stop()

    # Train/Test split
    if use_holdout and test_size > 0 and len(series) > test_size + 5:
        train = series.iloc[:-test_size]
        test = series.iloc[-test_size:]
    else:
        train = series
        test = None

    # Linear Regression
    X_train = train.index.values.reshape(-1, 1).astype(float)
    y_train = train.values.astype(float)
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    last_train_year = int(train.index.max())
    fc_years = list(range(last_train_year + 1, last_train_year + horizon + 1))
    X_future = np.array(fc_years).reshape(-1, 1).astype(float)
    lr_fc = lr.predict(X_future)

    # ARIMA
    arima_ok = True
    try:
        arima_model = ARIMA(train, order=(int(p), int(d), int(q)))
        arima_fitted = arima_model.fit()
        arima_fc = arima_fitted.forecast(steps=horizon).values
    except Exception as e:
        arima_ok = False
        arima_fc = np.array([np.nan] * horizon)
        st.error("ARIMA model failed. Try a different order (p,d,q) like (1,1,1) or (0,1,1).")
        st.code(str(e))

    # Forecast comparison table
    fc_df = pd.DataFrame({
        "year": fc_years,
        "ARIMA_forecast": arima_fc,
        "LinearRegression_forecast": lr_fc
    })

    # =========================
    # PLOT 1: Actual + Forecasts
    # =========================
    hist_df = pd.DataFrame({"year": series.index.astype(int), "actual": series.values})

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=hist_df["year"], y=hist_df["actual"],
        mode="lines+markers", name="Actual"
    ))
    fig1.add_trace(go.Scatter(
        x=fc_df["year"], y=fc_df["ARIMA_forecast"],
        mode="lines+markers", name="ARIMA Forecast",
        line=dict(dash="dash")
    ))
    fig1.add_trace(go.Scatter(
        x=fc_df["year"], y=fc_df["LinearRegression_forecast"],
        mode="lines+markers", name="Linear Regression Forecast",
        line=dict(dash="dot")
    ))
    fig1.update_layout(
        title=f"Forecast Comparison (Actual vs ARIMA vs Linear Regression): {sector_forecast}",
        xaxis_title="Year",
        yaxis_title="Allocation"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # =========================
    # PLOT 2: Predicted values only (ARIMA vs LR)
    # =========================
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=fc_df["year"], y=fc_df["ARIMA_forecast"],
        mode="lines+markers", name="ARIMA Predicted",
        line=dict(dash="dash")
    ))
    fig2.add_trace(go.Scatter(
        x=fc_df["year"], y=fc_df["LinearRegression_forecast"],
        mode="lines+markers", name="Linear Regression Predicted",
        line=dict(dash="dot")
    ))
    fig2.update_layout(
        title=f"Predicted Values Only (ARIMA vs Linear Regression): {sector_forecast}",
        xaxis_title="Year",
        yaxis_title="Predicted Allocation"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # =========================
    # PLOT 3: Difference (ARIMA - LR)
    # =========================
    diff_df = fc_df.copy()
    diff_df["diff_ARIMA_minus_LR"] = diff_df["ARIMA_forecast"] - diff_df["LinearRegression_forecast"]

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=diff_df["year"], y=diff_df["diff_ARIMA_minus_LR"],
        name="ARIMA - Linear Regression"
    ))
    fig3.update_layout(
        title=f"Prediction Difference (ARIMA - Linear Regression): {sector_forecast}",
        xaxis_title="Year",
        yaxis_title="Difference in Predicted Allocation"
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.write("**Forecast comparison table:**")
    st.dataframe(fc_df, use_container_width=True)

    # Holdout evaluation
    if test is not None and len(test) >= 3:
        st.divider()
        st.subheader("Holdout Evaluation (Test = last N years)")

        X_test = test.index.values.reshape(-1, 1).astype(float)
        lr_pred = lr.predict(X_test)

        if arima_ok:
            arima_pred = arima_fitted.forecast(steps=len(test)).values
        else:
            arima_pred = np.array([np.nan] * len(test))

        eval_df = pd.DataFrame({
            "year": test.index.astype(int),
            "actual": test.values.astype(float),
            "ARIMA_pred": arima_pred.astype(float),
            "LR_pred": lr_pred.astype(float)
        })

        st.write("**Predictions on test years:**")
        st.dataframe(eval_df, use_container_width=True)

        def rmse(y_true, y_pred):
            return float(np.sqrt(mean_squared_error(y_true, y_pred)))

        lr_mae = mean_absolute_error(eval_df["actual"], eval_df["LR_pred"])
        lr_rmse = rmse(eval_df["actual"], eval_df["LR_pred"])

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("LR MAE", f"{lr_mae:.4f}")
            st.metric("LR RMSE", f"{lr_rmse:.4f}")

        with c2:
            if arima_ok and np.isfinite(eval_df["ARIMA_pred"]).all():
                arima_mae = mean_absolute_error(eval_df["actual"], eval_df["ARIMA_pred"])
                arima_rmse = rmse(eval_df["actual"], eval_df["ARIMA_pred"])
                st.metric("ARIMA MAE", f"{arima_mae:.4f}")
                st.metric("ARIMA RMSE", f"{arima_rmse:.4f}")
            else:
                st.info("ARIMA metrics unavailable (ARIMA failed).")

        with c3:
            if arima_ok and np.isfinite(eval_df["ARIMA_pred"]).all():
                winner = "ARIMA" if arima_rmse < lr_rmse else "Linear Regression"
                st.metric("Winner (lower RMSE)", winner)
            else:
                st.metric("Winner (lower RMSE)", "Linear Regression")

# =========================
# Footer
# =========================
with st.expander("Show raw data (first 50 rows)"):
    st.dataframe(df.head(50), use_container_width=True)

st.caption("Dashboard ready for CRISP-DM Data Understanding → Data Exploration deliverables.")
