import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# =========================
# Config
# =========================
st.set_page_config(page_title="Malaysia Federal Finance Allocation Dashboard", layout="wide")

# =========================
# Load data
# =========================
st.title("📊 Data Exploration Dashboard: Federal Finance Allocation (1970–2023)")

st.sidebar.header("1) Data Source")
use_uploaded_path = st.sidebar.checkbox(
    "Use fixed path (your uploaded file in this environment)", value=True
)

DEFAULT_PATH = r"C:\Users\User\OneDrive - National Defence University of Malaysia\Documents\BIG DATA\federal_finance_year_de (1).csv"

uploaded = None
if not use_uploaded_path:
    uploaded = st.sidebar.file_uploader("Upload your CSV (same format)", type=["csv"])

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

if use_uploaded_path:
    df = load_data(DEFAULT_PATH)
else:
    if uploaded is None:
        st.info("Upload your CSV to begin, or tick 'Use fixed path'.")
        st.stop()
    df = load_data(uploaded)

# =========================
# Helper: build sector view
# =========================
def build_wide(df, level="category", include_total=False):
    """
    Returns:
      wide_df: year x sector (values)
      total_series: year -> total allocation (from total/total if exists else sum of sectors)
    """
    d = df.copy()

    # Total row usually: category=total & function=total
    total_rows = d[(d["category"] == "total") & (d["function"] == "total")]
    if len(total_rows) > 0:
        total_series = total_rows.groupby("year")["value"].sum().sort_index()
    else:
        total_series = None

    if level == "category":
        # categories include: total, defence, economy, social, admin
        sector_col = "category"
        # if exclude total:
        if not include_total:
            d = d[d["category"] != "total"]
        # use rows where function is total (category total line), if present
        # BUT your data has category groups and function breakdown. For category totals:
        # best approximation is: rows where function == 'total' for each category,
        # if they exist. If not, sum functions within category.
        cat_tot = d[d["function"] == "total"]
        if len(cat_tot) > 0:
            g = cat_tot.groupby(["year", sector_col])["value"].sum().reset_index()
        else:
            g = d.groupby(["year", sector_col])["value"].sum().reset_index()

    elif level == "function":
        # function includes: education, health, housing, transport, etc + total
        sector_col = "function"
        if not include_total:
            d = d[d["function"] != "total"]
        g = d.groupby(["year", sector_col])["value"].sum().reset_index()

    else:
        raise ValueError("level must be 'category' or 'function'")

    wide = g.pivot(index="year", columns=sector_col, values="value").sort_index()

    # If total_series not found, compute from wide sum
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
    """
    Rolling z-score outlier detection.
    Returns a DataFrame with year, value, rolling_mean, rolling_std, zscore, is_outlier
    """
    s = series.dropna().copy()
    rm = s.rolling(window, min_periods=max(3, window//2)).mean()
    rs = s.rolling(window, min_periods=max(3, window//2)).std(ddof=0)
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
tab1, tab2, tab3, tab4 = st.tabs([
    "1) Summary Statistics",
    "2) Distribution + Outliers",
    "3) Correlations",
    "4) Trends + Prioritization + Benchmarks"
])

# =========================
# TAB 1: Summary Statistics
# =========================
with tab1:
    st.subheader("1. Summary Statistics")

    stats = summary_stats(wide_sel)

    colA, colB = st.columns([1.2, 0.8])

    with colA:
        st.write("**Descriptive statistics (by selected sectors):**")
        st.dataframe(stats, use_container_width=True)

    with colB:
        # largest/smallest by latest year
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
# TAB 2: Distribution + Outliers
# =========================
with tab2:
    st.subheader("2. Distribution Analysis")

    c1, c2 = st.columns(2)
    with c1:
        # histogram (long)
        long_df = wide_sel.reset_index().melt(id_vars="year", var_name="sector", value_name="allocation")
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
        # plot one sector with outliers
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
# TAB 3: Correlations
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
        if len(valid) >= 3:
            r = valid.iloc[:, 0].corr(valid.iloc[:, 1])
        else:
            r = np.nan
        lag_rows.append((s, float(r) if pd.notna(r) else np.nan))

    lag_df = pd.DataFrame(lag_rows, columns=["sector", "corr(current, previous_year)"]).sort_values(
        "corr(current, previous_year)", ascending=False
    )
    st.dataframe(lag_df, use_container_width=True)

    st.info(
        "Note: GDP growth correlation needs a GDP dataset (year + GDP growth). "
        "If you provide a GDP file, you can merge by year and compute Pearson correlation the same way."
    )

# =========================
# TAB 4: Trends + Prioritization + Benchmarks
# =========================
with tab4:
    st.subheader("4. Trend Analysis (1970–2023)")

    # Trend line
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

    st.write("**Top increases/decreases (latest year vs previous year):**")
    ly = wide_sel.index.max()
    if ly - 1 in wide_sel.index:
        delta = (wide_sel.loc[ly] - wide_sel.loc[ly - 1]).sort_values(ascending=False)
        pct_delta = (wide_sel.loc[ly] / wide_sel.loc[ly - 1] - 1) * 100
        show = pd.DataFrame({"change_abs": delta, "change_pct": pct_delta}).sort_values("change_abs", ascending=False)
        st.dataframe(show, use_container_width=True)
    else:
        st.warning("Not enough years selected to compute latest vs previous year changes.")

    st.divider()
    st.subheader("5. Sector Prioritization (Share of Total)")

    # share of total
    total_use = total_f.reindex(wide_f.index).loc[year_range[0]:year_range[1]]
    total_use = total_use.replace(0, np.nan)

    share = wide_sel.divide(total_use, axis=0) * 100
    long_share = share.reset_index().melt(id_vars="year", var_name="sector", value_name="share_pct")

    c5, c6 = st.columns(2)

    with c5:
        fig_share = px.area(long_share, x="year", y="share_pct", color="sector",
                            title="Share of Total Allocation Over Time (%)")
        st.plotly_chart(fig_share, use_container_width=True)

    with c6:
        year_for_pie = st.selectbox("Choose year for pie chart", sorted(wide_sel.index.tolist()), index=len(wide_sel.index)-1)
        pie_vals = share.loc[year_for_pie].dropna().sort_values(ascending=False)
        fig_pie = px.pie(values=pie_vals.values, names=pie_vals.index,
                         title=f"Sector Share of Total (%) in {year_for_pie}")
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()
    st.subheader("6. Regional Allocation (if data available)")
    if "state" in df.columns:
        st.success("Detected a 'state' column ✅ Showing state-level analysis.")
        reg = df.copy()
        reg = reg[(reg["year"].between(year_range[0], year_range[1]))]
        # choose sector field based on dashboard level
        if level == "category":
            reg_sector = st.selectbox("Choose category", sorted(reg["category"].unique()))
            reg = reg[reg["category"] == reg_sector]
        else:
            reg_sector = st.selectbox("Choose function", sorted(reg["function"].unique()))
            reg = reg[reg["function"] == reg_sector]

        reg_p = reg.groupby(["year", "state"])["value"].sum().reset_index()
        pivot_reg = reg_p.pivot(index="state", columns="year", values="value")
        st.write("**Heatmap (State x Year)**")
        fig_hm = px.imshow(pivot_reg, aspect="auto", title=f"State Allocation Heatmap ({reg_sector})")
        st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.info("No state-level column found in this dataset, so regional allocation analysis is not available.")

    st.divider()
    st.subheader("8. Comparative Analysis (ASEAN benchmarking)")

    st.write("If you have ASEAN data, upload a CSV with columns like:")
    st.code("country, year, sector, value", language="text")

    asean_file = st.file_uploader("Upload ASEAN benchmark CSV (optional)", type=["csv"], key="asean")
    if asean_file is not None:
        asean = pd.read_csv(asean_file)
        for c in ["country", "sector"]:
            if c in asean.columns:
                asean[c] = asean[c].astype(str).str.strip().str.lower()
        asean["year"] = pd.to_numeric(asean["year"], errors="coerce").astype("Int64")
        asean["value"] = pd.to_numeric(asean["value"], errors="coerce")
        asean = asean.dropna(subset=["year", "value", "country", "sector"])

        sector_pick = st.selectbox("Choose sector for ASEAN comparison", sorted(asean["sector"].unique()))
        years_common = sorted(set(asean["year"].dropna().astype(int).unique()) & set(wide_sel.index))
        if len(years_common) == 0:
            st.warning("No overlapping years between your dataset and ASEAN file.")
        else:
            yr_pick = st.selectbox("Choose year (overlapping)", years_common, index=len(years_common)-1)

            # Malaysia value from our dataset for selected sector (if exists)
            my_val = None
            if sector_pick in wide.columns:
                if yr_pick in wide.index:
                    my_val = wide.loc[yr_pick, sector_pick]

            comp = asean[(asean["sector"] == sector_pick) & (asean["year"] == yr_pick)].copy()
            comp = comp.sort_values("value", ascending=False)

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
        st.info("No ASEAN file uploaded. Benchmarking section will remain inactive.")

# =========================
# Footer: show raw data
# =========================
with st.expander("Show raw data (first 50 rows)"):
    st.dataframe(df.head(50), use_container_width=True)

st.caption("Dashboard ready for CRISP-DM Data Understanding → Data Exploration deliverables.")
