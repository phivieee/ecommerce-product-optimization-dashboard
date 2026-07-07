from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ============================================================
# APP CONFIG
# ============================================================

st.set_page_config(
    page_title="E-Commerce Product Optimization",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "eda_output"

# ============================================================
# THEME CSS
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #111827;
    }

    .stApp {
        background: #EEF3FF;
    }

    section[data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid rgba(17, 24, 39, 0.08);
    }

    section[data-testid="stSidebar"] * {
        color: #111827 !important;
    }

    section[data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: #FFFFFF !important;
        color: #111827 !important;
        border-radius: 14px;
        border: 1px solid rgba(15, 23, 42, 0.14);
        box-shadow: 0 6px 14px rgba(15, 23, 42, 0.05);
    }

    section[data-testid="stSidebar"] [data-baseweb="select"] *,
    section[data-testid="stSidebar"] [data-baseweb="select"] span,
    section[data-testid="stSidebar"] [data-baseweb="select"] div,
    section[data-testid="stSidebar"] [data-baseweb="select"] input,
    section[data-testid="stSidebar"] [data-baseweb="select"] svg {
        color: #111827 !important;
        fill: #111827 !important;
        opacity: 1 !important;
    }

    .main .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1280px;
    }

    .dashboard-title {
        font-size: 32px;
        font-weight: 800;
        color: #111827;
        margin-bottom: 0.1rem;
    }

    .dashboard-subtitle {
        font-size: 14px;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }

    .kpi-card {
        padding: 20px 18px;
        border-radius: 22px;
        min-height: 118px;
        box-shadow: 0 14px 30px rgba(15, 23, 42, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.70);
    }

    .kpi-label {
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 22px;
        opacity: 0.92;
    }

    .kpi-value {
        font-size: 26px;
        font-weight: 800;
        line-height: 1;
    }

    .kpi-caption {
        font-size: 12px;
        opacity: 0.82;
        margin-top: 8px;
    }

    .card-blue {
        background: linear-gradient(135deg, #0E46A3 0%, #123C91 100%);
        color: white;
    }

    .card-cyan {
        background: #D7F2FF;
        color: #111827;
    }

    .card-green {
        background: #DDF6D9;
        color: #111827;
    }

    .card-pink {
        background: #FFE1E4;
        color: #111827;
    }

    .card-lavender {
        background: #E1E7FF;
        color: #111827;
    }

    .white-panel {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 24px;
        padding: 20px 20px 14px 20px;
        box-shadow: 0 16px 32px rgba(15, 23, 42, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.82);
        margin-bottom: 18px;
    }

    .panel-title {
        font-size: 19px;
        font-weight: 800;
        color: #111827;
        margin-bottom: 6px;
    }

    .panel-subtitle {
        font-size: 12.5px;
        color: #4B5563;
        margin-bottom: 12px;
    }

    .insight-box {
        background: #FFFFFF;
        border-left: 7px solid #0E46A3;
        border-radius: 20px;
        padding: 18px 20px;
        box-shadow: 0 16px 32px rgba(15, 23, 42, 0.08);
        color: #111827;
        margin-top: 5px;
    }

    .insight-title {
        font-size: 17px;
        font-weight: 800;
        margin-bottom: 8px;
        color: #111827;
    }

    .insight-text {
        font-size: 14px;
        line-height: 1.65;
        color: #374151;
    }

    .custom-table-wrap {
        background: #FFFFFF;
        border-radius: 22px;
        padding: 10px 10px 2px 10px;
        box-shadow: 0 16px 32px rgba(15, 23, 42, 0.08);
        overflow-x: auto;
        margin-top: 8px;
    }

    .custom-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 13px;
        color: #111827;
    }

    .custom-table thead th {
        background: #0F172A;
        color: #FFFFFF !important;
        text-align: left;
        padding: 12px 12px;
        font-weight: 700;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        position: sticky;
        top: 0;
        z-index: 1;
    }

    .custom-table thead th:first-child {
        border-top-left-radius: 16px;
    }

    .custom-table thead th:last-child {
        border-top-right-radius: 16px;
    }

    .custom-table tbody td {
        padding: 11px 12px;
        border-bottom: 1px solid #E5E7EB;
        vertical-align: top;
        color: #111827 !important;
    }

    .custom-table tbody tr:nth-child(even) td {
        background: #F8FAFC;
    }

    .custom-table tbody tr:hover td {
        background: #EEF4FF;
    }

    .custom-table .rank-pill {
        display: inline-block;
        background: #E8EEFF;
        color: #0E46A3;
        border-radius: 999px;
        padding: 4px 9px;
        font-weight: 700;
        font-size: 12px;
    }

    .custom-table .metric-badge {
        display: inline-block;
        border-radius: 999px;
        padding: 3px 8px;
        font-size: 12px;
        font-weight: 700;
        color: #111827;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: #E8EEFF;
        border-radius: 999px;
        padding: 8px 16px;
        color: #111827 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HELPERS
# ============================================================

@st.cache_data(show_spinner=False)
def load_csv(file_name: str) -> pd.DataFrame:
    path = DATA_DIR / file_name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def format_number(value: float | int | str, prefix: str = "", suffix: str = "") -> str:
    try:
        value = float(value)
    except Exception:
        return f"{prefix}{value}{suffix}"

    if abs(value) >= 1_000_000:
        return f"{prefix}{value/1_000_000:.2f}M{suffix}"
    if abs(value) >= 1_000:
        return f"{prefix}{value:,.0f}{suffix}"
    if value % 1 == 0:
        return f"{prefix}{value:,.0f}{suffix}"
    return f"{prefix}{value:,.2f}{suffix}"


def kpi_card(label: str, value: str, caption: str = "", color_class: str = "card-blue") -> None:
    st.markdown(
        f"""
        <div class="kpi-card {color_class}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def panel_header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="panel-title">{title}</div>
        <div class="panel-subtitle">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


def shorten_labels(df: pd.DataFrame, col: str, n: int = 34) -> pd.Series:
    return df[col].astype(str).apply(lambda x: x if len(x) <= n else x[: n - 3] + "...")


def apply_filters(df: pd.DataFrame, category: str, brand: str) -> pd.DataFrame:
    filtered = df.copy()
    if category != "All" and "category" in filtered.columns:
        filtered = filtered[filtered["category"] == category]
    if brand != "All" and "brand" in filtered.columns:
        filtered = filtered[filtered["brand"] == brand]
    return filtered


def insight_box(title: str, html_text: str) -> None:
    st.markdown(
        f"""
        <div class="insight-box">
            <div class="insight-title">{title}</div>
            <div class="insight-text">{html_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plotly_layout(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=60, b=20),
        font=dict(family="Inter, sans-serif", color="#111827", size=13),
        title_font=dict(family="Inter, sans-serif", size=20, color="#111827"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.26,
            xanchor="left",
            x=0,
            font=dict(color="#374151", size=12),
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="rgba(15,23,42,0.08)",
            borderwidth=1,
        ),
        hoverlabel=dict(bgcolor="#0E46A3", font_color="white", bordercolor="#0E46A3"),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(17,24,39,0.08)",
        title_font=dict(color="#374151", size=13),
        tickfont=dict(color="#4B5563", size=12),
        zeroline=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(17,24,39,0.08)",
        title_font=dict(color="#374151", size=13),
        tickfont=dict(color="#4B5563", size=12),
        zeroline=False,
    )
    return fig


def combo_bar_line(df: pd.DataFrame, sort_col: str, rate_col: str, label_col: str, title: str,
                   bar_name: str, line_name: str, top_n: int = 15,
                   bar_color: str = "#9EC5FE", line_color: str = "#0E46A3") -> go.Figure:
    plot_df = df.dropna(subset=[sort_col, rate_col]).copy()
    plot_df = plot_df.sort_values(sort_col, ascending=False).head(top_n).copy()
    plot_df["label"] = shorten_labels(plot_df, label_col, n=22)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=plot_df["label"],
            y=plot_df[sort_col],
            name=bar_name,
            marker_color=bar_color,
            opacity=0.88,
            hovertemplate=f"{bar_name}: %{{y:,.0f}}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["label"],
            y=plot_df[rate_col],
            name=line_name,
            mode="lines+markers+text",
            text=[f"{v:.1f}%" for v in plot_df[rate_col]],
            textposition="top center",
            line=dict(color=line_color, width=3),
            marker=dict(size=8, color=line_color),
            hovertemplate=f"{line_name}: %{{y:.2f}}%<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_yaxes(title_text=bar_name, secondary_y=False)
    fig.update_yaxes(title_text=line_name, secondary_y=True)
    fig.update_layout(title=title, xaxis_title="Product", xaxis_tickangle=-35)
    return plotly_layout(fig, 430)


def performance_quadrant(df: pd.DataFrame, x_col: str, y_col: str, size_col: str, label_col: str,
                         top_n: int, title: str, subtitle_mode: str = "views") -> go.Figure:
    plot_df = df.dropna(subset=[x_col, y_col]).copy()
    plot_df = plot_df[plot_df[x_col] > 0]
    plot_df = plot_df.sort_values(x_col, ascending=False).head(top_n).copy()

    x_med = plot_df[x_col].median()
    y_med = plot_df[y_col].median()

    conditions = [
        (plot_df[x_col] >= x_med) & (plot_df[y_col] >= y_med),
        (plot_df[x_col] >= x_med) & (plot_df[y_col] < y_med),
        (plot_df[x_col] < x_med) & (plot_df[y_col] >= y_med),
        (plot_df[x_col] < x_med) & (plot_df[y_col] < y_med),
    ]
    choices = [
        "High Exposure, High Efficiency",
        "High Exposure, Low Efficiency",
        "Low Exposure, High Efficiency",
        "Low Exposure, Low Efficiency",
    ]
    plot_df["quadrant"] = np.select(conditions, choices, default="Other")
    plot_df["short_name"] = shorten_labels(plot_df, label_col, n=24)

    color_map = {
        "High Exposure, High Efficiency": "#0E46A3",
        "High Exposure, Low Efficiency": "#FF7B7B",
        "Low Exposure, High Efficiency": "#59C3C3",
        "Low Exposure, Low Efficiency": "#A5B4C7",
    }

    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        size=size_col,
        color="quadrant",
        hover_name=label_col,
        hover_data={
            "category": True,
            "brand": True,
            x_col: ":,.0f",
            y_col: ":.2f",
            size_col: ":,.2f",
            "quadrant": True,
        },
        color_discrete_map=color_map,
        size_max=34,
        title=title,
    )

    highlight = plot_df.sort_values([x_col, size_col], ascending=[False, False]).head(8)
    fig.add_trace(
        go.Scatter(
            x=highlight[x_col],
            y=highlight[y_col],
            mode="text",
            text=highlight["short_name"],
            textposition="top center",
            textfont=dict(color="#111827", size=11),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.add_vline(x=x_med, line_dash="dash", line_color="#6B7280", opacity=0.8)
    fig.add_hline(y=y_med, line_dash="dash", line_color="#6B7280", opacity=0.8)

    x_label = "Median Views" if subtitle_mode == "views" else "Median Clicks"
    y_label = "Median CTR" if "ctr" in y_col else "Median Purchase Rate"
    fig.add_annotation(x=x_med, y=plot_df[y_col].max(), text=x_label, showarrow=False, xanchor="left",
                       font=dict(size=11, color="#4B5563"), bgcolor="rgba(255,255,255,0.8)")
    fig.add_annotation(x=plot_df[x_col].max(), y=y_med, text=y_label, showarrow=False, yanchor="bottom",
                       font=dict(size=11, color="#4B5563"), bgcolor="rgba(255,255,255,0.8)")

    fig.update_traces(marker=dict(line=dict(width=1, color="rgba(255,255,255,0.85)")))
    return plotly_layout(fig, 420)


def pretty_table(df: pd.DataFrame, columns: list[str], rename_map: dict[str, str] | None = None,
                 percentage_cols: list[str] | None = None, currency_cols: list[str] | None = None,
                 numeric_cols: list[str] | None = None, max_rows: int = 20) -> None:
    if df.empty:
        st.info("Data tidak tersedia untuk ditampilkan.")
        return

    rename_map = rename_map or {}
    percentage_cols = percentage_cols or []
    currency_cols = currency_cols or []
    numeric_cols = numeric_cols or []

    show_df = df[columns].head(max_rows).copy()
    show_df.insert(0, "rank", range(1, len(show_df) + 1))
    show_df = show_df.rename(columns=rename_map)

    html = ['<div class="custom-table-wrap"><table class="custom-table"><thead><tr>']
    for col in show_df.columns:
        header = "#" if col == "rank" else col.replace("_", " ").title()
        html.append(f"<th>{header}</th>")
    html.append("</tr></thead><tbody>")

    for _, row in show_df.iterrows():
        html.append("<tr>")
        for col in show_df.columns:
            raw_val = row[col]
            original_col = col
            display = raw_val
            if original_col == "rank":
                display = f'<span class="rank-pill">{int(raw_val)}</span>'
            else:
                src_name = [k for k, v in rename_map.items() if v == col]
                source_col = src_name[0] if src_name else col
                if source_col in percentage_cols and pd.notna(raw_val):
                    display = f'<span class="metric-badge" style="background:#E8EEFF;">{float(raw_val):.2f}%</span>'
                elif source_col in currency_cols and pd.notna(raw_val):
                    display = f"${float(raw_val):,.2f}"
                elif source_col in numeric_cols and pd.notna(raw_val):
                    display = f"{float(raw_val):,.0f}"
                elif isinstance(raw_val, float) and pd.notna(raw_val):
                    display = f"{raw_val:,.2f}"
                elif pd.isna(raw_val):
                    display = "-"
            html.append(f"<td>{display}</td>")
        html.append("</tr>")

    html.append("</tbody></table></div>")
    st.markdown("".join(html), unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================

summary = load_csv("02_python_summary.csv")
funnel = load_csv("17_funnel_summary.csv")
monthly = load_csv("14_monthly_trend.csv")
product_full = load_csv("13_product_performance_full.csv")
top_revenue = load_csv("03_top_products_by_revenue.csv")
top_ctr = load_csv("04_top_products_by_ctr.csv")
top_purchase = load_csv("15_top_products_by_purchases.csv")
high_view_low_ctr = load_csv("05_high_view_low_ctr_products.csv")
high_click_low_purchase = load_csv("16_high_click_low_purchase_products.csv")
category_insight = load_csv("07_category_insight.csv")
brand_insight = load_csv("08_brand_insight.csv")
channel_device = load_csv("09_channel_device_insight.csv")
review_summary = load_csv("11_review_summary_by_rating_segment.csv")
recommendation = load_csv("12_final_recommendation_table.csv")

# Build complete category and brand views from full product performance when available.
if not product_full.empty:
    category_full = (
        product_full.groupby("category", as_index=False)
        .agg(
            total_products=("product_id", "count"),
            total_views=("total_views", "sum"),
            total_clicks=("total_clicks", "sum"),
            total_add_to_cart=("total_add_to_cart", "sum"),
            total_purchase_rows=("total_purchase_rows", "sum"),
            total_revenue=("total_revenue", "sum"),
            avg_price=("price", "mean"),
            avg_rating=("rating_avg_clean", "mean"),
        )
        .round(2)
    )
    category_full["ctr_percent"] = (
        category_full["total_clicks"] / category_full["total_views"].replace(0, np.nan) * 100
    ).round(2)
    category_full["click_to_purchase_rate_percent"] = (
        category_full["total_purchase_rows"] / category_full["total_clicks"].replace(0, np.nan) * 100
    ).round(2)
    category_full["revenue_per_view"] = (
        category_full["total_revenue"] / category_full["total_views"].replace(0, np.nan)
    ).round(2)

    brand_full = (
        product_full.groupby(["brand", "category"], as_index=False)
        .agg(
            total_products=("product_id", "count"),
            total_views=("total_views", "sum"),
            total_clicks=("total_clicks", "sum"),
            total_purchase_rows=("total_purchase_rows", "sum"),
            total_revenue=("total_revenue", "sum"),
            avg_price=("price", "mean"),
            avg_rating=("rating_avg_clean", "mean"),
        )
        .round(2)
    )
    brand_full["ctr_percent"] = (
        brand_full["total_clicks"] / brand_full["total_views"].replace(0, np.nan) * 100
    ).round(2)
    brand_full["click_to_purchase_rate_percent"] = (
        brand_full["total_purchase_rows"] / brand_full["total_clicks"].replace(0, np.nan) * 100
    ).round(2)
else:
    category_full = category_insight.copy()
    brand_full = brand_insight.copy()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("### 📦 Dashboard")
st.sidebar.markdown("**E-Commerce Product Engagement & Conversion Optimization**")

page = st.sidebar.radio(
    "Pilih Halaman",
    [
        "1. Executive Overview",
        "2. Product Performance",
        "3. Product Optimization Opportunity",
        "4. Category & Brand Performance",
    ],
)

st.sidebar.divider()

category_options = ["All"]
if not product_full.empty and "category" in product_full.columns:
    category_options += sorted(product_full["category"].dropna().unique().tolist())
selected_category = st.sidebar.selectbox("Filter Category", category_options)

brand_options = ["All"]
if not product_full.empty:
    brand_base = product_full.copy()
    if selected_category != "All":
        brand_base = brand_base[brand_base["category"] == selected_category]
    brand_options += sorted(brand_base["brand"].dropna().unique().tolist())
selected_brand = st.sidebar.selectbox("Filter Brand", brand_options)

st.sidebar.caption("Dashboard membaca CSV hasil SQL + EDA dari folder data/eda_output.")

# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="dashboard-title">E-Commerce Product Engagement & Conversion Optimization</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="dashboard-subtitle">Dashboard interaktif untuk mengevaluasi funnel produk, performa produk, peluang optimasi, serta kategori dan brand terbaik.</div>',
    unsafe_allow_html=True,
)

# ============================================================
# PAGE 1: EXECUTIVE OVERVIEW
# ============================================================

if page == "1. Executive Overview":
    if summary.empty:
        st.error("File 02_python_summary.csv tidak ditemukan.")
        st.stop()

    s = summary.iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi_card("Total Products", format_number(s.get("total_products", 0)), "Produk dianalisis", "card-blue")
    with c2:
        kpi_card("Total Views", format_number(s.get("total_views", 0)), "Product views", "card-cyan")
    with c3:
        kpi_card("Total Clicks", format_number(s.get("total_clicks", 0)), f"CTR {s.get('overall_ctr_percent', 0):.2f}%", "card-green")
    with c4:
        kpi_card("Purchases", format_number(s.get("total_purchase_rows", 0)), f"Click → Purchase {s.get('overall_click_to_purchase_rate_percent', 0):.2f}%", "card-pink")
    with c5:
        kpi_card("Revenue", format_number(s.get("total_revenue", 0), prefix="$"), "Total sales value", "card-lavender")

    left, right = st.columns([1.05, 1.25])

    with left:
        st.markdown('<div class="white-panel">', unsafe_allow_html=True)
        panel_header("Funnel Chart", "View → Click → Add to Cart → Purchase")
        if not funnel.empty:
            funnel_sorted = funnel.sort_values("sort_order")
            fig = px.funnel(
                funnel_sorted,
                x="total_events",
                y="funnel_stage",
                title="Product Interaction Funnel",
                text="total_events",
            )
            fig.update_traces(
                texttemplate="%{text:,.0f}",
                textfont=dict(color="#111827", size=15),
                marker_color=["#0E46A3", "#4EADEB", "#A6E89A", "#FF9AA2"],
            )
            fig = plotly_layout(fig, 410)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Funnel data belum tersedia.")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="white-panel">', unsafe_allow_html=True)
        panel_header("Monthly Trend", "Views, clicks, purchases, dan revenue per bulan")
        if not monthly.empty:
            monthly_plot = monthly.copy()
            monthly_plot["month"] = pd.to_datetime(monthly_plot["month"], errors="coerce")
            monthly_plot = monthly_plot.sort_values("month")

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=monthly_plot["month"], y=monthly_plot["total_views"], mode="lines+markers", name="Views", line=dict(color="#0E46A3", width=3)), secondary_y=False)
            fig.add_trace(go.Scatter(x=monthly_plot["month"], y=monthly_plot["total_clicks"], mode="lines+markers", name="Clicks", line=dict(color="#4EADEB", width=3)), secondary_y=False)
            fig.add_trace(go.Scatter(x=monthly_plot["month"], y=monthly_plot["total_purchase_rows"], mode="lines+markers", name="Purchases", line=dict(color="#7BC97B", width=3)), secondary_y=False)
            fig.add_trace(go.Scatter(x=monthly_plot["month"], y=monthly_plot["total_revenue"], mode="lines+markers", name="Revenue", line=dict(color="#F97373", width=3, dash="dot")), secondary_y=True)
            fig.update_yaxes(title_text="Events", secondary_y=False)
            fig.update_yaxes(title_text="Revenue", secondary_y=True)
            fig.update_layout(title="Monthly Product Engagement and Revenue")
            fig = plotly_layout(fig, 410)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Monthly trend belum tersedia. Tambahkan file 14_monthly_trend.csv.")
        st.markdown('</div>', unsafe_allow_html=True)

    if not category_full.empty and not recommendation.empty:
        top_rev_cat = category_full.sort_values("total_revenue", ascending=False).iloc[0]
        top_ctr_cat = category_full.sort_values("ctr_percent", ascending=False).iloc[0]
        total_need_opt = len(recommendation)
        insight = (
            f"Secara keseluruhan, dashboard mencatat <b>{format_number(s.get('total_views', 0))}</b> views dan "
            f"<b>{format_number(s.get('total_clicks', 0))}</b> clicks, dengan CTR sebesar "
            f"<b>{s.get('overall_ctr_percent', 0):.2f}%</b>. Funnel menunjukkan penurunan paling besar terjadi dari "
            f"<b>Click ke Purchase</b>, sehingga tahap konversi akhir masih menjadi fokus utama. "
            f"Kategori dengan revenue terbesar adalah <b>{top_rev_cat['category']}</b> dengan revenue "
            f"<b>{format_number(top_rev_cat['total_revenue'], prefix='$')}</b>, sedangkan kategori dengan CTR tertinggi adalah "
            f"<b>{top_ctr_cat['category']}</b> dengan CTR <b>{top_ctr_cat['ctr_percent']:.2f}%</b>. "
            f"Terdapat <b>{format_number(total_need_opt)}</b> produk yang masuk daftar rekomendasi optimasi, "
            f"sehingga prioritas berikutnya adalah memperbaiki produk dengan exposure tinggi namun conversion belum optimal."
        )
        insight_box("Summary Insight", insight)

# ============================================================
# PAGE 2: PRODUCT PERFORMANCE
# ============================================================

elif page == "2. Product Performance":
    if product_full.empty:
        st.error("File 13_product_performance_full.csv tidak ditemukan.")
        st.stop()

    df = apply_filters(product_full, selected_category, selected_brand)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Products", format_number(len(df)), "Filtered products", "card-blue")
    with c2:
        kpi_card("Revenue", format_number(df["total_revenue"].sum(), prefix="$"), "Filtered revenue", "card-cyan")
    with c3:
        ctr = df["total_clicks"].sum() / df["total_views"].replace(0, np.nan).sum() * 100
        kpi_card("CTR", f"{ctr:.2f}%", "Clicks / Views", "card-green")
    with c4:
        pr = df["total_purchase_rows"].sum() / df["total_clicks"].replace(0, np.nan).sum() * 100
        kpi_card("Purchase Rate", f"{pr:.2f}%", "Purchases / Clicks", "card-pink")

    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.markdown('<div class="white-panel">', unsafe_allow_html=True)
        panel_header("Top 10 Products by CTR", "Minimum 30 views agar CTR tidak bias")
        plot_df = df[df["total_views"] >= 30].sort_values("ctr_percent", ascending=False).head(10).copy()
        plot_df["product_label"] = shorten_labels(plot_df, "product_name")
        fig = px.bar(plot_df, x="ctr_percent", y="product_label", orientation="h", color="ctr_percent", color_continuous_scale="Blues", title="Highest Product CTR", text="ctr_percent")
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        fig = plotly_layout(fig, 390)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with row1_col2:
        st.markdown('<div class="white-panel">', unsafe_allow_html=True)
        panel_header("Top 10 Products by Revenue", "Produk dengan nilai transaksi terbesar")
        plot_df = df.sort_values("total_revenue", ascending=False).head(10).copy()
        plot_df["product_label"] = shorten_labels(plot_df, "product_name")
        fig = px.bar(plot_df, x="total_revenue", y="product_label", orientation="h", color="total_revenue", color_continuous_scale="Blues", title="Highest Revenue Products", text="total_revenue")
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        fig = plotly_layout(fig, 390)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        st.markdown('<div class="white-panel">', unsafe_allow_html=True)
        panel_header("Top 10 Products by Purchases", "Produk dengan jumlah pembelian terbanyak")
        plot_df = df.sort_values("total_purchase_rows", ascending=False).head(10).copy()
        plot_df["product_label"] = shorten_labels(plot_df, "product_name")
        fig = px.bar(plot_df, x="total_purchase_rows", y="product_label", orientation="h", color="total_purchase_rows", color_continuous_scale="Greens", title="Highest Purchased Products", text="total_purchase_rows")
        fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        fig = plotly_layout(fig, 390)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with row2_col2:
        st.markdown('<div class="white-panel">', unsafe_allow_html=True)
        panel_header("Views and CTR by Product", "Top products by views dengan CTR sebagai garis")
        fig = combo_bar_line(
            df,
            sort_col="total_views",
            rate_col="ctr_percent",
            label_col="product_name",
            title="Top Products by Views with CTR Overlay",
            bar_name="Views",
            line_name="CTR (%)",
            top_n=15,
            bar_color="#B8D9FF",
            line_color="#0E46A3",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="white-panel">', unsafe_allow_html=True)
    panel_header("Clicks and Purchase Rate by Product", "Top products by clicks dengan purchase rate sebagai garis")
    fig = combo_bar_line(
        df,
        sort_col="total_clicks",
        rate_col="click_to_purchase_rate_percent",
        label_col="product_name",
        title="Top Products by Clicks with Purchase Rate Overlay",
        bar_name="Clicks",
        line_name="Purchase Rate (%)",
        top_n=15,
        bar_color="#CDEFD2",
        line_color="#2E8B57",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# PAGE 3: PRODUCT OPTIMIZATION OPPORTUNITY
# ============================================================

elif page == "3. Product Optimization Opportunity":
    if product_full.empty:
        st.error("File 13_product_performance_full.csv tidak ditemukan.")
        st.stop()

    df = apply_filters(product_full, selected_category, selected_brand)

    hv = df[
        (df["total_views"] >= df["total_views"].quantile(0.75))
        & (df["ctr_percent"] < df["ctr_percent"].median())
    ].sort_values(["total_views", "ctr_percent"], ascending=[False, True])

    hc = df[
        (df["total_clicks"] >= 20)
        & (df["click_to_purchase_rate_percent"].fillna(0) <= 5)
    ].sort_values(["total_clicks", "click_to_purchase_rate_percent"], ascending=[False, True])

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("High View, Low CTR", format_number(len(hv)), "Exposure tinggi, klik relatif rendah", "card-blue")
    with c2:
        kpi_card("High Click, Low Purchase", format_number(len(hc)), "Klik tinggi, pembelian rendah", "card-pink")
    with c3:
        kpi_card("Recommendation Rows", format_number(len(recommendation)), "Daftar rekomendasi SQL/EDA", "card-lavender")

    tab1, tab2 = st.tabs(["High View, Low CTR", "High Click, Low Purchase"])

    with tab1:
        st.markdown('<div class="white-panel">', unsafe_allow_html=True)
        panel_header("High View, Low CTR", "Produk sering dilihat tetapi kurang mendorong klik")
        plot_df = hv.head(12).copy()
        plot_df["product_label"] = shorten_labels(plot_df, "product_name")
        fig = px.bar(plot_df, x="total_views", y="product_label", orientation="h", color="ctr_percent", color_continuous_scale="Reds", title="High Exposure Products with Lower CTR", text="total_views")
        fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        fig = plotly_layout(fig, 430)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        pretty_table(
            hv,
            columns=["product_name", "category", "brand", "price", "rating_avg_clean", "total_views", "total_clicks", "ctr_percent", "total_revenue"],
            rename_map={
                "product_name": "Product Name",
                "rating_avg_clean": "Rating",
                "total_views": "Views",
                "total_clicks": "Clicks",
                "ctr_percent": "CTR",
                "total_revenue": "Revenue",
            },
            percentage_cols=["ctr_percent"],
            currency_cols=["price", "total_revenue"],
            numeric_cols=["total_views", "total_clicks"],
            max_rows=10,
        )

    with tab2:
        st.markdown('<div class="white-panel">', unsafe_allow_html=True)
        panel_header("High Click, Low Purchase", "Produk diklik cukup tinggi tetapi pembelian rendah")
        if not hc.empty:
            plot_df = hc.head(12).copy()
            plot_df["product_label"] = shorten_labels(plot_df, "product_name")
            fig = px.bar(plot_df, x="total_clicks", y="product_label", orientation="h", color="click_to_purchase_rate_percent", color_continuous_scale="OrRd_r", title="High Click Products with Low Purchase Rate", text="total_clicks")
            fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
            fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
            fig = plotly_layout(fig, 430)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Tidak ada produk yang memenuhi kriteria high click dan purchase rate <= 5% pada filter ini.")
        st.markdown('</div>', unsafe_allow_html=True)
        if not hc.empty:
            pretty_table(
                hc,
                columns=["product_name", "category", "brand", "price", "rating_avg_clean", "total_clicks", "total_purchase_rows", "click_to_purchase_rate_percent", "total_revenue"],
                rename_map={
                    "product_name": "Product Name",
                    "rating_avg_clean": "Rating",
                    "total_clicks": "Clicks",
                    "total_purchase_rows": "Purchases",
                    "click_to_purchase_rate_percent": "Purchase Rate",
                    "total_revenue": "Revenue",
                },
                percentage_cols=["click_to_purchase_rate_percent"],
                currency_cols=["price", "total_revenue"],
                numeric_cols=["total_clicks", "total_purchase_rows"],
                max_rows=10,
            )

# ============================================================
# PAGE 4: CATEGORY & BRAND PERFORMANCE
# ============================================================

elif page == "4. Category & Brand Performance":
    if category_full.empty or brand_full.empty:
        st.error("Data category/brand tidak ditemukan.")
        st.stop()

    cat = category_full.copy()

    brand = brand_full.copy()
    if selected_category != "All":
        brand = brand[brand["category"] == selected_category]
    if selected_brand != "All":
        brand = brand[brand["brand"] == selected_brand]

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Categories", format_number(cat["category"].nunique()), "Active categories", "card-blue")
    with c2:
        kpi_card("Category Revenue", format_number(cat["total_revenue"].sum(), prefix="$"), "Filtered revenue", "card-cyan")
    with c3:
        cat_ctr = cat["total_clicks"].sum() / cat["total_views"].replace(0, np.nan).sum() * 100
        kpi_card("Category CTR", f"{cat_ctr:.2f}%", "Clicks / Views", "card-green")

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.markdown('<div class="white-panel">', unsafe_allow_html=True)
        panel_header("CTR by Category", "Kategori dengan daya tarik klik paling tinggi")
        plot_df = cat.sort_values("ctr_percent", ascending=False)
        fig = px.bar(plot_df, x="category", y="ctr_percent", color="ctr_percent", color_continuous_scale="Blues", title="CTR by Category", text="ctr_percent")
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(coloraxis_showscale=False)
        fig = plotly_layout(fig, 390)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r1c2:
        st.markdown('<div class="white-panel">', unsafe_allow_html=True)
        panel_header("Revenue by Category", "Kategori dengan kontribusi revenue terbesar")
        plot_df = cat.sort_values("total_revenue", ascending=False)
        fig = px.bar(plot_df, x="category", y="total_revenue", color="total_revenue", color_continuous_scale="Blues", title="Revenue by Category", text="total_revenue")
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig.update_layout(coloraxis_showscale=False)
        fig = plotly_layout(fig, 390)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.markdown('<div class="white-panel">', unsafe_allow_html=True)
        panel_header("Purchase Rate by Category", "Purchase rows dibanding total clicks")
        plot_df = cat.sort_values("click_to_purchase_rate_percent", ascending=False)
        fig = px.bar(plot_df, x="category", y="click_to_purchase_rate_percent", color="click_to_purchase_rate_percent", color_continuous_scale="Greens", title="Purchase Rate by Category", text="click_to_purchase_rate_percent")
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(coloraxis_showscale=False)
        fig = plotly_layout(fig, 390)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r2c2:
        st.markdown('<div class="white-panel">', unsafe_allow_html=True)
        panel_header("Top Brands by Revenue", "Brand dengan revenue terbesar")
        plot_df = brand.sort_values("total_revenue", ascending=False).head(10).copy()
        plot_df["brand_label"] = shorten_labels(plot_df, "brand", n=24)
        fig = px.bar(plot_df, x="total_revenue", y="brand_label", orientation="h", color="category", title="Top Brands by Revenue", text="total_revenue")
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig.update_layout(yaxis=dict(autorange="reversed"))
        fig = plotly_layout(fig, 390)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="white-panel">', unsafe_allow_html=True)
    panel_header("Top Brands by CTR", "")
    plot_df = brand[brand["total_views"] >= 100].sort_values("ctr_percent", ascending=False).head(12).copy()
    plot_df["brand_category"] = plot_df["brand"].astype(str) + " | " + plot_df["category"].astype(str)
    fig = px.bar(plot_df, x="ctr_percent", y="brand_category", orientation="h", color="ctr_percent", color_continuous_scale="Blues", title="Top Brands by CTR", text="ctr_percent")
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
    fig = plotly_layout(fig, 430)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

