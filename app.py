from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="E-Commerce Product Optimization",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "eda_output"

CHART_CONFIG = {"displayModeBar": False, "responsive": True}

# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #0f172a;
    }

    .stApp {
        background: #EEF3FF;
    }

    .main .block-container {
        padding-top: 1.3rem;
        padding-bottom: 2rem;
        max-width: 1320px;
    }

    section[data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid rgba(15, 23, 42, 0.08);
    }

    section[data-testid="stSidebar"] * {
        color: #0f172a !important;
    }

    .dashboard-title {
        font-size: 32px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 2px;
    }

    .dashboard-subtitle {
        font-size: 14px;
        color: #475569;
        margin-bottom: 18px;
    }

    .kpi-card {
        padding: 18px 18px;
        border-radius: 22px;
        min-height: 112px;
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
        border: 1px solid rgba(255,255,255,0.9);
    }

    .kpi-label {
        font-size: 13px;
        font-weight: 700;
        margin-bottom: 20px;
        opacity: 0.92;
    }

    .kpi-value {
        font-size: 27px;
        font-weight: 800;
        line-height: 1;
    }

    .kpi-caption {
        font-size: 12px;
        margin-top: 8px;
        opacity: 0.82;
    }

    .card-blue {background: linear-gradient(135deg, #0E46A3 0%, #123C91 100%); color: #fff;}
    .card-cyan {background: #D7F2FF; color: #0f172a;}
    .card-green {background: #DDF6D9; color: #0f172a;}
    .card-pink {background: #FFE1E4; color: #0f172a;}
    .card-lavender {background: #E1E7FF; color: #0f172a;}

    .chart-panel {
        background: rgba(255,255,255,0.46);
        border-radius: 24px;
        padding: 20px 20px 12px 20px;
        box-shadow: 0 15px 32px rgba(15, 23, 42, 0.06);
        border: 1px solid rgba(255,255,255,0.72);
        margin-bottom: 18px;
    }

    .panel-title {
        font-size: 19px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 6px;
    }

    .panel-subtitle {
        font-size: 12.5px;
        color: #475569;
        margin-bottom: 10px;
    }

    .insight-box {
        background: rgba(255,255,255,0.78);
        border-left: 7px solid #0E46A3;
        border-radius: 20px;
        padding: 18px 20px;
        box-shadow: 0 14px 28px rgba(15, 23, 42, 0.07);
        margin-top: 8px;
    }

    .insight-title {
        color: #0f172a;
        font-size: 17px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .insight-text {
        color: #334155;
        font-size: 14px;
        line-height: 1.7;
    }

    .custom-table-wrap {
        background: rgba(255,255,255,0.78);
        border-radius: 22px;
        padding: 10px;
        box-shadow: 0 14px 28px rgba(15, 23, 42, 0.07);
        overflow-x: auto;
        margin: 12px 0 22px 0;
    }

    .custom-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 13px;
        color: #0f172a;
    }

    .custom-table th {
        background: #0f172a;
        color: #ffffff !important;
        text-align: left;
        padding: 12px 12px;
        font-weight: 700;
    }

    .custom-table th:first-child {border-top-left-radius: 14px;}
    .custom-table th:last-child {border-top-right-radius: 14px;}

    .custom-table td {
        padding: 11px 12px;
        border-bottom: 1px solid #e2e8f0;
        color: #0f172a !important;
        background: rgba(255,255,255,0.96);
    }

    .custom-table tr:nth-child(even) td {background: #F8FAFC;}
    .custom-table tr:hover td {background: #EDF4FF;}

    .rank-pill {
        background: #E8EEFF;
        color: #0E46A3;
        padding: 4px 9px;
        border-radius: 999px;
        font-weight: 800;
        font-size: 12px;
    }

    .metric-badge {
        background: #E8EEFF;
        color: #0f172a;
        padding: 4px 8px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 12px;
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


def fmt_num(value, prefix: str = "", suffix: str = "") -> str:
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


def kpi_card(label: str, value: str, caption: str, color_class: str) -> None:
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


def panel(title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="panel-title">{title}</div>
        <div class="panel-subtitle">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


def insight_box(title: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="insight-box">
            <div class="insight-title">{title}</div>
            <div class="insight-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def short_text(value: str, n: int = 32) -> str:
    value = str(value)
    return value if len(value) <= n else value[: n - 3] + "..."


def prepare_label(df: pd.DataFrame, source_col: str, target_col: str = "label", n: int = 32) -> pd.DataFrame:
    out = df.copy()
    out[target_col] = out[source_col].astype(str).apply(lambda x: short_text(x, n))
    return out


def nice_label(col: str) -> str:
    mapping = {
        "total_views": "Total Views",
        "total_clicks": "Total Clicks",
        "total_purchase_rows": "Total Purchases",
        "total_revenue": "Total Revenue",
        "ctr_percent": "CTR (%)",
        "click_to_purchase_rate_percent": "Purchase Rate (%)",
        "purchase_rate_percent": "Purchase Rate (%)",
        "price": "Price",
        "rating_avg_clean": "Average Rating",
        "category": "Category",
        "brand": "Brand",
        "label": "Product",
        "product_name": "Product",
    }
    return mapping.get(col, col.replace("_", " ").title())


def chart_layout(fig: go.Figure, height: int = 390, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=12, r=18, t=54, b=40),
        font=dict(family="Inter, sans-serif", size=13, color="#0f172a"),
        title_font=dict(family="Inter, sans-serif", size=20, color="#0f172a"),
        hoverlabel=dict(bgcolor="#0E46A3", font_color="#ffffff", bordercolor="#0E46A3"),
        showlegend=legend,
    )
    if legend:
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.22,
                xanchor="left",
                x=0,
                font=dict(color="#334155", size=12),
                bgcolor="rgba(255,255,255,0.55)",
            )
        )
    fig.update_xaxes(
        title_font=dict(color="#334155", size=13),
        tickfont=dict(color="#334155", size=12),
        gridcolor="rgba(15,23,42,0.10)",
        zeroline=False,
    )
    fig.update_yaxes(
        title_font=dict(color="#334155", size=13),
        tickfont=dict(color="#334155", size=12),
        gridcolor="rgba(15,23,42,0.10)",
        zeroline=False,
    )
    return fig


def show_chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)


def barh(df: pd.DataFrame, x: str, y: str, title: str, color: str, text_prefix: str = "", text_suffix: str = "", height: int = 390) -> go.Figure:
    plot_df = df.copy()
    fig = px.bar(
        plot_df,
        x=x,
        y=y,
        orientation="h",
        title=title,
        text=x,
        labels={x: nice_label(x), y: nice_label(y)},
    )
    fig.update_traces(marker_color=color, textposition="outside", cliponaxis=False)
    if text_prefix or text_suffix:
        if text_suffix == "%":
            fig.update_traces(texttemplate=f"{text_prefix}%{{text:.2f}}{text_suffix}")
        elif text_prefix == "$":
            fig.update_traces(texttemplate=f"{text_prefix}%{{text:,.0f}}{text_suffix}")
        else:
            fig.update_traces(texttemplate=f"{text_prefix}%{{text:,.0f}}{text_suffix}")
    else:
        fig.update_traces(texttemplate="%{text:,.0f}")
    fig.update_layout(yaxis=dict(autorange="reversed"), showlegend=False, xaxis_title=nice_label(x), yaxis_title=nice_label(y))
    return chart_layout(fig, height=height, legend=False)


def barv(df: pd.DataFrame, x: str, y: str, title: str, color: str, text_prefix: str = "", text_suffix: str = "", height: int = 390) -> go.Figure:
    fig = px.bar(df, x=x, y=y, title=title, text=y, labels={x: nice_label(x), y: nice_label(y)})
    fig.update_traces(marker_color=color, textposition="outside", cliponaxis=False)
    if text_suffix == "%":
        fig.update_traces(texttemplate=f"{text_prefix}%{{text:.2f}}{text_suffix}")
    elif text_prefix == "$":
        fig.update_traces(texttemplate=f"{text_prefix}%{{text:,.0f}}{text_suffix}")
    else:
        fig.update_traces(texttemplate=f"{text_prefix}%{{text:,.0f}}{text_suffix}")
    fig.update_layout(xaxis_tickangle=-22, showlegend=False, xaxis_title=nice_label(x), yaxis_title=nice_label(y))
    return chart_layout(fig, height=height, legend=False)


def combo_bar_line(df: pd.DataFrame, sort_col: str, rate_col: str, label_col: str, title: str, bar_name: str, line_name: str, color_bar: str, color_line: str) -> go.Figure:
    plot_df = df.dropna(subset=[sort_col, rate_col]).sort_values(sort_col, ascending=False).head(10).copy()
    plot_df["label"] = plot_df[label_col].astype(str).apply(lambda x: short_text(x, 20))
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=plot_df["label"], y=plot_df[sort_col], name=bar_name, marker_color=color_bar, opacity=0.82),
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
            line=dict(color=color_line, width=3),
            marker=dict(size=8, color=color_line),
        ),
        secondary_y=True,
    )
    fig.update_yaxes(title_text=bar_name, secondary_y=False)
    fig.update_yaxes(title_text=line_name, secondary_y=True)
    fig.update_layout(
        title=title,
        xaxis_title="Product",
        xaxis_tickangle=-32,
        margin=dict(b=120),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.28,
            xanchor="left",
            x=0,
            font=dict(color="#334155", size=12),
            bgcolor="rgba(255,255,255,0.55)",
        ),
    )
    return chart_layout(fig, height=455, legend=True)


def pretty_table(df: pd.DataFrame, columns: list[str], rename: dict[str, str] | None = None, max_rows: int = 12) -> None:
    if df.empty:
        st.info("Data tidak tersedia untuk ditampilkan.")
        return
    rename = rename or {}
    show = df[columns].head(max_rows).copy().rename(columns=rename)
    show.insert(0, "#", range(1, len(show) + 1))

    html = '<div class="custom-table-wrap"><table class="custom-table"><thead><tr>'
    for col in show.columns:
        html += f"<th>{col}</th>"
    html += "</tr></thead><tbody>"
    for _, row in show.iterrows():
        html += "<tr>"
        for col in show.columns:
            val = row[col]
            if col == "#":
                out = f'<span class="rank-pill">{int(val)}</span>'
            elif pd.isna(val):
                out = "-"
            elif isinstance(val, (float, np.floating)):
                if "Revenue" in col or "Price" in col:
                    out = f"${val:,.2f}"
                elif "CTR" in col or "Rate" in col:
                    out = f'<span class="metric-badge">{val:.2f}%</span>'
                else:
                    out = f"{val:,.2f}"
            elif isinstance(val, (int, np.integer)):
                out = f"{val:,.0f}"
            else:
                out = str(val)
            html += f"<td>{out}</td>"
        html += "</tr>"
    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================

summary = load_csv("02_python_summary.csv")
funnel = load_csv("17_funnel_summary.csv")
monthly = load_csv("14_monthly_trend.csv")
product_full = load_csv("13_product_performance_full.csv")
high_view_low_ctr_file = load_csv("05_high_view_low_ctr_products.csv")
high_click_low_purchase_file = load_csv("16_high_click_low_purchase_products.csv")
recommendation = load_csv("12_final_recommendation_table.csv")

if product_full.empty:
    st.error("File 13_product_performance_full.csv tidak ditemukan di data/eda_output.")
    st.stop()

# Strong, consistent aggregations from full product data.
category_full = (
    product_full.groupby("category", as_index=False)
    .agg(
        total_products=("product_id", "count"),
        total_views=("total_views", "sum"),
        total_clicks=("total_clicks", "sum"),
        total_add_to_cart=("total_add_to_cart", "sum"),
        total_purchase_rows=("total_purchase_rows", "sum"),
        total_revenue=("total_revenue", "sum"),
        avg_rating=("rating_avg_clean", "mean"),
    )
    .round(2)
)
category_full["ctr_percent"] = (category_full["total_clicks"] / category_full["total_views"].replace(0, np.nan) * 100).round(2)
category_full["purchase_rate_percent"] = (category_full["total_purchase_rows"] / category_full["total_clicks"].replace(0, np.nan) * 100).round(2)

brand_full = (
    product_full.groupby("brand", as_index=False)
    .agg(
        total_products=("product_id", "count"),
        total_views=("total_views", "sum"),
        total_clicks=("total_clicks", "sum"),
        total_purchase_rows=("total_purchase_rows", "sum"),
        total_revenue=("total_revenue", "sum"),
        avg_rating=("rating_avg_clean", "mean"),
    )
    .round(2)
)
brand_full["ctr_percent"] = (brand_full["total_clicks"] / brand_full["total_views"].replace(0, np.nan) * 100).round(2)
brand_full["purchase_rate_percent"] = (brand_full["total_purchase_rows"] / brand_full["total_clicks"].replace(0, np.nan) * 100).round(2)

high_view_low_ctr = high_view_low_ctr_file.copy()
if high_view_low_ctr.empty:
    high_view_low_ctr = product_full[
        (product_full["total_views"] >= product_full["total_views"].quantile(0.75))
        & (product_full["ctr_percent"] < product_full["ctr_percent"].median())
    ].sort_values(["total_views", "ctr_percent"], ascending=[False, True])

high_click_low_purchase = high_click_low_purchase_file.copy()
if high_click_low_purchase.empty:
    high_click_low_purchase = product_full[
        (product_full["total_clicks"] >= product_full["total_clicks"].quantile(0.75))
        & (product_full["click_to_purchase_rate_percent"] < product_full["click_to_purchase_rate_percent"].median())
    ].sort_values(["total_clicks", "click_to_purchase_rate_percent"], ascending=[False, True])

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

# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="dashboard-title">E-Commerce Product Engagement & Conversion Optimization</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="dashboard-subtitle">Analisis funnel, performa produk, peluang optimasi, kategori, dan brand berdasarkan data e-commerce.</div>',
    unsafe_allow_html=True,
)

# ============================================================
# PAGE 1
# ============================================================

if page == "1. Executive Overview":
    s = summary.iloc[0] if not summary.empty else pd.Series(dtype="float64")
    total_views = product_full["total_views"].sum()
    total_clicks = product_full["total_clicks"].sum()
    total_purchases = product_full["total_purchase_rows"].sum()
    total_revenue = product_full["total_revenue"].sum()
    ctr = total_clicks / total_views * 100
    purchase_rate = total_purchases / total_clicks * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi_card("Total Products", fmt_num(len(product_full)), "Produk dianalisis", "card-blue")
    with c2:
        kpi_card("Total Views", fmt_num(total_views), "Product views", "card-cyan")
    with c3:
        kpi_card("Total Clicks", fmt_num(total_clicks), f"CTR {ctr:.2f}%", "card-green")
    with c4:
        kpi_card("Purchases", fmt_num(total_purchases), f"Click → Purchase {purchase_rate:.2f}%", "card-pink")
    with c5:
        kpi_card("Revenue", fmt_num(total_revenue, prefix="$"), "Total sales value", "card-lavender")

    left, right = st.columns([1.05, 1.25])
    with left:
        st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
        panel("Funnel Chart", "View → Click → Add to Cart → Purchase")
        if not funnel.empty:
            f = funnel.sort_values("sort_order")
            fig = px.funnel(f, x="total_events", y="funnel_stage", text="total_events", title="Product Interaction Funnel")
            fig.update_traces(
                marker_color=["#0E46A3", "#4EADEB", "#A6E89A", "#FF9AA2"],
                texttemplate="%{text:,.0f}",
                textfont=dict(color="#0f172a", size=14),
            )
            fig = chart_layout(fig, height=410, legend=False)
            show_chart(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
        panel("Monthly Trend", "Views, clicks, purchases, dan revenue per bulan")
        if not monthly.empty:
            m = monthly.copy()
            m["month"] = pd.to_datetime(m["month"], errors="coerce")
            m = m.sort_values("month")
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=m["month"], y=m["total_views"], mode="lines+markers", name="Views", line=dict(color="#0E46A3", width=3)), secondary_y=False)
            fig.add_trace(go.Scatter(x=m["month"], y=m["total_clicks"], mode="lines+markers", name="Clicks", line=dict(color="#4EADEB", width=3)), secondary_y=False)
            fig.add_trace(go.Scatter(x=m["month"], y=m["total_purchase_rows"], mode="lines+markers", name="Purchases", line=dict(color="#79C779", width=3)), secondary_y=False)
            fig.add_trace(go.Scatter(x=m["month"], y=m["total_revenue"], mode="lines+markers", name="Revenue", line=dict(color="#FF7777", width=3, dash="dot")), secondary_y=True)
            fig.update_yaxes(title_text="Events", secondary_y=False)
            fig.update_yaxes(title_text="Total Revenue", secondary_y=True)
            fig.update_layout(title="Monthly Product Engagement and Revenue")
            fig = chart_layout(fig, height=410, legend=True)
            show_chart(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    top_rev_cat = category_full.sort_values("total_revenue", ascending=False).iloc[0]
    top_ctr_cat = category_full.sort_values("ctr_percent", ascending=False).iloc[0]
    insight_box(
        "Summary Insight",
        f"Total data menunjukkan <b>{fmt_num(total_views)}</b> views, <b>{fmt_num(total_clicks)}</b> clicks, dan <b>{fmt_num(total_purchases)}</b> purchase rows. "
        f"CTR keseluruhan sebesar <b>{ctr:.2f}%</b>, sedangkan click-to-purchase rate sebesar <b>{purchase_rate:.2f}%</b>. "
        f"Kategori dengan revenue terbesar adalah <b>{top_rev_cat['category']}</b> dengan revenue <b>{fmt_num(top_rev_cat['total_revenue'], prefix='$')}</b>. "
        f"Kategori dengan CTR tertinggi adalah <b>{top_ctr_cat['category']}</b> dengan CTR <b>{top_ctr_cat['ctr_percent']:.2f}%</b>. Fokus optimasi utama adalah produk dengan exposure tinggi tetapi CTR/purchase rate rendah.",
    )

# ============================================================
# PAGE 2
# ============================================================

elif page == "2. Product Performance":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Products", fmt_num(len(product_full)), "Total products", "card-blue")
    with c2:
        kpi_card("Revenue", fmt_num(product_full["total_revenue"].sum(), prefix="$"), "Total revenue", "card-cyan")
    with c3:
        kpi_card("CTR", f"{product_full['total_clicks'].sum()/product_full['total_views'].sum()*100:.2f}%", "Clicks / Views", "card-green")
    with c4:
        kpi_card("Purchase Rate", f"{product_full['total_purchase_rows'].sum()/product_full['total_clicks'].sum()*100:.2f}%", "Purchases / Clicks", "card-pink")

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
        panel("Top 10 Products by CTR", "Minimum 30 views agar CTR tidak bias")
        plot_df = product_full[product_full["total_views"] >= 30].sort_values("ctr_percent", ascending=False).head(10).copy()
        plot_df = prepare_label(plot_df, "product_name", n=34)
        show_chart(barh(plot_df, "ctr_percent", "label", "Highest Product CTR", "#0E46A3", text_suffix="%"))
        st.markdown('</div>', unsafe_allow_html=True)
    with r1c2:
        st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
        panel("Top 10 Products by Revenue", "Produk dengan nilai transaksi terbesar")
        plot_df = product_full.sort_values("total_revenue", ascending=False).head(10).copy()
        plot_df = prepare_label(plot_df, "product_name", n=34)
        show_chart(barh(plot_df, "total_revenue", "label", "Highest Revenue Products", "#2474B5", text_prefix="$"))
        st.markdown('</div>', unsafe_allow_html=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
        panel("Top 10 Products by Purchases", "Produk dengan jumlah pembelian terbanyak")
        plot_df = product_full.sort_values("total_purchase_rows", ascending=False).head(10).copy()
        plot_df = prepare_label(plot_df, "product_name", n=34)
        show_chart(barh(plot_df, "total_purchase_rows", "label", "Highest Purchased Products", "#147A3D"))
        st.markdown('</div>', unsafe_allow_html=True)
    with r2c2:
        st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
        panel("Views and CTR by Product", "Top 10 produk berdasarkan views, dengan CTR sebagai garis")
        show_chart(combo_bar_line(product_full, "total_views", "ctr_percent", "product_name", "Top Products by Views with CTR Overlay", "Views", "CTR (%)", "#B8D9FF", "#0E46A3"))
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    panel("Clicks and Purchase Rate by Product", "Top 10 produk berdasarkan clicks, dengan purchase rate sebagai garis")
    show_chart(combo_bar_line(product_full, "total_clicks", "click_to_purchase_rate_percent", "product_name", "Top Products by Clicks with Purchase Rate Overlay", "Clicks", "Purchase Rate (%)", "#CDEFD2", "#2E8B57"))
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# PAGE 3
# ============================================================

elif page == "3. Product Optimization Opportunity":
    c1, c2 = st.columns(2)
    with c1:
        kpi_card("High View, Low CTR", fmt_num(len(high_view_low_ctr)), "Exposure tinggi, CTR relatif rendah", "card-blue")
    with c2:
        kpi_card("High Click, Low Purchase", fmt_num(len(high_click_low_purchase)), "Klik tinggi, pembelian rendah", "card-pink")

    tab1, tab2 = st.tabs(["High View, Low CTR", "High Click, Low Purchase"])

    with tab1:
        st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
        panel("High View, Low CTR", "Produk sering dilihat tetapi kurang mendorong klik")
        plot_df = high_view_low_ctr.sort_values("total_views", ascending=False).head(10).copy()
        plot_df = prepare_label(plot_df, "product_name", n=36)
        show_chart(barh(plot_df, "total_views", "label", "High Exposure Products with Lower CTR", "#B91C1C"))
        st.markdown('</div>', unsafe_allow_html=True)
        pretty_table(
            high_view_low_ctr.sort_values("total_views", ascending=False),
            ["product_name", "category", "brand", "price", "rating_avg_clean", "total_views", "total_clicks", "ctr_percent", "total_revenue"],
            {"product_name": "Product Name", "rating_avg_clean": "Rating", "total_views": "Views", "total_clicks": "Clicks", "ctr_percent": "CTR", "total_revenue": "Revenue", "price": "Price"},
            max_rows=10,
        )

    with tab2:
        st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
        panel("High Click, Low Purchase", "Produk sudah menarik klik, tetapi purchase rate masih rendah")
        plot_df = high_click_low_purchase.sort_values("total_clicks", ascending=False).head(10).copy()
        plot_df = prepare_label(plot_df, "product_name", n=36)
        show_chart(barh(plot_df, "total_clicks", "label", "High Click Products with Low Purchase Rate", "#EA580C"))
        st.markdown('</div>', unsafe_allow_html=True)
        pretty_table(
            high_click_low_purchase.sort_values("total_clicks", ascending=False),
            ["product_name", "category", "brand", "price", "rating_avg_clean", "total_clicks", "total_purchase_rows", "click_to_purchase_rate_percent", "total_revenue"],
            {"product_name": "Product Name", "rating_avg_clean": "Rating", "total_clicks": "Clicks", "total_purchase_rows": "Purchases", "click_to_purchase_rate_percent": "Purchase Rate", "total_revenue": "Revenue", "price": "Price"},
            max_rows=10,
        )

# ============================================================
# PAGE 4
# ============================================================

elif page == "4. Category & Brand Performance":
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Categories", fmt_num(category_full["category"].nunique()), "Total categories", "card-blue")
    with c2:
        kpi_card("Category Revenue", fmt_num(category_full["total_revenue"].sum(), prefix="$"), "Total category revenue", "card-cyan")
    with c3:
        cat_ctr = category_full["total_clicks"].sum() / category_full["total_views"].sum() * 100
        kpi_card("Category CTR", f"{cat_ctr:.2f}%", "Clicks / Views", "card-green")

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
        panel("CTR by Category", "Seluruh kategori, diurutkan dari CTR tertinggi")
        plot_df = category_full.sort_values("ctr_percent", ascending=False)
        show_chart(barv(plot_df, "category", "ctr_percent", "CTR by Category", "#6AAED6", text_suffix="%"))
        st.markdown('</div>', unsafe_allow_html=True)
    with r1c2:
        st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
        panel("Revenue by Category", "Seluruh kategori, diurutkan dari revenue terbesar")
        plot_df = category_full.sort_values("total_revenue", ascending=False)
        show_chart(barv(plot_df, "category", "total_revenue", "Revenue by Category", "#2474B5", text_prefix="$"))
        st.markdown('</div>', unsafe_allow_html=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
        panel("Purchase Rate by Category", "Purchases dibanding total clicks per kategori")
        plot_df = category_full.sort_values("purchase_rate_percent", ascending=False)
        show_chart(barv(plot_df, "category", "purchase_rate_percent", "Purchase Rate by Category", "#70BF73", text_suffix="%"))
        st.markdown('</div>', unsafe_allow_html=True)
    with r2c2:
        st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
        panel("Top 10 Brands by Revenue", "Brand dengan revenue terbesar")
        plot_df = brand_full.sort_values("total_revenue", ascending=False).head(10).copy()
        plot_df = prepare_label(plot_df, "brand", "label", n=26)
        show_chart(barh(plot_df, "total_revenue", "label", "Top 10 Brands by Revenue", "#79BFF2", text_prefix="$"))
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    panel("Top 10 Brands by CTR", "Minimum 100 views agar ranking CTR lebih stabil")
    plot_df = brand_full[brand_full["total_views"] >= 100].sort_values("ctr_percent", ascending=False).head(10).copy()
    plot_df = prepare_label(plot_df, "brand", "label", n=28)
    show_chart(barh(plot_df, "ctr_percent", "label", "Top 10 Brands by CTR", "#0E46A3", text_suffix="%", height=420))
    st.markdown('</div>', unsafe_allow_html=True)
