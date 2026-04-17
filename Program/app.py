"""智慧食堂数据管理系统 - Streamlit 交互看板

启动方式: cd Program && streamlit run app.py
"""

import os
import sys

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from config import *

# ============ 全局样式设置 ============
sns.set_style("whitegrid")
plt.rcParams["font.sans-serif"] = FONT_CONFIG["font_family"]
plt.rcParams["axes.unicode_minus"] = FONT_CONFIG["axes_unicode_minus"]

st.set_page_config(
    page_title=f"{CANTEEN_NAME}数据分析看板",
    page_icon="🍽",
    layout="wide",
)


# ============ 辅助函数 ============
@st.cache_data
def load_daily_summary():
    path = os.path.join(PROCESSED_DIR, "daily_summary.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date")


@st.cache_data
def load_cleaned_data():
    path = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    df = pd.read_csv(path, parse_dates=["transaction_time"])
    return df


@st.cache_data
def load_service_interval_stats():
    path = os.path.join(PROCESSED_DIR, "service_interval_stats.csv")
    return pd.read_csv(path)


@st.cache_data
def load_service_interval_meal_stats():
    path = os.path.join(PROCESSED_DIR, "service_interval_meal_stats.csv")
    return pd.read_csv(path)


@st.cache_data
def load_cluster_profiles():
    path = os.path.join(PROCESSED_DIR, "cluster_profiles.csv")
    return pd.read_csv(path)


@st.cache_data
def load_prediction_metrics():
    path = os.path.join(PROCESSED_DIR, "prediction_metrics.csv")
    return pd.read_csv(path)


def load_figure(fig_name):
    """加载图表图片"""
    path = os.path.join(FIGURE_DIR, fig_name)
    if os.path.exists(path):
        return path
    return None


def render_fig(fig_path, caption=None):
    """渲染图表"""
    if fig_path and os.path.exists(fig_path):
        st.image(fig_path, use_container_width=True)
        if caption:
            st.caption(caption)
    else:
        st.warning(f"图表未找到: {fig_path}")


# ============ 侧边栏 ============
st.sidebar.title(f":knife_spoon: {CANTEEN_NAME}")
st.sidebar.markdown("---")
st.sidebar.markdown("**分析模块**")

selected = st.sidebar.radio(
    "选择模块",
    [
        "📊 概览仪表盘",
        "⏱ 服务间隔分析",
        "👥 用户聚类画像",
        "📈 消费预测",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"<small>数据范围: {SEMESTER_START} ~ {SEMESTER_END}</small>",
    unsafe_allow_html=True,
)


# ============ 页面：概览仪表盘 ============
if selected == "📊 概览仪表盘":
    st.title("📊 概览仪表盘")
    st.markdown("---")

    df_daily = load_daily_summary()
    df_clean = load_cleaned_data()

    col1, col2, col3, col4 = st.columns(4)

    total_days = len(df_daily)
    total_traffic = int(df_daily["total_transaction_count"].sum())
    total_amount = df_daily["total_amount"].sum()
    avg_daily = total_traffic / total_days if total_days > 0 else 0

    col1.metric("分析天数", f"{total_days} 天")
    col2.metric("总就餐人次", f"{total_traffic:,} 人次")
    col3.metric("日均就餐", f"{avg_daily:.0f} 人次/天")
    col4.metric("总消费额", f"¥{total_amount:,.0f}")

    st.markdown("---")

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.subheader("每日就餐人次趋势")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df_daily["date"], df_daily["total_transaction_count"],
               color="#3498DB", linewidth=1.2, alpha=0.8)
        ax.fill_between(df_daily["date"], df_daily["total_transaction_count"],
                        alpha=0.2, color="#3498DB")
        ax.set_xlabel("")
        ax.set_ylabel("就餐人次")
        ax.set_title("日均就餐人次趋势", fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    with col_right:
        st.subheader("周内就餐规律")
        weekday_map = {0: "周一", 1: "周二", 2: "周三", 3: "周四",
                       4: "周五", 5: "周六", 6: "周日"}
        df_daily["weekday"] = df_daily["date"].dt.dayofweek.map(weekday_map)
        weekday_avg = df_daily.groupby("weekday")["total_transaction_count"].mean()
        weekday_order = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        weekday_avg = weekday_avg.reindex([w for w in weekday_order if w in weekday_avg.index])

        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars = ax.bar(weekday_avg.index, weekday_avg.values,
                      color=sns.color_palette("Set2", len(weekday_avg)))
        ax.set_ylabel("日均就餐人次")
        ax.set_title("周内就餐规律", fontsize=12)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{bar.get_height():.0f}", ha="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("各窗口客流占比")

    fig_path = load_figure("05_window_share.png")
    if fig_path:
        st.image(fig_path, width=600)
    else:
        st.info("图表正在生成中...")


# ============ 页面：服务间隔分析 ============
elif selected == "⏱ 服务间隔分析":
    st.title("⏱ 服务间隔分析")
    st.markdown(
        "<span style='color:#e74c3c'>[估算说明]</span> "
        "本分析基于相邻交易时间差，并非精确排队时间。"
        "该值含顾客等待时间与服务时间，低峰时段会被客流稀疏拉大。",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    stats = load_service_interval_stats()
    meal_stats = load_service_interval_meal_stats()

    # 关键指标
    avg_all = stats["avg_interval"].mean()
    median_all = stats["median_interval"].median()
    col1, col2, col3 = st.columns(3)
    col1.metric("全局平均服务间隔", f"{avg_all:.1f} 秒")
    col2.metric("全局中位服务间隔", f"{median_all:.1f} 秒")
    col3.metric("分析窗口数", f"{len(stats)} 个")

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        fig_path = load_figure("08_avg_service_interval.png")
        render_fig(fig_path, "各窗口平均服务间隔估算（高峰时段）")
    with col_b:
        fig_path = load_figure("11_service_interval_meal_comparison.png")
        render_fig(fig_path, "午餐/晚餐服务间隔对比")

    st.markdown("---")

    col_c, col_d = st.columns(2)
    with col_c:
        fig_path = load_figure("09_service_interval_boxplot.png")
        render_fig(fig_path, "服务间隔分布箱线图（≤5分钟）")
    with col_d:
        fig_path = load_figure("10_service_interval_heatmap.png")
        render_fig(fig_path, "窗口×小时服务间隔热力图")

    st.markdown("---")

    st.subheader("详细统计数据")
    tab1, tab2 = st.tabs(["按窗口", "按餐时段"])

    with tab1:
        st.dataframe(stats, use_container_width=True, hide_index=True)

    with tab2:
        st.dataframe(meal_stats, use_container_width=True, hide_index=True)


# ============ 页面：用户聚类画像 ============
elif selected == "👥 用户聚类画像":
    st.title("👥 用户聚类画像")
    st.markdown("---")

    profiles = load_cluster_profiles()

    col1, col2 = st.columns(2)
    with col1:
        fig_path = load_figure("12_k_evaluation.png")
        render_fig(fig_path, "K-Means 聚类数量评估")
    with col2:
        fig_path = load_figure("16_silhouette.png")
        render_fig(fig_path, "轮廓系数分析")

    st.markdown("---")

    col3, col4 = st.columns(2)
    with col3:
        fig_path = load_figure("13_cluster_radar.png")
        render_fig(fig_path, "各用户群体特征雷达图")
    with col4:
        fig_path = load_figure("14_cluster_scatter.png")
        render_fig(fig_path, "用户聚类散点图")

    st.markdown("---")

    col5, col6 = st.columns(2)
    with col5:
        fig_path = load_figure("15_cluster_heatmap.png")
        render_fig(fig_path, "各群体消费热力图")
    with col6:
        if not profiles.empty:
            st.subheader("各群体统计概览")
            st.dataframe(profiles, use_container_width=True, hide_index=True)


# ============ 页面：消费预测 ============
elif selected == "📈 消费预测":
    st.title("📈 消费预测")
    st.markdown("---")

    metrics = load_prediction_metrics()
    if not metrics.empty:
        st.subheader("模型性能对比")
        col1, col2, col3 = st.columns(3)
        for i, row in metrics.iterrows():
            name = row["name"]
            mae = row["MAE"]
            rmse = row["RMSE"]
            mape = row["MAPE"]
            with [col1, col2, col3][i % 3]:
                st.metric(f"{name} - MAE", f"{mae:.1f}")
                st.caption(f"RMSE: {rmse:.1f} | MAPE: {mape:.1f}%")

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        fig_path = load_figure("17_arima_forecast.png")
        render_fig(fig_path, "ARIMA 时间序列预测")
    with col_b:
        fig_path = load_figure("18_xgboost_forecast.png")
        render_fig(fig_path, "XGBoost 特征预测")

    st.markdown("---")

    col_c, col_d = st.columns(2)
    with col_c:
        fig_path = load_figure("19_model_comparison.png")
        render_fig(fig_path, "模型 MAE/RMSE/MAPE 对比")
    with col_d:
        fig_path = load_figure("20_feature_importance.png")
        render_fig(fig_path, "XGBoost 特征重要性 Top10")

    if not metrics.empty:
        st.markdown("---")
        st.subheader("评估指标详情")
        st.dataframe(metrics, use_container_width=True, hide_index=True)
