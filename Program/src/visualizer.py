"""数据可视化模块

生成多维度图表，涵盖：
1. 时段分布热力图
2. 各窗口营收对比（柱状图）
3. 消费趋势折线图
4. 客单价分布（直方图/箱线图）
5. 窗口消费占比（饼图）
6. 就餐人数时间序列
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


def setup_plot_style():
    """配置matplotlib中文显示与全局样式"""
    import matplotlib.font_manager as fm

    # 查找系统中可用的中文字体
    chinese_fonts = []
    for f in fm.fontManager.ttflist:
        if any(name in f.name for name in ["SimHei", "Microsoft YaHei", "SimSun", "KaiTi", "FangSong", "STSong", "Noto Sans CJK", "WenQuanYi"]):
            chinese_fonts.append(f.name)

    if chinese_fonts:
        preferred = chinese_fonts[0]
        plt.rcParams["font.sans-serif"] = [preferred] + plt.rcParams["font.sans-serif"]
        print(f"使用中文字体: {preferred}")
    else:
        print("警告: 未找到中文字体，中文可能无法正常显示")

    plt.rcParams["axes.unicode_minus"] = FONT_CONFIG["axes_unicode_minus"]
    sns.set_theme(style="whitegrid", palette=COLOR_PALETTE)

    # sns.set_theme会重置字体，必须在之后重新设置
    if chinese_fonts:
        plt.rcParams["font.sans-serif"] = [preferred] + plt.rcParams["font.sans-serif"]
        plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False


def save_figure(fig, name, output_dir=FIGURE_DIR):
    """保存图表"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.{FIGURE_FORMAT}")
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  已保存: {path}")
    return path


def load_processed_data(data_dir=PROCESSED_DIR):
    """加载预处理后的数据"""
    cleaned = pd.read_csv(os.path.join(data_dir, "cleaned_data.csv"), parse_dates=["transaction_time"])
    daily = pd.read_csv(os.path.join(data_dir, "daily_summary.csv"))
    hourly = pd.read_csv(os.path.join(data_dir, "hourly_summary.csv"))
    user_features = pd.read_csv(os.path.join(data_dir, "user_features.csv"))
    return cleaned, daily, hourly, user_features


# ============================================================
# 图表1: 时段分布热力图 - 窗口×小时 平均客流量
# ============================================================
def plot_traffic_heatmap(cleaned_df, output_dir=FIGURE_DIR):
    """各窗口在不同小时的平均客流量热力图"""
    # 按窗口×小时统计平均每日客流
    heatmap_data = (
        cleaned_df.groupby(["window_name", "hour"])["amount"]
        .count()
        .reset_index()
    )
    heatmap_data["avg_daily"] = heatmap_data["amount"] / cleaned_df["date"].nunique()

    # 透视表
    pivot = heatmap_data.pivot(index="window_name", columns="hour", values="avg_daily").fillna(0)

    # 按窗口配置顺序排列
    window_order = [cfg["name"] for cfg in WINDOW_CONFIG.values()]
    pivot = pivot.reindex([w for w in window_order if w in pivot.index])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pivot, annot=True, fmt=".0f", cmap="YlOrRd",
        linewidths=0.5, ax=ax, cbar_kws={"label": "日均客流（人次）"}
    )
    ax.set_title(f"{CANTEEN_NAME} 各窗口分时段日均客流热力图", fontsize=14, fontweight="bold")
    ax.set_xlabel("小时", fontsize=12)
    ax.set_ylabel("窗口", fontsize=12)

    return save_figure(fig, "01_traffic_heatmap", output_dir)


# ============================================================
# 图表2: 各窗口营收对比柱状图
# ============================================================
def plot_window_revenue(cleaned_df, output_dir=FIGURE_DIR):
    """各窗口总营收与平均客单价对比"""
    window_stats = (
        cleaned_df.groupby("window_name")
        .agg(total_revenue=("amount", "sum"), avg_price=("amount", "mean"), count=("amount", "count"))
        .reset_index()
        .sort_values("total_revenue", ascending=True)
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 总营收
    colors = sns.color_palette(COLOR_PALETTE, len(window_stats))
    axes[0].barh(window_stats["window_name"], window_stats["total_revenue"] / 10000, color=colors)
    axes[0].set_xlabel("总营收（万元）", fontsize=12)
    axes[0].set_title("各窗口学期总营收", fontsize=13, fontweight="bold")
    for i, v in enumerate(window_stats["total_revenue"] / 10000):
        axes[0].text(v + 0.1, i, f"{v:.1f}万", va="center", fontsize=10)

    # 平均客单价
    axes[1].barh(window_stats["window_name"], window_stats["avg_price"], color=colors)
    axes[1].set_xlabel("平均客单价（元）", fontsize=12)
    axes[1].set_title("各窗口平均客单价", fontsize=13, fontweight="bold")
    for i, v in enumerate(window_stats["avg_price"]):
        axes[1].text(v + 0.2, i, f"{v:.1f}元", va="center", fontsize=10)

    plt.tight_layout()
    return save_figure(fig, "02_window_revenue", output_dir)


# ============================================================
# 图表3: 消费趋势折线图（按周）
# ============================================================
def plot_weekly_trend(daily_summary, output_dir=FIGURE_DIR):
    """按周统计的消费趋势折线图"""
    daily = daily_summary.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["week"] = daily["date"].dt.isocalendar().week.astype(int)
    daily["year"] = daily["date"].dt.year
    daily["year_week"] = daily["date"].dt.strftime("%Y-W%V")

    weekly = (
        daily.groupby("year_week")
        .agg(
            total_amount=("total_amount", "sum"),
            total_transactions=("total_transaction_count", "sum"),
        )
        .reset_index()
        .sort_values("year_week")
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # 消费总额趋势
    axes[0].plot(range(len(weekly)), weekly["total_amount"] / 10000, "o-",
                 color="#E74C3C", linewidth=2, markersize=6)
    axes[0].fill_between(range(len(weekly)), weekly["total_amount"] / 10000,
                         alpha=0.15, color="#E74C3C")
    axes[0].set_ylabel("周消费总额（万元）", fontsize=12)
    axes[0].set_title(f"{CANTEEN_NAME} 学期周消费趋势", fontsize=14, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    # 交易笔数趋势
    axes[1].plot(range(len(weekly)), weekly["total_transactions"], "s-",
                 color="#3498DB", linewidth=2, markersize=6)
    axes[1].fill_between(range(len(weekly)), weekly["total_transactions"],
                         alpha=0.15, color="#3498DB")
    axes[1].set_ylabel("周交易笔数", fontsize=12)
    axes[1].set_xlabel("周次", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # X轴标签
    tick_labels = [f"W{i+1}" for i in range(len(weekly))]
    axes[1].set_xticks(range(len(weekly)))
    axes[1].set_xticklabels(tick_labels, rotation=45, fontsize=9)

    plt.tight_layout()
    return save_figure(fig, "03_weekly_trend", output_dir)


# ============================================================
# 图表4: 客单价分布直方图 + 箱线图
# ============================================================
def plot_amount_distribution(cleaned_df, output_dir=FIGURE_DIR):
    """客单价分布直方图和各窗口箱线图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 直方图
    axes[0].hist(cleaned_df["amount"], bins=40, color="#3498DB", edgecolor="white", alpha=0.8)
    axes[0].axvline(cleaned_df["amount"].mean(), color="#E74C3C", linestyle="--",
                    linewidth=2, label=f'均值: {cleaned_df["amount"].mean():.1f}元')
    axes[0].axvline(cleaned_df["amount"].median(), color="#2ECC71", linestyle="--",
                    linewidth=2, label=f'中位数: {cleaned_df["amount"].median():.1f}元')
    axes[0].set_xlabel("消费金额（元）", fontsize=12)
    axes[0].set_ylabel("交易笔数", fontsize=12)
    axes[0].set_title("客单价分布直方图", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=11)

    # 箱线图
    window_order = [cfg["name"] for cfg in WINDOW_CONFIG.values()]
    plot_data = cleaned_df[cleaned_df["window_name"].isin(window_order)]
    sns.boxplot(data=plot_data, y="window_name", x="amount", order=window_order,
                palette=COLOR_PALETTE, ax=axes[1])
    axes[1].set_xlabel("消费金额（元）", fontsize=12)
    axes[1].set_ylabel("窗口", fontsize=12)
    axes[1].set_title("各窗口客单价箱线图", fontsize=13, fontweight="bold")

    plt.tight_layout()
    return save_figure(fig, "04_amount_distribution", output_dir)


# ============================================================
# 图表5: 窗口消费占比饼图
# ============================================================
def plot_window_share(cleaned_df, output_dir=FIGURE_DIR):
    """各窗口交易笔数和营收占比饼图"""
    window_stats = (
        cleaned_df.groupby("window_name")
        .agg(count=("amount", "count"), revenue=("amount", "sum"))
        .reset_index()
    )

    # 按配置顺序
    window_order = [cfg["name"] for cfg in WINDOW_CONFIG.values()]
    window_stats["order"] = window_stats["window_name"].map({n: i for i, n in enumerate(window_order)})
    window_stats = window_stats.sort_values("order")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = sns.color_palette(COLOR_PALETTE, len(window_stats))

    # 交易笔数占比
    axes[0].pie(
        window_stats["count"], labels=window_stats["window_name"],
        autopct="%1.1f%%", colors=colors, startangle=90, pctdistance=0.85
    )
    axes[0].set_title("各窗口交易笔数占比", fontsize=13, fontweight="bold")

    # 营收占比
    axes[1].pie(
        window_stats["revenue"], labels=window_stats["window_name"],
        autopct="%1.1f%%", colors=colors, startangle=90, pctdistance=0.85
    )
    axes[1].set_title("各窗口营收占比", fontsize=13, fontweight="bold")

    plt.tight_layout()
    return save_figure(fig, "05_window_share", output_dir)


# ============================================================
# 图表6: 每日就餐人数时间序列
# ============================================================
def plot_daily_traffic(daily_summary, output_dir=FIGURE_DIR):
    """每日就餐人数时间序列图（午餐/晚餐分开）"""
    daily = daily_summary.copy()
    daily["date"] = pd.to_datetime(daily["date"])

    fig, ax = plt.subplots(figsize=(14, 6))

    lunch_col = [c for c in daily.columns if "午餐" in c and "transaction_count" in c]
    dinner_col = [c for c in daily.columns if "晚餐" in c and "transaction_count" in c]

    if lunch_col:
        ax.plot(daily["date"], daily[lunch_col[0]], "-o", color="#E67E22",
                markersize=3, linewidth=1.5, label="午餐", alpha=0.8)
    if dinner_col:
        ax.plot(daily["date"], daily[dinner_col[0]], "-s", color="#8E44AD",
                markersize=3, linewidth=1.5, label="晚餐", alpha=0.8)

    # 总计
    ax.plot(daily["date"], daily["total_transaction_count"], "-", color="#2C3E50",
            linewidth=2, label="合计", alpha=0.6)

    ax.set_xlabel("日期", fontsize=12)
    ax.set_ylabel("交易笔数", fontsize=12)
    ax.set_title(f"{CANTEEN_NAME} 每日就餐交易笔数趋势", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()
    return save_figure(fig, "06_daily_traffic", output_dir)


# ============================================================
# 图表7: 星期效应分析
# ============================================================
def plot_weekday_pattern(cleaned_df, output_dir=FIGURE_DIR):
    """不同星期的消费模式对比"""
    weekday_names = ["周一", "周二", "周三", "周四", "周五"]

    weekday_stats = (
        cleaned_df.groupby("weekday")
        .agg(
            avg_daily_count=("amount", "count"),
            avg_amount=("amount", "mean"),
        )
        .reset_index()
    )

    # 计算日均（除以该星期的天数）
    days_per_weekday = cleaned_df.groupby("weekday")["date"].nunique().values
    weekday_stats["avg_daily_count"] = weekday_stats["avg_daily_count"] / days_per_weekday
    weekday_stats["weekday_name"] = weekday_names

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 日均客流
    bars = axes[0].bar(weekday_stats["weekday_name"], weekday_stats["avg_daily_count"],
                       color=sns.color_palette(COLOR_PALETTE, 5))
    axes[0].set_ylabel("日均交易笔数", fontsize=12)
    axes[0].set_title("各星期日均客流量", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, weekday_stats["avg_daily_count"]):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     f"{val:.0f}", ha="center", fontsize=10)

    # 平均消费
    bars2 = axes[1].bar(weekday_stats["weekday_name"], weekday_stats["avg_amount"],
                        color=sns.color_palette(COLOR_PALETTE, 5))
    axes[1].set_ylabel("平均消费金额（元）", fontsize=12)
    axes[1].set_title("各星期平均客单价", fontsize=13, fontweight="bold")
    for bar, val in zip(bars2, weekday_stats["avg_amount"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f"{val:.1f}", ha="center", fontsize=10)

    plt.tight_layout()
    return save_figure(fig, "07_weekday_pattern", output_dir)


def generate_all_visualizations(data_dir=PROCESSED_DIR, output_dir=FIGURE_DIR):
    """生成所有可视化图表"""
    setup_plot_style()
    os.makedirs(output_dir, exist_ok=True)

    print("正在加载数据...")
    cleaned, daily, hourly, user_features = load_processed_data(data_dir)

    print("\n===== 生成可视化图表 =====")

    print("\n[1/7] 时段分布热力图...")
    plot_traffic_heatmap(cleaned, output_dir)

    print("[2/7] 窗口营收对比...")
    plot_window_revenue(cleaned, output_dir)

    print("[3/7] 消费趋势折线图...")
    plot_weekly_trend(daily, output_dir)

    print("[4/7] 客单价分布...")
    plot_amount_distribution(cleaned, output_dir)

    print("[5/7] 窗口消费占比...")
    plot_window_share(cleaned, output_dir)

    print("[6/7] 每日就餐趋势...")
    plot_daily_traffic(daily, output_dir)

    print("[7/7] 星期效应分析...")
    plot_weekday_pattern(cleaned, output_dir)

    print(f"\n所有图表已保存至: {output_dir}")


if __name__ == "__main__":
    generate_all_visualizations()
