"""窗口服务间隔分析模块

基于刷卡交易时间戳，估算各窗口的相邻交易服务间隔。
核心逻辑：同一窗口在同一天同一餐时段内，相邻两笔刷卡交易的时间差，
可近似估算为"交易服务间隔"（含服务时间 + 真实排队等待时间）。

⚠ 注意：此指标为估算值，不代表精确排队时间。详见各图表中的说明。

分析步骤：
1. 读取清洗后交易数据
2. 按窗口+日期+餐时段分组，按交易时间排序
3. 计算相邻交易的时间差（秒）
4. 剔除异常值（负数、极短/极长间隔）
5. 筛选高峰时段数据（客流密集时估算更准确）
6. 汇总各窗口平均服务间隔
7. 输出可视化图表
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from src.visualizer import setup_plot_style, save_figure


# ============ 异常值过滤阈值 ============
MIN_INTERVAL_SEC = 3       # 最短有效间隔（秒），低于此视为系统刷重复记录
MAX_INTERVAL_SEC = 600     # 最长有效间隔（秒=10分钟），超过此视为客流断档而非排队


def load_data(data_dir=PROCESSED_DIR):
    """加载清洗后的交易数据，确保时间戳格式统一"""
    filepath = os.path.join(data_dir, "cleaned_data.csv")
    df = pd.read_csv(filepath, parse_dates=["transaction_time"])

    # 校验：确保时间戳解析成功
    if df["transaction_time"].isna().any():
        bad_count = df["transaction_time"].isna().sum()
        print(f"  警告: 发现 {bad_count} 条时间戳解析失败的记录，已剔除")
        df = df.dropna(subset=["transaction_time"])

    # 校验：确保关键列存在
    required_cols = ["window_id", "window_name", "transaction_time", "meal_period", "date"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"数据缺少必要列: {missing}")

    print(f"  已加载 {len(df)} 条交易记录，覆盖 {df['window_id'].nunique()} 个窗口")
    return df


def compute_intervals(df):
    """计算同一窗口、同一天、同一餐时段内相邻交易的服务间隔

    逻辑：
    - 按 window_id + date + meal_period 分组
    - 组内按 transaction_time 升序排列
    - 计算相邻记录的时间差（秒）
    - 服务间隔 = 第N笔交易时间戳 - 第(N-1)笔交易时间戳
    """
    # 按 窗口+日期+餐时段 分组，并按时间排序
    df_sorted = df.sort_values(
        ["window_id", "date", "meal_period", "transaction_time"]
    ).copy()

    # 在每个分组内计算与上一条记录的时间差
    df_sorted["prev_time"] = df_sorted.groupby(
        ["window_id", "date", "meal_period"]
    )["transaction_time"].shift(1)

    df_sorted["interval_sec"] = (
        df_sorted["transaction_time"] - df_sorted["prev_time"]
    ).dt.total_seconds()

    # 剔除第一条记录（无前序，interval为NaN）
    result = df_sorted.dropna(subset=["interval_sec"]).copy()

    return result


def filter_anomalies(df, min_sec=MIN_INTERVAL_SEC, max_sec=MAX_INTERVAL_SEC):
    """剔除异常服务间隔

    过滤规则：
    - 负数间隔（时间戳乱序）
    - 极短间隔（< min_sec）：系统重复刷卡或同一人多次刷卡
    - 极长间隔（> max_sec）：客流断档，非连续排队
    """
    total = len(df)
    reasons = {
        "负数间隔": (df["interval_sec"] < 0).sum(),
        f"极短间隔(<{min_sec}s)": (df["interval_sec"] < min_sec).sum(),
        f"极长间隔(>{max_sec}s)": (df["interval_sec"] > max_sec).sum(),
    }

    # 过滤
    mask = (
        (df["interval_sec"] >= min_sec) &
        (df["interval_sec"] <= max_sec)
    )
    filtered = df[mask].copy()

    removed = total - len(filtered)
    print(f"\n  数据校验报告:")
    print(f"    总间隔记录: {total}")
    for reason, count in reasons.items():
        if count > 0:
            print(f"    剔除 {reason}: {count} 条")
    print(f"    保留有效记录: {len(filtered)} 条 (剔除率 {removed/total*100:.1f}%)")

    return filtered


def aggregate_window_service_interval(df):
    """汇总各窗口的平均服务间隔

    返回 DataFrame，包含：
    - window_id / window_name
    - avg_interval: 平均服务间隔（秒）
    - median_interval: 中位服务间隔（秒）
    - p25 / p75: 25/75分位数
    - sample_count: 有效样本数
    """
    stats = (
        df.groupby(["window_id", "window_name"])
        .agg(
            avg_interval=("interval_sec", "mean"),
            median_interval=("interval_sec", "median"),
            p25=("interval_sec", lambda x: x.quantile(0.25)),
            p75=("interval_sec", lambda x: x.quantile(0.75)),
            std_interval=("interval_sec", "std"),
            sample_count=("interval_sec", "count"),
        )
        .reset_index()
    )

    # 按窗口配置顺序排列
    window_order = list(WINDOW_CONFIG.keys())
    stats["order"] = stats["window_id"].map({w: i for i, w in enumerate(window_order)})
    stats = stats.sort_values("order").drop(columns="order").reset_index(drop=True)

    return stats


def aggregate_window_meal_service_interval(df):
    """按窗口+餐时段汇总平均服务间隔"""
    stats = (
        df.groupby(["window_id", "window_name", "meal_period"])
        .agg(
            avg_interval=("interval_sec", "mean"),
            median_interval=("interval_sec", "median"),
            sample_count=("interval_sec", "count"),
        )
        .reset_index()
    )

    window_order = list(WINDOW_CONFIG.keys())
    stats["order"] = stats["window_id"].map({w: i for i, w in enumerate(window_order)})
    stats = stats.sort_values(["order", "meal_period"]).drop(columns="order").reset_index(drop=True)

    return stats


def aggregate_window_hour_service_interval(df):
    """按窗口+小时汇总平均服务间隔（用于热力图）"""
    # 只取有效的小时
    valid_hours = sorted(df["hour"].unique())

    stats = (
        df.groupby(["window_name", "hour"])
        .agg(avg_interval=("interval_sec", "mean"))
        .reset_index()
    )

    return stats, valid_hours


# ============================================================
# 图表1: 各窗口平均服务间隔柱状图
# ============================================================
def plot_avg_service_interval(window_stats, output_dir=FIGURE_DIR):
    """各窗口平均服务间隔柱状图（含误差线表示分布）

    ⚠ 估算说明：
    本图展示的是"相邻交易服务间隔"，即同一窗口同一天同一餐时段内，
    相邻两笔刷卡交易的时间差。该值包含：
      - 顾客排队等待时间（近似）
      - 窗口服务操作时间
    ⚠ 注意：低峰时段该值会被人流稀疏拉大，不代表真实排队时间。
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = sns.color_palette(COLOR_PALETTE, len(window_stats))
    x = range(len(window_stats))
    names = window_stats["window_name"]
    avgs = window_stats["avg_interval"]
    p25 = window_stats["p25"]
    p75 = window_stats["p75"]

    bars = ax.bar(x, avgs, color=colors, edgecolor="white", linewidth=0.8)

    # 误差线: IQR (p25 ~ p75)
    lower_err = avgs.values - p25.values
    upper_err = p75.values - avgs.values
    ax.errorbar(x, avgs, yerr=[lower_err, upper_err],
                fmt="none", ecolor="#555555", capsize=4, linewidth=1.2)

    # 标注数值
    for i, (bar, val) in enumerate(zip(bars, avgs)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                f"{val:.1f}s", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("平均服务间隔（秒）", fontsize=12)
    ax.set_title(f"{CANTEEN_NAME} 各窗口平均服务间隔估算\n（基于相邻交易时间差，含等待+服务时间）",
                fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # 添加参考线
    overall_avg = avgs.mean()
    ax.axhline(overall_avg, color="#E74C3C", linestyle="--", linewidth=1.5,
               label=f"全局均值: {overall_avg:.1f}s")
    ax.legend(fontsize=10, loc="upper right")

    plt.tight_layout()
    return save_figure(fig, "08_avg_service_interval", output_dir)


# ============================================================
# 图表2: 服务间隔分布箱线图
# ============================================================
def plot_service_interval_boxplot(df, output_dir=FIGURE_DIR):
    """各窗口服务间隔分布箱线图

    ⚠ 估算说明：展示各窗口服务间隔的分布情况，仅展示≤5分钟的间隔。
    """
    window_order = [cfg["name"] for cfg in WINDOW_CONFIG.values()]

    fig, ax = plt.subplots(figsize=(12, 6))

    # 限制y轴范围以避免极端值影响可视化
    plot_df = df[df["interval_sec"] <= 300].copy()  # 只展示5分钟以内的

    sns.boxplot(
        data=plot_df, x="window_name", y="interval_sec",
        order=window_order, hue="window_name", hue_order=window_order,
        palette=COLOR_PALETTE, ax=ax,
        showfliers=False, legend=False
    )

    ax.set_xlabel("窗口", fontsize=12)
    ax.set_ylabel("服务间隔（秒）", fontsize=12)
    ax.set_title(f"{CANTEEN_NAME} 各窗口服务间隔分布（箱线图，≤5分钟）", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return save_figure(fig, "09_service_interval_boxplot", output_dir)


# ============================================================
# 图表3: 窗口×小时 服务间隔热力图
# ============================================================
def plot_service_interval_heatmap(hour_stats, valid_hours, output_dir=FIGURE_DIR):
    """各窗口×时段平均服务间隔热力图"""
    if hour_stats.empty or len(valid_hours) == 0:
        print("  [WARNING] 热力图数据为空，跳过...")
        return None

    window_order = [cfg["name"] for cfg in WINDOW_CONFIG.values()]

    pivot = hour_stats.pivot(index="window_name", columns="hour", values="avg_interval")
    pivot = pivot.reindex(index=[w for w in window_order if w in pivot.index],
                          columns=valid_hours)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pivot, annot=True, fmt=".0f", cmap="YlOrRd",
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "平均服务间隔（秒）"}
    )
    ax.set_title(f"{CANTEEN_NAME} 各窗口分时段平均服务间隔热力图", fontsize=14, fontweight="bold")
    ax.set_xlabel("小时", fontsize=12)
    ax.set_ylabel("窗口", fontsize=12)

    plt.tight_layout()
    return save_figure(fig, "10_service_interval_heatmap", output_dir)


# ============================================================
# 图表4: 午餐/晚餐服务间隔对比
# ============================================================
def plot_meal_period_comparison(meal_stats, output_dir=FIGURE_DIR):
    """午餐 vs 晚餐各窗口平均服务间隔对比"""
    window_order = [cfg["name"] for cfg in WINDOW_CONFIG.values()]

    pivot = meal_stats.pivot(index="window_name", columns="meal_period", values="avg_interval")
    pivot = pivot.reindex(index=[w for w in window_order if w in pivot.index])

    # 确保两列都存在
    lunch_col = "午餐" if "午餐" in pivot.columns else pivot.columns[0]
    dinner_col = "晚餐" if "晚餐" in pivot.columns else pivot.columns[-1]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(pivot))
    width = 0.35

    lunch_vals = pivot[lunch_col].values
    dinner_vals = pivot[dinner_col].values

    bars1 = ax.bar(x - width / 2, lunch_vals, width, label=lunch_col,
                   color="#E67E22", edgecolor="white")
    bars2 = ax.bar(x + width / 2, dinner_vals, width, label=dinner_col,
                   color="#8E44AD", edgecolor="white")

    # 标注数值
    for bar in bars1:
        if not np.isnan(bar.get_height()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{bar.get_height():.1f}", ha="center", fontsize=9)
    for bar in bars2:
        if not np.isnan(bar.get_height()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{bar.get_height():.1f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, fontsize=11)
    ax.set_ylabel("平均服务间隔（秒）", fontsize=12)
    ax.set_title(f"{CANTEEN_NAME} 午餐/晚餐各窗口平均服务间隔对比", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return save_figure(fig, "11_service_interval_meal_comparison", output_dir)


def run_queue_analysis(data_dir=PROCESSED_DIR, output_dir=FIGURE_DIR):
    """执行完整的服务间隔分析流程"""
    setup_plot_style()
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 50)
    print("窗口服务间隔分析（基于相邻交易时间差）")
    print("=" * 50)

    # Step 1: 加载数据
    print("\n[Step 1] 加载交易数据...")
    df = load_data(data_dir)

    # Step 2: 计算服务间隔
    print("\n[Step 2] 计算相邻交易服务间隔...")
    df_intervals = compute_intervals(df)
    print(f"  生成服务间隔记录: {len(df_intervals)} 条")

    # Step 3: 异常值过滤
    print("\n[Step 3] 数据校验与异常值剔除...")
    df_clean = filter_anomalies(df_intervals)

    # Step 4: 高峰时段过滤
    # 只保留平均间隔≤180秒的窗口×餐时段组合（反映客流密集的连续服务状态）
    print("\n[Step 4] 筛选高峰时段（仅保留平均间隔≤180秒的时段）...")
    group_avg = (
        df_clean.groupby(["window_id", "meal_period"])["interval_sec"]
        .mean().reset_index()
        .rename(columns={"interval_sec": "group_avg"})
    )
    peak_groups = group_avg[group_avg["group_avg"] <= 180][["window_id", "meal_period"]]
    df_peak = df_clean.merge(peak_groups, on=["window_id", "meal_period"])
    peak_ratio = len(df_peak) / len(df_clean) if len(df_clean) > 0 else 0
    print(f"  原始记录: {len(df_clean)} 条 -> 高峰时段记录: {len(df_peak)} 条 "
          f"(保留率 {peak_ratio*100:.1f}%)")
    if len(df_peak) < 10:
        print("  [WARNING] 高峰时段数据过少，回退使用全量数据...")
        df_peak = df_clean

    # Step 5: 汇总统计
    print("\n[Step 5] 汇总各窗口服务间隔...")
    window_stats = aggregate_window_service_interval(df_peak)
    meal_stats = aggregate_window_meal_service_interval(df_peak)
    hour_stats, valid_hours = aggregate_window_hour_service_interval(df_peak)

    print("\n  [估算说明] 本分析基于相邻交易时间差，并非精确排队时间。"
          "\n  [估算说明] 该值含顾客等待时间与服务时间，低峰时段会被客流稀疏拉大。")
    print("\n  各窗口平均服务间隔（高峰时段）:")
    print("-" * 60)
    print(f"  {'窗口':<8} {'平均(秒)':<10} {'中位数(秒)':<12} {'P25(秒)':<10} {'P75(秒)':<10} {'样本数':<8}")
    print("-" * 60)
    for _, row in window_stats.iterrows():
        print(f"  {row['window_name']:<8} {row['avg_interval']:<10.1f} "
              f"{row['median_interval']:<12.1f} {row['p25']:<10.1f} "
              f"{row['p75']:<10.1f} {row['sample_count']:<8.0f}")
    print("-" * 60)

    # Step 6: 保存统计结果
    stats_path = os.path.join(data_dir, "service_interval_stats.csv")
    window_stats.to_csv(stats_path, index=False, encoding="utf-8-sig")
    print(f"\n  统计结果已保存: {stats_path}")

    meal_stats_path = os.path.join(data_dir, "service_interval_meal_stats.csv")
    meal_stats.to_csv(meal_stats_path, index=False, encoding="utf-8-sig")
    print(f"  分餐段统计已保存: {meal_stats_path}")

    # Step 7: 生成图表
    print("\n[Step 6] 生成可视化图表...")

    print("  [1/4] 各窗口平均服务间隔柱状图...")
    plot_avg_service_interval(window_stats, output_dir)

    print("  [2/4] 服务间隔分布箱线图...")
    plot_service_interval_boxplot(df_peak, output_dir)

    print("  [3/4] 窗口×小时服务间隔热力图...")
    plot_service_interval_heatmap(hour_stats, valid_hours, output_dir)

    print("  [4/4] 午餐/晚餐服务间隔对比...")
    plot_meal_period_comparison(meal_stats, output_dir)

    print(f"\n服务间隔分析完成！图表已保存至: {output_dir}")
    return window_stats, meal_stats


if __name__ == "__main__":
    run_queue_analysis()
