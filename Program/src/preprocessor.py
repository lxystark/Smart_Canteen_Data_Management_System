"""数据预处理与特征工程模块

对原始交易数据进行清洗、特征提取和聚合，为可视化和建模提供高质量数据
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


def load_raw_data(data_dir=RAW_DIR):
    """加载原始交易数据"""
    path = os.path.join(data_dir, "canteen_transactions.csv")
    df = pd.read_csv(path, parse_dates=["transaction_time"])
    print(f"已加载原始数据: {len(df):,} 条记录")
    return df


def clean_data(df):
    """数据清洗

    - 剔除异常金额（0元、超大额）
    - 剔除时间错乱记录
    - 处理重复记录
    """
    original_len = len(df)

    # 剔除金额异常
    df = df[df["amount"] > 0]
    # 剔除极端金额（超过窗口价格范围的2倍）
    price_limits = {wid: cfg["price_range"] for wid, cfg in WINDOW_CONFIG.items()}
    mask = df.apply(
        lambda row: price_limits.get(row["window_id"], (0, 100))[0] * 0.5
        <= row["amount"]
        <= price_limits.get(row["window_id"], (0, 100))[1] * 2.0,
        axis=1,
    )
    df = df[mask]

    # 剔除重复交易
    df = df.drop_duplicates(subset=["card_id", "window_id", "transaction_time"])

    # 确保时间列类型
    df["transaction_time"] = pd.to_datetime(df["transaction_time"])

    cleaned_len = len(df)
    print(f"数据清洗: {original_len:,} → {cleaned_len:,} (移除 {original_len - cleaned_len} 条)")
    return df.reset_index(drop=True)


def extract_time_features(df):
    """提取时间相关特征"""
    df = df.copy()
    dt = df["transaction_time"]

    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["hour"] = dt.dt.hour
    df["minute"] = dt.dt.minute
    df["weekday"] = dt.dt.dayofweek  # 0=周一, 6=周日
    df["weekday_name"] = dt.dt.day_name()
    df["is_monday"] = (df["weekday"] == 0).astype(int)
    df["is_friday"] = (df["weekday"] == 4).astype(int)
    df["week_of_semester"] = (
        (dt - pd.Timestamp(SEMESTER_START)).dt.days // 7 + 1
    )

    return df


def extract_behavior_features(df):
    """提取用户级行为特征（聚合到每个用户）

    Returns:
        pd.DataFrame: 每行一个用户，列为行为特征
    """
    # 基础统计
    user_stats = (
        df.groupby("card_id")
        .agg(
            total_transactions=("amount", "count"),
            total_spending=("amount", "sum"),
            avg_amount=("amount", "mean"),
            std_amount=("amount", "std"),
            min_amount=("amount", "min"),
            max_amount=("amount", "max"),
            active_days=("date", "nunique"),
            favorite_window=("window_id", lambda x: x.mode().iloc[0]),
            num_windows_used=("window_id", "nunique"),
        )
        .reset_index()
    )

    # 午餐/晚餐偏好
    meal_counts = df.groupby(["card_id", "meal_period"]).size().unstack(fill_value=0)
    meal_counts.columns = [f"meal_{c}" for c in meal_counts.columns]
    meal_counts = meal_counts.reset_index()

    # 时段偏好占比
    if "meal_午餐" in meal_counts.columns and "meal_晚餐" in meal_counts.columns:
        meal_counts["lunch_ratio"] = meal_counts["meal_午餐"] / (
            meal_counts["meal_午餐"] + meal_counts["meal_晚餐"]
        )
    else:
        meal_counts["lunch_ratio"] = 0.5

    # 每周平均消费次数
    user_stats["weekly_avg_transactions"] = (
        user_stats["total_transactions"] / user_stats["active_days"] * 5
    )
    user_stats["daily_avg_spending"] = (
        user_stats["total_spending"] / user_stats["active_days"]
    )

    # 合并
    user_features = user_stats.merge(meal_counts, on="card_id", how="left")

    # 填充NaN
    user_features = user_features.fillna(0)

    return user_features


def aggregate_daily_summary(df):
    """按日聚合：每日就餐人数和消费总额

    Returns:
        pd.DataFrame: 每行一天，含午餐/晚餐分时段统计
    """
    # 按日期+时段聚合
    daily = (
        df.groupby(["date", "meal_period"])
        .agg(
            transaction_count=("amount", "count"),
            total_amount=("amount", "sum"),
            avg_amount=("amount", "mean"),
            unique_users=("card_id", "nunique"),
        )
        .reset_index()
    )

    # 透视：午餐和晚餐各一列
    pivot = daily.pivot(
        index="date",
        columns="meal_period",
        values=["transaction_count", "total_amount", "unique_users"],
    ).fillna(0)

    # 展平列名
    pivot.columns = [f"{col[1]}_{col[0]}" for col in pivot.columns]
    daily_summary = pivot.reset_index()

    # 总计
    daily_summary["total_transaction_count"] = (
        daily_summary.get("午餐_transaction_count", 0)
        + daily_summary.get("晚餐_transaction_count", 0)
    )
    daily_summary["total_amount"] = (
        daily_summary.get("午餐_total_amount", 0)
        + daily_summary.get("晚餐_total_amount", 0)
    )
    daily_summary["total_unique_users"] = (
        daily_summary.get("午餐_unique_users", 0)
        + daily_summary.get("晚餐_unique_users", 0)
    )

    return daily_summary


def aggregate_hourly_summary(df):
    """按小时聚合：用于热力图等"""
    hourly = (
        df.groupby(["weekday", "hour", "window_id"])
        .agg(
            avg_transactions=("amount", "count"),
            avg_amount=("amount", "mean"),
        )
        .reset_index()
    )

    # 按周几和小时取平均（跨所有周）
    hourly_avg = (
        hourly.groupby(["weekday", "hour", "window_id"])
        .agg(
            avg_transactions=("avg_transactions", "mean"),
            avg_amount=("avg_amount", "mean"),
        )
        .reset_index()
    )

    return hourly_avg


def preprocess_pipeline(output_dir=PROCESSED_DIR):
    """完整预处理流水线

    Returns:
        tuple: (cleaned_df, user_features, daily_summary, hourly_summary)
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载原始数据
    df = load_raw_data()

    # 2. 清洗
    df = clean_data(df)

    # 3. 时间特征
    df = extract_time_features(df)

    # 4. 用户行为特征
    print("正在提取用户行为特征...")
    user_features = extract_behavior_features(df)

    # 5. 日聚合
    print("正在聚合日级数据...")
    daily_summary = aggregate_daily_summary(df)

    # 6. 小时聚合
    print("正在聚合小时级数据...")
    hourly_summary = aggregate_hourly_summary(df)

    # 保存
    df.to_csv(os.path.join(output_dir, "cleaned_data.csv"), index=False, encoding="utf-8-sig")
    user_features.to_csv(os.path.join(output_dir, "user_features.csv"), index=False, encoding="utf-8-sig")
    daily_summary.to_csv(os.path.join(output_dir, "daily_summary.csv"), index=False, encoding="utf-8-sig")
    hourly_summary.to_csv(os.path.join(output_dir, "hourly_summary.csv"), index=False, encoding="utf-8-sig")

    print(f"\n预处理完成，所有文件已保存至: {output_dir}")
    print(f"  清洗后数据: {len(df):,} 条")
    print(f"  用户特征: {len(user_features)} 个用户, {len(user_features.columns)} 个特征")
    print(f"  日级汇总: {len(daily_summary)} 天")
    print(f"  小时级汇总: {len(hourly_summary)} 条")

    return df, user_features, daily_summary, hourly_summary


if __name__ == "__main__":
    df, user_features, daily_summary, hourly_summary = preprocess_pipeline()
    print("\n===== 用户特征预览 =====")
    print(user_features.head())
    print("\n===== 日级汇总预览 =====")
    print(daily_summary.head())
