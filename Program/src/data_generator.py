"""模拟数据生成器 - 生成单食堂多窗口交易记录

生成覆盖整个学期的食堂刷卡交易数据，包含：
- 交易时间（仅工作日午餐/晚餐时段）
- 脱敏卡号
- 窗口编号
- 消费金额
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


def get_chinese_holidays(year):
    """获取中国法定节假日（2025-2026学年常用节假日）

    Returns:
        set: 节假日日期字符串集合
    """
    holidays = set()

    # 2025年秋季学期节假日
    # 中秋节 2025
    holidays.update(["2025-10-06"])  # 补休
    # 国庆节 2025
    holidays.update([f"2025-10-{d}" for d in range(1, 8)])
    # 元旦 2026
    holidays.update(["2026-01-01", "2026-01-02", "2026-01-03"])

    # 寒假（假设1月17日开始）
    for d in pd.date_range("2026-01-17", "2026-02-16"):
        holidays.add(d.strftime("%Y-%m-%d"))

    return holidays


def generate_user_profiles(num_users, random_state):
    """生成用户消费行为画像

    每个用户分配一种行为类型，影响其消费频率和偏好

    Returns:
        dict: {user_id: {"type": str, "lunch_prob": float, "dinner_prob": float,
                         "preferred_windows": list, "price_offset": float}}
    """
    rng = random_state
    profiles = {}

    type_weights = list(USER_TYPE_WEIGHTS.values())
    type_names = list(USER_TYPE_WEIGHTS.keys())
    user_types = rng.choice(type_names, size=num_users, p=type_weights)

    window_ids = list(WINDOW_CONFIG.keys())

    for i, utype in enumerate(user_types):
        user_id = f"U{i + 1:04d}"

        if utype == "regular":
            # 规律型：午餐和晚餐都高概率出现
            lunch_prob = rng.uniform(0.75, 0.95)
            dinner_prob = rng.uniform(0.65, 0.85)
            preferred = rng.choice(window_ids, size=rng.randint(2, 4), replace=False).tolist()
            price_offset = rng.uniform(-0.05, 0.05)
        elif utype == "moderate":
            # 适中型：中等概率出现
            lunch_prob = rng.uniform(0.45, 0.70)
            dinner_prob = rng.uniform(0.35, 0.60)
            preferred = rng.choice(window_ids, size=rng.randint(3, 6), replace=False).tolist()
            price_offset = rng.uniform(-0.03, 0.08)
        elif utype == "occasional":
            # 偶尔型：低概率出现
            lunch_prob = rng.uniform(0.15, 0.35)
            dinner_prob = rng.uniform(0.10, 0.30)
            preferred = rng.choice(window_ids, size=rng.randint(3, 8), replace=False).tolist()
            price_offset = rng.uniform(-0.08, 0.03)
        else:  # high_spender
            # 高消费型：偏好高价位窗口
            lunch_prob = rng.uniform(0.55, 0.80)
            dinner_prob = rng.uniform(0.50, 0.75)
            preferred = ["W03", "W04", "W08"]  # 小炒、麻辣烫、特色档口
            price_offset = rng.uniform(0.10, 0.25)

        profiles[user_id] = {
            "type": utype,
            "lunch_prob": lunch_prob,
            "dinner_prob": dinner_prob,
            "preferred_windows": preferred,
            "price_offset": price_offset,
        }

    return profiles


def generate_traffic_multiplier(hour, meal_period):
    """生成某个时段内的客流波动系数

    模拟下课高峰、逐渐回落等真实模式

    Args:
        hour: 小时
        meal_period: "lunch" 或 "dinner"

    Returns:
        float: 客流倍率
    """
    period = MEAL_PERIODS[meal_period]
    start_h, end_h = period["start"], period["end"]
    mid_h = (start_h + end_h) / 2

    # 高峰在时段中段，两端较低
    # 用正态分布模拟：中间最高
    dist_from_mid = abs(hour + 0.5 - mid_h)
    half_range = (end_h - start_h) / 2
    if dist_from_mid < half_range * 0.3:
        return np.random.uniform(1.1, 1.4)  # 高峰期
    elif dist_from_mid < half_range * 0.7:
        return np.random.uniform(0.8, 1.1)  # 平峰
    else:
        return np.random.uniform(0.5, 0.8)  # 边缘时段


def generate_transactions_for_day(date, user_profiles, rng):
    """生成某一天的所有交易记录

    Args:
        date: datetime.date
        user_profiles: dict, 用户画像
        rng: numpy随机状态

    Returns:
        list of dict
    """
    transactions = []
    date_str = date.strftime("%Y-%m-%d")
    weekday = date.weekday()

    # 周一客流略高（攒了周末没吃），周五略低
    weekday_factor = {0: 1.10, 1: 1.00, 2: 1.00, 3: 0.98, 4: 0.92}.get(weekday, 1.0)

    for meal_key, meal_info in MEAL_PERIODS.items():
        for hour in range(meal_info["start"], meal_info["end"]):
            traffic_mult = generate_traffic_multiplier(hour, meal_key)

            for window_id, wconfig in WINDOW_CONFIG.items():
                # 计算该窗口该小时的基础客流量
                base_traffic = BASE_TRAFFIC_PER_WINDOW * wconfig["weight"]
                traffic = int(base_traffic * traffic_mult * weekday_factor)
                traffic = max(1, rng.poisson(traffic))

                # 随机选择用户（有消费概率）
                candidate_users = list(user_profiles.keys())
                rng.shuffle(candidate_users)

                count = 0
                for user_id in candidate_users:
                    if count >= traffic:
                        break

                    profile = user_profiles[user_id]
                    prob = profile["lunch_prob"] if meal_key == "lunch" else profile["dinner_prob"]

                    # 窗口偏好：首选窗口概率更高
                    if window_id in profile["preferred_windows"]:
                        prob *= 1.5

                    if rng.random() < prob * 0.3:  # 0.3避免所有人都出现
                        # 生成交易时间（精确到分钟秒）
                        minute = rng.randint(0, 60)
                        second = rng.randint(0, 60)
                        trans_time = f"{date_str} {hour:02d}:{minute:02d}:{second:02d}"

                        # 生成消费金额
                        low, high = wconfig["price_range"]
                        price = rng.uniform(low, high)
                        price = price * (1 + profile["price_offset"])
                        price = round(max(low * 0.8, min(high * 1.3, price)), 1)

                        transactions.append({
                            "transaction_id": f"T{len(transactions) + 1:08d}",
                            "card_id": user_id,
                            "window_id": window_id,
                            "window_name": wconfig["name"],
                            "transaction_time": trans_time,
                            "amount": price,
                            "meal_period": meal_info["label"],
                        })
                        count += 1

    return transactions


def generate_canteen_data(
    semester_start=SEMESTER_START,
    semester_end=SEMESTER_END,
    num_users=NUM_USERS,
    random_seed=RANDOM_SEED,
    output_dir=RAW_DIR,
):
    """生成完整学期食堂交易数据

    Args:
        semester_start: 学期开始日期
        semester_end: 学期结束日期
        num_users: 模拟用户数
        random_seed: 随机种子
        output_dir: 输出目录

    Returns:
        pd.DataFrame: 交易数据
    """
    rng = np.random.RandomState(random_seed)

    # 获取节假日
    holidays = get_chinese_holidays(None)

    # 生成用户画像
    print(f"正在生成 {num_users} 个用户画像...")
    user_profiles = generate_user_profiles(num_users, rng)

    # 生成交易日
    date_range = pd.date_range(semester_start, semester_end)
    workdays = [
        d.date()
        for d in date_range
        if d.weekday() in WORKDAYS and d.strftime("%Y-%m-%d") not in holidays
    ]
    print(f"学期内有效工作日: {len(workdays)} 天")

    # 逐日生成交易
    all_transactions = []
    for i, day in enumerate(workdays):
        if (i + 1) % 20 == 0:
            print(f"  已处理 {i + 1}/{len(workdays)} 天...")
        day_txns = generate_transactions_for_day(day, user_profiles, rng)
        all_transactions.extend(day_txns)

    # 构建DataFrame
    df = pd.DataFrame(all_transactions)

    # 重新编号transaction_id
    df["transaction_id"] = [f"T{i + 1:08d}" for i in range(len(df))]

    # 转换时间列
    df["transaction_time"] = pd.to_datetime(df["transaction_time"])
    df = df.sort_values("transaction_time").reset_index(drop=True)

    # 添加辅助列
    df["date"] = df["transaction_time"].dt.date
    df["hour"] = df["transaction_time"].dt.hour
    df["weekday"] = df["transaction_time"].dt.dayofweek
    df["weekday_name"] = df["transaction_time"].dt.day_name()

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "canteen_transactions.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n数据已保存至: {output_path}")
    print(f"总交易记录数: {len(df):,}")
    print(f"时间范围: {df['transaction_time'].min()} ~ {df['transaction_time'].max()}")
    print(f"独立用户数: {df['card_id'].nunique()}")
    print(f"窗口数: {df['window_id'].nunique()}")
    print(f"金额范围: {df['amount'].min():.1f} ~ {df['amount'].max():.1f} 元")
    print(f"平均客单价: {df['amount'].mean():.2f} 元")

    # 同时保存用户画像
    profiles_path = os.path.join(output_dir, "user_profiles.csv")
    profiles_df = pd.DataFrame([
        {"card_id": uid, **profile}
        for uid, profile in user_profiles.items()
    ])
    profiles_df.to_csv(profiles_path, index=False, encoding="utf-8-sig")
    print(f"用户画像已保存至: {profiles_path}")

    return df


if __name__ == "__main__":
    df = generate_canteen_data()
    print("\n===== 数据预览 =====")
    print(df.head(10))
    print("\n===== 各窗口统计 =====")
    print(df.groupby("window_name")["amount"].agg(["count", "mean", "sum"]).round(2))
