"""消费预测模块

基于历史消费数据，预测未来时段的就餐人数和消费总额。

模型体系：
1. 基线模型：Prophet（自动捕获周期性、节假日效应）
2. 进阶模型：XGBoost（利用多特征进行回归预测）

评估指标：MAE、RMSE、MAPE
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from src.visualizer import setup_plot_style, save_figure

# 抑制Prophet等库的警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ============ 预测参数 ============
FORECAST_HORIZON = 14     # 预测天数
TEST_RATIO = 0.15         # 测试集比例
RANDOM_SEED_VAL = RANDOM_SEED


def load_daily_data(data_dir=PROCESSED_DIR):
    """加载日级汇总数据，准备预测用时间序列"""
    path = os.path.join(data_dir, "daily_summary.csv")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 确保目标列存在
    if "total_transaction_count" not in df.columns:
        # 尝试从午餐+晚餐列合并
        lunch_col = [c for c in df.columns if "午餐" in c and "transaction_count" in c]
        dinner_col = [c for c in df.columns if "晚餐" in c and "transaction_count" in c]
        if lunch_col and dinner_col:
            df["total_transaction_count"] = df[lunch_col[0]].fillna(0) + df[dinner_col[0]].fillna(0)
        else:
            raise ValueError("无法确定总交易笔数列")

    if "total_amount" not in df.columns:
        lunch_amt = [c for c in df.columns if "午餐" in c and "total_amount" in c]
        dinner_amt = [c for c in df.columns if "晚餐" in c and "total_amount" in c]
        if lunch_amt and dinner_amt:
            df["total_amount"] = df[lunch_amt[0]].fillna(0) + df[dinner_amt[0]].fillna(0)

    print(f"已加载日级数据: {len(df)} 天, 日期范围 {df['date'].min().date()} ~ {df['date'].max().date()}")
    return df


def split_train_test(df, test_ratio=TEST_RATIO):
    """按时间顺序划分训练集和测试集"""
    n_test = max(1, int(len(df) * test_ratio))
    train = df.iloc[:-n_test].copy()
    test = df.iloc[-n_test:].copy()
    print(f"训练集: {len(train)} 天, 测试集: {len(test)} 天")
    return train, test


def compute_metrics(y_true, y_pred, name="模型"):
    """计算评估指标"""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    # MAPE: 避免除零
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = float("inf")

    print(f"  {name} - MAE: {mae:.1f}, RMSE: {rmse:.1f}, MAPE: {mape:.1f}%")
    return {"name": name, "MAE": mae, "RMSE": rmse, "MAPE": mape}


# ============================================================
# 基线模型: ARIMA (statsmodels)
# ============================================================
def run_arima(train, test, target_col="total_transaction_count"):
    """使用ARIMA进行时间序列预测

    自动确定(p,d,q)参数，使用AIC准则选优
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
        import itertools
    except ImportError:
        print("  statsmodels未安装，跳过ARIMA基线模型")
        return None

    y_train = train[target_col].values

    # 简单参数搜索
    best_aic = float("inf")
    best_order = (1, 1, 1)
    p_range = range(0, 4)
    d_range = range(0, 2)
    q_range = range(0, 4)

    print("  ARIMA参数搜索中...")
    for p, d, q in itertools.product(p_range, d_range, q_range):
        if p == 0 and d == 0 and q == 0:
            continue
        try:
            model = ARIMA(y_train, order=(p, d, q))
            result = model.fit()
            if result.aic < best_aic:
                best_aic = result.aic
                best_order = (p, d, q)
        except Exception:
            continue

    print(f"  最优ARIMA参数: {best_order}, AIC: {best_aic:.1f}")

    # 使用最优参数拟合
    model = ARIMA(y_train, order=best_order)
    fitted = model.fit()

    # 预测测试期
    n_test = len(test)
    forecast = fitted.forecast(steps=n_test)
    y_pred = np.array(forecast)

    # 评估
    y_true = test[target_col].values
    min_len = min(len(y_true), len(y_pred))
    metrics = compute_metrics(y_true[:min_len], y_pred[:min_len], name="ARIMA")

    # 生成未来FORECAST_HORIZEN天的预测
    full_series = np.concatenate([y_train, y_true])
    model_full = ARIMA(full_series, order=best_order)
    fitted_full = model_full.fit()
    future_forecast = fitted_full.forecast(steps=FORECAST_HORIZON)

    return {
        "fitted": fitted,
        "forecast_test": y_pred,
        "future_forecast": future_forecast,
        "best_order": best_order,
        "metrics": metrics,
        "train_values": y_train,
    }


# ============================================================
# 进阶模型: XGBoost
# ============================================================
def create_features(df):
    """为XGBoost创建时间特征（需在 train+test 合并后调用以确保滞后特征有效）"""
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    dt = df["date"]

    df["dayofweek"] = dt.dt.dayofweek
    df["dayofmonth"] = dt.dt.day
    df["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    df["is_monday"] = (dt.dt.dayofweek == 0).astype(int)
    df["is_friday"] = (dt.dt.dayofweek == 4).astype(int)

    # 滞后特征：基于日级时间序列整体shift，保证lag_1等有实际值
    for lag in [1, 2, 5, 7]:
        col = f"lag_{lag}"
        df[col] = df["total_transaction_count"].shift(lag)

    # 滚动统计
    df["rolling_mean_3"] = df["total_transaction_count"].rolling(3, min_periods=1).mean()
    df["rolling_mean_7"] = df["total_transaction_count"].rolling(7, min_periods=1).mean()
    df["rolling_std_7"] = df["total_transaction_count"].rolling(7, min_periods=1).std()

    return df


def run_xgboost(train, test, target_col="total_transaction_count"):
    """使用XGBoost进行回归预测"""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("  XGBoost未安装，跳过进阶模型")
        return None

    # 合并 train+test 后统一创建特征，确保滞后特征对 test 集有效
    train["_split"] = "train"
    test["_split"] = "test"
    combined = pd.concat([train, test], ignore_index=True).sort_values("date").reset_index(drop=True)
    combined_fe = create_features(combined)

    train_fe = combined_fe[combined_fe["_split"] == "train"].copy()
    test_fe = combined_fe[combined_fe["_split"] == "test"].copy()

    feature_cols = [
        "dayofweek", "dayofmonth", "weekofyear", "is_monday", "is_friday",
        "lag_1", "lag_2", "lag_5", "lag_7",
        "rolling_mean_3", "rolling_mean_7", "rolling_std_7",
    ]

    # 去除NaN
    train_clean = train_fe.dropna(subset=feature_cols + [target_col])
    test_clean = test_fe.dropna(subset=feature_cols + [target_col])

    if len(train_clean) < 10 or len(test_clean) < 2:
        print("  XGBoost: 训练或测试数据不足，跳过")
        return None

    X_train = train_clean[feature_cols]
    y_train = train_clean[target_col]
    X_test = test_clean[feature_cols]
    y_test = test_clean[target_col]

    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        random_state=RANDOM_SEED_VAL,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test.values, y_pred, name="XGBoost")

    # 特征重要性
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return {
        "model": model,
        "metrics": metrics,
        "importance": importance,
        "feature_cols": feature_cols,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "test_dates": test_clean["date"],
    }


# ============================================================
# 图表1: ARIMA预测对比图
# ============================================================
def plot_arima_forecast(arima_result, train, test, target_col, output_dir=FIGURE_DIR):
    """ARIMA预测结果与实际值对比"""
    if arima_result is None:
        return None

    target_label = "日均就餐人次" if "transaction" in target_col else "日均消费额"

    fig, ax = plt.subplots(figsize=(14, 6))

    # 训练集实际值
    ax.plot(train["date"], train[target_col], "-", color="#3498DB",
            linewidth=1, alpha=0.7, label="训练数据")

    # 测试集实际值
    ax.plot(test["date"], test[target_col], "o-", color="#2C3E50",
            markersize=5, linewidth=1.5, label="实际值")

    # ARIMA预测
    y_pred = arima_result["forecast_test"]
    ax.plot(test["date"].values[:len(y_pred)], y_pred, "s-",
            color="#E74C3C", markersize=5, linewidth=1.5, label="ARIMA预测")

    # 未来预测
    future_dates = pd.date_range(
        start=test["date"].max() + pd.Timedelta(days=1),
        periods=FORECAST_HORIZON, freq="B"  # 工作日
    )
    future_pred = arima_result["future_forecast"]
    if len(future_pred) > 0 and len(future_dates) > 0:
        min_len = min(len(future_dates), len(future_pred))
        ax.plot(future_dates[:min_len], future_pred[:min_len], "--",
                color="#27AE60", linewidth=1.5, label=f"未来{FORECAST_HORIZON}天预测")

    ax.set_xlabel("日期", fontsize=12)
    ax.set_ylabel(target_label, fontsize=12)
    ax.set_title(f"{CANTEEN_NAME} ARIMA预测 - {target_label} (order={arima_result['best_order']})",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()
    return save_figure(fig, "17_arima_forecast", output_dir)


# ============================================================
# 图表2: XGBoost预测对比图
# ============================================================
def plot_xgboost_forecast(xgb_result, target_col, output_dir=FIGURE_DIR):
    """XGBoost预测结果与实际值对比"""
    if xgb_result is None:
        return None

    target_label = "日均就餐人次" if "transaction" in target_col else "日均消费额"

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(xgb_result["test_dates"], xgb_result["y_test"].values, "o-",
            color="#2C3E50", markersize=4, linewidth=1.5, label="实际值")
    ax.plot(xgb_result["test_dates"], xgb_result["y_pred"], "s-",
            color="#E67E22", markersize=4, linewidth=1.5, label="XGBoost预测")

    ax.set_xlabel("日期", fontsize=12)
    ax.set_ylabel(target_label, fontsize=12)
    ax.set_title(f"{CANTEEN_NAME} XGBoost预测 - {target_label}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()
    return save_figure(fig, "18_xgboost_forecast", output_dir)


# ============================================================
# 图表3: 模型对比评估图
# ============================================================
def plot_model_comparison(all_metrics, output_dir=FIGURE_DIR):
    """模型性能对比柱状图"""
    if not all_metrics:
        return None

    metrics_df = pd.DataFrame(all_metrics)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metric_names = ["MAE", "RMSE", "MAPE"]
    metric_labels = ["MAE (绝对误差)", "RMSE (均方根误差)", "MAPE (%)"]

    for ax, metric, label in zip(axes, metric_names, metric_labels):
        bars = ax.bar(metrics_df["name"], metrics_df[metric],
                      color=sns.color_palette(COLOR_PALETTE, len(metrics_df)))
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold")
        for bar, val in zip(bars, metrics_df[metric]):
            fmt = ".1f" if metric != "MAPE" else ".1f"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:{fmt}}", ha="center", fontsize=10)

    plt.suptitle(f"{CANTEEN_NAME} 预测模型性能对比", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return save_figure(fig, "19_model_comparison", output_dir)


# ============================================================
# 图表4: XGBoost特征重要性
# ============================================================
def plot_feature_importance(xgb_result, output_dir=FIGURE_DIR):
    """XGBoost特征重要性条形图"""
    if xgb_result is None:
        return None

    importance = xgb_result["importance"].head(10)

    # 特征名中文映射
    name_map = {
        "dayofweek": "星期几", "dayofmonth": "月中第几天", "weekofyear": "年中第几周",
        "is_monday": "是否周一", "is_friday": "是否周五",
        "lag_1": "前1天值", "lag_2": "前2天值", "lag_5": "前5天值", "lag_7": "前7天值",
        "rolling_mean_3": "3日滚动均值", "rolling_mean_7": "7日滚动均值",
        "rolling_std_7": "7日滚动标准差",
    }
    importance["feature_cn"] = importance["feature"].map(lambda x: name_map.get(x, x))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(importance["feature_cn"], importance["importance"],
                   color=sns.color_palette(COLOR_PALETTE, len(importance)))

    ax.set_xlabel("特征重要性", fontsize=12)
    ax.set_title(f"{CANTEEN_NAME} XGBoost特征重要性 Top10", fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    plt.tight_layout()
    return save_figure(fig, "20_feature_importance", output_dir)


def run_prediction(data_dir=PROCESSED_DIR, output_dir=FIGURE_DIR):
    """执行完整预测分析流程"""
    setup_plot_style()
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 50)
    print("消费预测分析")
    print("=" * 50)

    # Step 1: 加载数据
    print("\n[Step 1] 加载日级数据...")
    df = load_daily_data(data_dir)

    # Step 2: 划分训练/测试集
    print("\n[Step 2] 划分训练/测试集...")
    train, test = split_train_test(df)

    all_metrics = []

    # Step 3: ARIMA基线
    print("\n[Step 3] ARIMA基线模型...")
    arima_result = run_arima(train, test, target_col="total_transaction_count")
    if arima_result:
        all_metrics.append(arima_result["metrics"])

    # Step 4: XGBoost进阶
    print("\n[Step 4] XGBoost进阶模型...")
    xgb_result = run_xgboost(train, test, target_col="total_transaction_count")
    if xgb_result:
        all_metrics.append(xgb_result["metrics"])

    # Step 5: 保存评估结果
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_path = os.path.join(data_dir, "prediction_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
        print(f"\n  评估指标已保存: {metrics_path}")

    # Step 6: 生成图表
    print("\n[Step 5] 生成可视化图表...")

    print("  [1/4] ARIMA预测对比图...")
    plot_arima_forecast(arima_result, train, test, "total_transaction_count", output_dir)

    print("  [2/4] XGBoost预测对比图...")
    plot_xgboost_forecast(xgb_result, "total_transaction_count", output_dir)

    print("  [3/4] 模型对比评估图...")
    plot_model_comparison(all_metrics, output_dir)

    print("  [4/4] XGBoost特征重要性...")
    plot_feature_importance(xgb_result, output_dir)

    print(f"\n预测分析完成！图表已保存至: {output_dir}")
    return arima_result, xgb_result


if __name__ == "__main__":
    run_prediction()
