"""消费行为聚类模块

基于用户消费行为特征，使用K-Means对消费者进行分群画像。

分析流程：
1. 加载用户特征数据
2. 选取聚类特征并标准化
3. 使用肘部法则和轮廓系数确定最优K值
4. 执行K-Means聚类
5. 输出群体画像（雷达图）和评估指标
6. 保存聚类结果
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from src.visualizer import setup_plot_style, save_figure


# ============ 聚类特征选择 ============
# 用于聚类的数值型特征（排除card_id、favorite_window等非数值/非核心特征）
CLUSTER_FEATURES = [
    "total_transactions",    # 总消费次数
    "total_spending",        # 总消费金额
    "avg_amount",            # 平均客单价
    "active_days",           # 活跃天数
    "num_windows_used",      # 使用的窗口数
    "weekly_avg_transactions",  # 周均消费次数
    "daily_avg_spending",    # 日均消费额
    "lunch_ratio",           # 午餐偏好比
]

# 聚类标签映射（根据聚类中心自动判定）
CLUSTER_LABELS = {
    0: "待定",
    1: "待定",
    2: "待定",
    3: "待定",
}


def load_user_features(data_dir=PROCESSED_DIR):
    """加载用户特征数据"""
    path = os.path.join(data_dir, "user_features.csv")
    df = pd.read_csv(path)
    print(f"已加载用户特征: {len(df)} 个用户, {len(df.columns)} 个特征")
    return df


def prepare_features(df, features=CLUSTER_FEATURES):
    """准备聚类特征矩阵

    - 选取指定特征列
    - 处理缺失值和无穷大
    - 标准化
    """
    # 校验特征列是否存在
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"缺少聚类特征列: {missing}")

    X = df[features].copy()

    # 处理无穷大和NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    nan_count = X.isna().any(axis=1).sum()
    if nan_count > 0:
        print(f"  警告: {nan_count} 条记录含缺失值，用中位数填充")
        X = X.fillna(X.median())

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=features, index=df.index)

    print(f"  聚类特征: {features}")
    print(f"  标准化后矩阵: {X_scaled.shape}")

    return X_scaled, scaler, features


def find_optimal_k(X_scaled, k_range=range(2, 7), random_state=RANDOM_SEED):
    """使用肘部法则和轮廓系数确定最优K值

    Returns:
        dict: {"inertia": [...], "silhouette": [...], "k_values": list}
    """
    inertias = []
    silhouettes = []
    k_values = list(k_range)

    print("\n  K值选择评估:")
    print(f"  {'K':<5} {'惯性(Inertia)':<18} {'轮廓系数':<12}")
    print("  " + "-" * 35)

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=300)
        labels = km.fit_predict(X_scaled)
        inertia = km.inertia_
        sil = silhouette_score(X_scaled, labels)

        inertias.append(inertia)
        silhouettes.append(sil)
        print(f"  {k:<5} {inertia:<18.1f} {sil:<12.4f}")

    return {
        "k_values": k_values,
        "inertia": inertias,
        "silhouette": silhouettes,
    }


def perform_kmeans(X_scaled, n_clusters=4, random_state=RANDOM_SEED):
    """执行K-Means聚类

    Returns:
        tuple: (labels, model, centers_original)
    """
    km = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=300,
    )
    labels = km.fit_predict(X_scaled)

    sil_score = silhouette_score(X_scaled, labels)
    print(f"\n  K-Means聚类结果 (K={n_clusters}):")
    print(f"    轮廓系数: {sil_score:.4f}")
    print(f"    各群体分布:")
    for c in range(n_clusters):
        count = (labels == c).sum()
        pct = count / len(labels) * 100
        print(f"      群体{c}: {count}人 ({pct:.1f}%)")

    return labels, km


def assign_cluster_labels(labels, centers_df):
    """根据聚类中心特征自动分配群体标签

    根据各群体在各维度上的相对高低来命名
    """
    # 计算各群体特征均值在所有群体中的排名
    ranks = centers_df.rank(ascending=False)

    label_map = {}
    for cluster_id in centers_df.index:
        row = centers_df.loc[cluster_id]
        rank = ranks.loc[cluster_id]

        # 判断逻辑
        if row["weekly_avg_transactions"] > centers_df["weekly_avg_transactions"].median() and \
           row["active_days"] > centers_df["active_days"].median():
            if row["avg_amount"] > centers_df["avg_amount"].median():
                label_map[cluster_id] = "规律高消费型"
            else:
                label_map[cluster_id] = "规律经济型"
        elif row["avg_amount"] > centers_df["avg_amount"].quantile(0.75):
            label_map[cluster_id] = "高消费型"
        elif row["active_days"] < centers_df["active_days"].median() and \
             row["total_transactions"] < centers_df["total_transactions"].median():
            label_map[cluster_id] = "偶尔就餐型"
        else:
            label_map[cluster_id] = "普通消费型"

    return label_map


def compute_cluster_profiles(df, labels, features=CLUSTER_FEATURES):
    """计算各群体画像特征均值"""
    df = df.copy()
    df["cluster"] = labels

    # 各群体的特征均值
    profile = df.groupby("cluster")[features].mean()

    # 各群体人数
    counts = df.groupby("cluster").size().rename("count")
    profile = profile.join(counts)

    return profile


# ============================================================
# 图表1: 肘部法则 & 轮廓系数图
# ============================================================
def plot_k_evaluation(eval_results, output_dir=FIGURE_DIR):
    """K值评估图：肘部法则 + 轮廓系数"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    k_values = eval_results["k_values"]

    # 肘部法则
    axes[0].plot(k_values, eval_results["inertia"], "bo-", linewidth=2, markersize=8)
    axes[0].set_xlabel("K (聚类数)", fontsize=12)
    axes[0].set_ylabel("惯性 (Inertia)", fontsize=12)
    axes[0].set_title("肘部法则", fontsize=13, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    # 轮廓系数
    axes[1].plot(k_values, eval_results["silhouette"], "rs-", linewidth=2, markersize=8)
    axes[1].set_xlabel("K (聚类数)", fontsize=12)
    axes[1].set_ylabel("轮廓系数", fontsize=12)
    axes[1].set_title("轮廓系数法", fontsize=13, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    # 标注最优K
    best_k_idx = np.argmax(eval_results["silhouette"])
    best_k = k_values[best_k_idx]
    axes[1].annotate(
        f"最优K={best_k}",
        xy=(best_k, eval_results["silhouette"][best_k_idx]),
        xytext=(best_k + 0.5, eval_results["silhouette"][best_k_idx] - 0.03),
        arrowprops=dict(arrowstyle="->", color="#E74C3C"),
        fontsize=11, color="#E74C3C", fontweight="bold",
    )

    plt.tight_layout()
    return save_figure(fig, "12_k_evaluation", output_dir)


# ============================================================
# 图表2: 群体画像雷达图
# ============================================================
def plot_radar_chart(profile, features, label_map, output_dir=FIGURE_DIR):
    """各群体画像雷达图"""
    # 标准化到0-1范围用于雷达图
    profile_norm = profile[features].copy()
    for col in features:
        min_val = profile_norm[col].min()
        max_val = profile_norm[col].max()
        if max_val > min_val:
            profile_norm[col] = (profile_norm[col] - min_val) / (max_val - min_val)
        else:
            profile_norm[col] = 0.5

    n_features = len(features)
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    n_clusters = len(profile_norm)
    colors = sns.color_palette(COLOR_PALETTE, n_clusters)

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    for i, (cluster_id, row) in enumerate(profile_norm.iterrows()):
        values = row.values.tolist()
        values += values[:1]  # 闭合
        label = label_map.get(cluster_id, f"群体{cluster_id}")
        ax.plot(angles, values, "o-", linewidth=2, markersize=6,
                color=colors[i], label=label)
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    # 特征标签
    feature_labels = [
        "总消费次数", "总消费额", "平均客单价", "活跃天数",
        "窗口多样性", "周均消费", "日均消费", "午餐偏好"
    ]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_labels, fontsize=11)
    ax.set_title(f"{CANTEEN_NAME} 消费者群体画像雷达图", fontsize=14,
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)

    plt.tight_layout()
    return save_figure(fig, "13_cluster_radar", output_dir)


# ============================================================
# 图表3: 聚类散点图 (前两个主成分)
# ============================================================
def plot_cluster_scatter(X_scaled, labels, label_map, output_dir=FIGURE_DIR):
    """聚类结果散点图（基于前两个标准化特征的PCA投影）"""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)

    n_clusters = len(set(labels))
    colors = sns.color_palette(COLOR_PALETTE, n_clusters)

    fig, ax = plt.subplots(figsize=(10, 8))

    for c in range(n_clusters):
        mask = labels == c
        label = label_map.get(c, f"群体{c}")
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=[colors[c]], label=label, alpha=0.6, s=30, edgecolors="white", linewidth=0.3
        )

    ax.set_xlabel(f"主成分1 (方差解释: {pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12)
    ax.set_ylabel(f"主成分2 (方差解释: {pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12)
    ax.set_title(f"{CANTEEN_NAME} 消费者聚类散点图 (PCA降维)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return save_figure(fig, "14_cluster_scatter", output_dir)


# ============================================================
# 图表4: 群体特征对比热力图
# ============================================================
def plot_cluster_heatmap(profile, features, label_map, output_dir=FIGURE_DIR):
    """各群体特征标准化后的热力图"""
    # 标准化到0-1
    profile_norm = profile[features].copy()
    for col in features:
        min_val = profile_norm[col].min()
        max_val = profile_norm[col].max()
        if max_val > min_val:
            profile_norm[col] = (profile_norm[col] - min_val) / (max_val - min_val)
        else:
            profile_norm[col] = 0.5

    # 行标签
    row_labels = [f"{label_map.get(i, f'群体{i}')} (n={int(profile.loc[i, 'count'])})"
                  for i in profile_norm.index]

    feature_labels = [
        "总消费次数", "总消费额", "平均客单价", "活跃天数",
        "窗口多样性", "周均消费", "日均消费", "午餐偏好"
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        profile_norm, annot=True, fmt=".2f", cmap="YlOrRd",
        linewidths=0.5, ax=ax,
        yticklabels=row_labels,
        xticklabels=feature_labels,
        cbar_kws={"label": "标准化值 (0-1)"}
    )
    ax.set_title(f"{CANTEEN_NAME} 各群体特征对比热力图", fontsize=14, fontweight="bold")

    plt.tight_layout()
    return save_figure(fig, "15_cluster_heatmap", output_dir)


# ============================================================
# 图表5: 轮廓系数分布图
# ============================================================
def plot_silhouette(X_scaled, labels, label_map, output_dir=FIGURE_DIR):
    """各群体轮廓系数分布图"""
    n_clusters = len(set(labels))
    sil_values = silhouette_samples(X_scaled, labels)
    colors = sns.color_palette(COLOR_PALETTE, n_clusters)

    fig, ax = plt.subplots(figsize=(10, 6))

    y_lower = 10
    for c in range(n_clusters):
        cluster_sil = sil_values[labels == c]
        cluster_sil.sort()
        size = len(cluster_sil)
        y_upper = y_lower + size

        ax.fill_betweenx(
            np.arange(y_lower, y_upper), 0, cluster_sil,
            alpha=0.7, color=colors[c]
        )
        label = label_map.get(c, f"群体{c}")
        ax.text(-0.05, y_lower + size / 2, label, fontsize=10)
        y_lower = y_upper + 10

    sil_avg = silhouette_score(X_scaled, labels)
    ax.axvline(sil_avg, color="#E74C3C", linestyle="--", linewidth=1.5,
               label=f"平均轮廓系数: {sil_avg:.3f}")
    ax.set_xlabel("轮廓系数", fontsize=12)
    ax.set_ylabel("样本", fontsize=12)
    ax.set_title(f"{CANTEEN_NAME} 各群体轮廓系数分布", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)

    plt.tight_layout()
    return save_figure(fig, "16_silhouette", output_dir)


def run_clustering(data_dir=PROCESSED_DIR, output_dir=FIGURE_DIR, n_clusters=4):
    """执行完整聚类分析流程

    Args:
        data_dir: 数据目录
        output_dir: 图表输出目录
        n_clusters: 聚类数（默认4，可通过K评估调整）
    """
    setup_plot_style()
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 50)
    print("消费行为聚类分析")
    print("=" * 50)

    # Step 1: 加载数据
    print("\n[Step 1] 加载用户特征数据...")
    df = load_user_features(data_dir)

    # Step 2: 特征准备
    print("\n[Step 2] 准备聚类特征...")
    X_scaled, scaler, features = prepare_features(df)

    # Step 3: K值评估
    print("\n[Step 3] 评估最优K值...")
    eval_results = find_optimal_k(X_scaled)

    # 如果未指定，使用轮廓系数最优的K
    best_k_idx = np.argmax(eval_results["silhouette"])
    suggested_k = eval_results["k_values"][best_k_idx]
    if n_clusters is None:
        n_clusters = suggested_k
    print(f"\n  轮廓系数最优K={suggested_k}, 使用K={n_clusters}")

    # Step 4: 执行聚类
    print("\n[Step 4] 执行K-Means聚类...")
    labels, model = perform_kmeans(X_scaled, n_clusters=n_clusters)

    # Step 5: 群体画像
    print("\n[Step 5] 计算群体画像...")
    profile = compute_cluster_profiles(df, labels, features)
    label_map = assign_cluster_labels(labels, profile[features])

    print("\n  群体画像标签:")
    for c, label in label_map.items():
        count = int(profile.loc[c, "count"])
        print(f"    群体{c} → {label} ({count}人)")

    # 保存聚类结果
    df_result = df.copy()
    df_result["cluster"] = labels
    df_result["cluster_label"] = df_result["cluster"].map(label_map)

    result_path = os.path.join(data_dir, "user_clusters.csv")
    df_result.to_csv(result_path, index=False, encoding="utf-8-sig")
    print(f"\n  聚类结果已保存: {result_path}")

    profile_path = os.path.join(data_dir, "cluster_profiles.csv")
    profile_out = profile.copy()
    profile_out.index = [label_map.get(i, f"群体{i}") for i in profile_out.index]
    profile_out.to_csv(profile_path, encoding="utf-8-sig")
    print(f"  群体画像已保存: {profile_path}")

    # Step 6: 生成图表
    print("\n[Step 6] 生成可视化图表...")

    print("  [1/5] K值评估图...")
    plot_k_evaluation(eval_results, output_dir)

    print("  [2/5] 群体画像雷达图...")
    plot_radar_chart(profile, features, label_map, output_dir)

    print("  [3/5] 聚类散点图...")
    plot_cluster_scatter(X_scaled, labels, label_map, output_dir)

    print("  [4/5] 群体特征热力图...")
    plot_cluster_heatmap(profile, features, label_map, output_dir)

    print("  [5/5] 轮廓系数分布图...")
    plot_silhouette(X_scaled, labels, label_map, output_dir)

    print(f"\n聚类分析完成！图表已保存至: {output_dir}")
    return df_result, profile, label_map


if __name__ == "__main__":
    run_clustering()
