"""智慧食堂数据管理系统 - 全局配置"""

import os

# ============ 项目路径 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

# ============ 食堂基础配置 ============
CANTEEN_NAME = "第一食堂"
NUM_WINDOWS = 8  # 窗口数量

# 窗口配置：(窗口名称, 最低价, 最高价, 日均客流权重)
# 客单价范围参考高校食堂实际情况
WINDOW_CONFIG = {
    "W01": {"name": "家常菜",   "price_range": (8, 15),  "weight": 1.2},
    "W02": {"name": "面食",     "price_range": (7, 12),  "weight": 1.0},
    "W03": {"name": "小炒",     "price_range": (12, 22), "weight": 0.8},
    "W04": {"name": "麻辣烫",   "price_range": (10, 20), "weight": 1.1},
    "W05": {"name": "饺子",     "price_range": (8, 14),  "weight": 0.9},
    "W06": {"name": "盖浇饭",   "price_range": (9, 16),  "weight": 1.3},
    "W07": {"name": "粥饼",     "price_range": (5, 10),  "weight": 0.6},
    "W08": {"name": "特色档口", "price_range": (14, 25), "weight": 0.7},
}

# ============ 时间配置 ============
# 学期时间范围（2025年秋季学期示例）
SEMESTER_START = "2025-09-01"
SEMESTER_END = "2026-01-16"

# 用餐时段（24小时制）
MEAL_PERIODS = {
    "lunch": {"start": 11, "end": 13, "label": "午餐"},
    "dinner": {"start": 17, "end": 19, "label": "晚餐"},
}

# 工作日（周一=0, 周日=6）
WORKDAYS = [0, 1, 2, 3, 4]  # 周一到周五

# ============ 数据生成配置 ============
NUM_USERS = 500  # 模拟用户（卡号）数量
RANDOM_SEED = 42  # 随机种子，确保可复现

# 每个时段每个窗口的基础客流（会乘以权重和随机波动）
BASE_TRAFFIC_PER_WINDOW = 30

# 用户消费行为类型比例
USER_TYPE_WEIGHTS = {
    "regular": 0.40,     # 规律三餐型：几乎每天固定时段就餐
    "moderate": 0.30,    # 适中型：经常但非每天
    "occasional": 0.20,  # 偶尔型：随机出现
    "high_spender": 0.10 # 高消费型：偏好高价位窗口
}

# ============ 可视化配置 ============
FIGURE_DPI = 150
FIGURE_FORMAT = "png"
COLOR_PALETTE = "Set2"

# 中文字体配置（matplotlib显示中文）
FONT_CONFIG = {
    "font_family": "SimHei",  # 黑体，Windows自带
    "axes_unicode_minus": False,
}
