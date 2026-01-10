import numpy as np

"""
[配置模块] Settings
功能：集中管理项目的所有超参数、路径配置、物理常数及业务映射关系。
"""

# =========================================
# 1. 基础资源配置
# =========================================
VIDEO_PATH = "video/input_video.MP4"
TARGET_VIDEO_PATH = "video/output_video.MP4"
DB_PATH = "data/traffic_data.db" # 数据库路径
FPS = 30  # 视频源帧率 (用于时间积分)

# =========================================
# 2. 运动学估算参数 (Kinematics)
# =========================================
SPEED_WINDOW = 15  # 速度平滑窗口 (帧)，越大越平滑但滞后
ACCEL_WINDOW = 15  # 加速度计算窗口 (帧)

# =========================================
# 3. 数据门控与过滤 (Data Gating)
# =========================================
# 边缘缓冲区：距离图像边缘多少像素内的目标视为"未完全进入/正在离开"，不计算物理参数
BORDER_MARGIN = 20

# 预热帧数：新 ID 出现后的前几帧不输出数据 (等待卡尔曼滤波器收敛)
MIN_TRACKING_FRAMES = 10 

# 物理极限：忽略超过此值的异常加速度 (m/s^2)，用于过滤视觉追踪漂移
MAX_PHYSICAL_ACCEL = 6.0

# 存活时间阈值：生命周期小于此帧数的轨迹视为噪点 (路标/行人误检)，不产生离场报告
MIN_SURVIVAL_FRAMES = 30

# 车牌识别门控：车辆 Bounding Box 像素面积小于此值时，不进行 OCR (太小无法识别)
MIN_PLATE_AREA = 3000

# =========================================
# 4. MOVES 排放模型参数 (Emission Model)
# =========================================
# VSP 计算系数
VSP_COEFF_A = 1.1
VSP_COEFF_B = 0.132
VSP_COEFF_C = 0.000302

# 阈值定义
BRAKING_DECEL_THRESHOLD = -0.89  # m/s^2 (OpMode 0 判定阈值)
IDLING_SPEED_THRESHOLD = 0.45    # m/s (OpMode 1 判定阈值)
LOW_SPEED_THRESHOLD = 11.17      # m/s (25 mph, 用于区分低速/高速巡航)

# 新能源增重系数
MASS_FACTOR_EV = 1.25

# EPA MOVES 刹车磨损排放因子表 (单位: mg/s) ---
# 参考数据源: MOVES Default Brake Wear Emission Rates (PM2.5)
# OpMode 说明:
# 0: Braking (刹车，排放主力)
# 1: Idling (怠速，无磨损)
# 11: Coasting (滑行，微量磨损)
# 21: Cruise/Accel Low Speed (<25mph)
# 33: Cruise/Accel High Speed (>=25mph)

MOVES_BRAKE_WEAR_RATES = {
    'LDV': {  # 轻型车 (Light Duty)
        0:  8.25,   # 刹车时磨损显著
        1:  0.00,
        11: 0.05,   # 滑行时有轻微接触
        21: 0.12,   # 低速巡航/加速 (包含轻微修正刹车)
        33: 0.18    # 高速巡航 (空气阻力大，修正刹车稍多)
    },
    'HDV': {  # 重型车 (Heavy Duty) - 通常是 LDV 的 5-10 倍
        0:  65.40,  # 重车刹车排放极大
        1:  0.00,
        11: 0.40,
        21: 1.50,
        33: 2.20
    }
}

# =========================================
# 5. 车辆类型映射策略 (Mapping Strategy)
# =========================================
# 键格式: (YOLO_Class_ID, Plate_Color)
# YOLO IDs: 2=Car, 5=Bus, 7=Truck
TYPE_MAP = {
    # --- 蓝牌 (Blue): 轻型燃油车 ---
    (2, 'Blue'): "LDV-gasoline",
    (5, 'Blue'): "LDV-gasoline",   # 金杯/面包车 (YOLO常识别为Bus)
    (7, 'Blue'): "LDV-gasoline",   # 皮卡/轻卡 (YOLO常识别为Truck)

    # --- 黄牌 (Yellow): 重型柴油车 ---
    (2, 'Yellow'): "HDV-diesel",   # 罕见，保守归为重型
    (5, 'Yellow'): "HDV-diesel",   # 柴油大巴
    (7, 'Yellow'): "HDV-diesel",   # 重卡

    # --- 绿牌 (Green): 新能源车 ---
    (2, 'Green'): "LDV-electric",  # 纯电轿车
    (5, 'Green'): "HDV-electric",  # 电动公交
    (7, 'Green'): "HDV-electric",  # 电动渣土车/重卡

    # --- 兜底默认值 (Fallback) ---
    'Default_Car': "LDV-gasoline",
    'Default_Heavy': "HDV-electric" # [策略] 大车识别失败优先归为电车(尝试捕捉)或柴油(保守)
}

# =========================================
# 6. 调试与日志参数
# =========================================
# 是否启用控制台调试打印 (True: 打印离场报告; False: 静默运行)
DEBUG_MODE = True

# 最小存活帧数：小于此值的轨迹被视为误检噪声，离场时不打印报告
MIN_SURVIVAL_FRAMES = 30
