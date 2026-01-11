import json
import os
import sys

"""
[配置模块] Settings
功能：负责读取外部 config.json 配置文件，并将其解析为 Python 原生数据类型。
      作为全局参数的单一真值来源 (Source of Truth)。
"""

# 配置文件路径
CONFIG_FILE = "config.json"

# =========================================
# 0. 配置加载 (Config Loading)
# =========================================
try:
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"配置文件 '{CONFIG_FILE}' 未找到，请确保其位于项目根目录。")
        
    with open(CONFIG_FILE, "r", encoding='utf-8') as f:
        _cfg = json.load(f)
        
except Exception as e:
    print(f"[Error] 配置文件加载失败: {e}")
    sys.exit(1)

# =========================================
# 1. 基础资源配置 (System Resources)
# =========================================
VIDEO_PATH = _cfg["system"]["video_path"]
TARGET_VIDEO_PATH = _cfg["system"]["target_video_path"]
DB_PATH = _cfg["system"]["db_path"]
FPS = _cfg["system"]["fps"]
DEBUG_MODE = _cfg["system"]["debug_mode"]

# =========================================
# 2. 功能开关 (Feature Switches)
# =========================================
ENABLE_MOTION = _cfg["switches"]["enable_motion"]      # 是否开启运动感知 (速度/加速度)
ENABLE_OCR = _cfg["switches"]["enable_ocr"]            # 是否开启车牌识别
ENABLE_EMISSION = _cfg["switches"]["enable_emission"]  # 是否开启排放计算

# =========================================
# 3. 运动学参数 (Kinematics)
# =========================================
_k = _cfg["kinematics"]
SPEED_WINDOW = _k["speed_window"]
ACCEL_WINDOW = _k["accel_window"]
BORDER_MARGIN = _k["border_margin"]
MIN_TRACKING_FRAMES = _k["min_tracking_frames"]
MAX_PHYSICAL_ACCEL = _k["max_physical_accel"]
MIN_SURVIVAL_FRAMES = _k["min_survival_frames"]

# =========================================
# 4. 车牌识别参数 (OCR Params)
# =========================================
_o = _cfg["ocr_params"]
MIN_PLATE_AREA = _o["min_plate_area"]
OCR_RETRY_COOLDOWN = _o["retry_cooldown"]  # OCR 重试冷却 (帧)
OCR_INTERVAL = _o["run_interval"]          # OCR 运行间隔 (每 N 帧运行一次)
OCR_CONF_THRESHOLD = _o["confidence_threshold"] # 结果置信度阈值

# =========================================
# 5. 排放模型参数 (Emission Model)
# =========================================
_e = _cfg["emission_params"]
VSP_COEFF_A = _e["vsp_coeff_a"]
VSP_COEFF_B = _e["vsp_coeff_b"]
VSP_COEFF_C = _e["vsp_coeff_c"]
BRAKING_DECEL_THRESHOLD = _e["braking_decel_threshold"]
IDLING_SPEED_THRESHOLD = _e["idling_speed_threshold"]
LOW_SPEED_THRESHOLD = _e["low_speed_threshold"]
MASS_FACTOR_EV = _e["mass_factor_ev"]

# =========================================
# 6. 车型识别参数 (YOLO Class)
# =========================================
YOLO_CLASS_CAR = 2
YOLO_CLASS_BUS = 5
YOLO_CLASS_TRUCK = 7
YOLO_INTEREST_CLASSES = [YOLO_CLASS_CAR, YOLO_CLASS_BUS, YOLO_CLASS_TRUCK]

# [数据转换] MOVES 排放因子表
# 说明: JSON 中 Key 必须为字符串，此处需转换回 int 类型作为 OpMode ID
MOVES_BRAKE_WEAR_RATES = {}
for cat, rates in _cfg["moves_brake_wear_rates"].items():
    MOVES_BRAKE_WEAR_RATES[cat] = {int(k): v for k, v in rates.items()}

# [数据转换] 车辆类型映射策略
# 说明: 将 "2,Blue" 格式的字符串键转换为 Python 元组 (2, 'Blue')
TYPE_MAP = {}
for k, v in _cfg["type_map"].items():
    if "," in k:
        parts = k.split(",")
        # 尝试转换 Class ID (兼容字符串和整数)
        try:
            class_id = int(parts[0])
        except ValueError:
            class_id = parts[0]
        color = parts[1]
        TYPE_MAP[(class_id, color)] = v
    else:
        # 处理 Default_Car 等非元组键
        TYPE_MAP[k] = v
