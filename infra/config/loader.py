import json
import os
import sys

"""
[基础层] 配置文件读取器
功能：负责读取外部 config.json 配置文件，并将其解析为 Python 原生数据类型。
"""

# 配置文件路径
CONFIG_FILE = "config.json"

# ==============================================================================
# Phase 0: 配置文件加载
# ==============================================================================
try:
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"配置文件 '{CONFIG_FILE}' 未找到")
        
    with open(CONFIG_FILE, "r", encoding='utf-8') as f:
        _cfg = json.load(f)
        
except Exception as e:
    print(f"[Settings] Error: 配置文件加载失败 - {e}")
    sys.exit(1)

# ==============================================================================
# Phase 1: 基础参数提取
# ==============================================================================

# 1.1 系统资源
_sys = _cfg["system"]
VIDEO_PATH = _sys["video_path"]
TARGET_VIDEO_PATH = _sys["target_video_path"]
DB_PATH = _sys["db_path"]
FPS = _sys["fps"]
DEBUG_MODE = _sys["debug_mode"]

# 1.2 功能开关
_sw = _cfg["switches"]
ENABLE_MOTION = _sw["enable_motion"]
ENABLE_OCR = _sw["enable_ocr"]
ENABLE_EMISSION = _sw["enable_emission"]

# 1.3 算法参数
_k = _cfg["kinematics"]
SPEED_WINDOW = _k.get("speed_window", 15)  # 支持读取分离的窗口参数
ACCEL_WINDOW = _k.get("accel_window", 30)
BORDER_MARGIN = _k["border_margin"]
MIN_TRACKING_FRAMES = _k["min_tracking_frames"]
MAX_PHYSICAL_ACCEL = _k["max_physical_accel"]
MIN_SURVIVAL_FRAMES = _k["min_survival_frames"]
EXIT_THRESHOLD = _k["exit_threshold"]
KINEMATICS_POLY_ORDER = _k.get("poly_order", 3)

_o = _cfg["ocr_params"]
MIN_PLATE_AREA = _o["min_plate_area"]
OCR_RETRY_COOLDOWN = _o["retry_cooldown"]
OCR_INTERVAL = _o["run_interval"]
OCR_CONF_THRESHOLD = _o["confidence_threshold"]

# 1.4 排放模型阈值
_e = _cfg["emission_params"]
BRAKING_DECEL_THRESHOLD = _e["braking_decel_threshold"]
IDLING_SPEED_THRESHOLD = _e["idling_speed_threshold"]
LOW_SPEED_THRESHOLD = _e["low_speed_threshold"]
MASS_FACTOR_EV = _e["mass_factor_ev"]
ROAD_GRADE_PERCENT = _e["road_grade_percent"]

# ==============================================================================
# Phase 2: 核心常量定义
# ==============================================================================
_y = _cfg["yolo_classes"]
YOLO_CLASS_CAR = _y["yolo_class_car"]
YOLO_CLASS_BUS = _y["yolo_class_bus"]
YOLO_CLASS_TRUCK = _y["yolo_class_truck"]
YOLO_INTEREST_CLASSES = [YOLO_CLASS_CAR, YOLO_CLASS_BUS, YOLO_CLASS_TRUCK]

VEHICLE_SEMANTIC_MAP = {
    "car": YOLO_CLASS_CAR,
    "bus": YOLO_CLASS_BUS,
    "truck": YOLO_CLASS_TRUCK
}

# ==============================================================================
# Phase 3: 复杂数据结构构建
# ==============================================================================

# 3.1 VSP 系数表
VSP_COEFFS = {}
_vsp_raw = _cfg.get("vsp_coefficients", {})
VSP_COEFFS["default"] = _vsp_raw.get("default", {"a_m": 0.156, "b_m": 0.002, "c_m": 0.0005})

for sem_key, coeff_data in _vsp_raw.items():
    if sem_key in VEHICLE_SEMANTIC_MAP:
        cls_id = VEHICLE_SEMANTIC_MAP[sem_key]
        VSP_COEFFS[cls_id] = coeff_data

# 3.2 MOVES 刹车排放因子表
MOVES_BRAKE_WEAR_RATES = {}
for cat, rates in _cfg["moves_brake_wear_rates"].items():
    if not isinstance(rates, dict): continue
    MOVES_BRAKE_WEAR_RATES[cat] = {int(op_mode): val for op_mode, val in rates.items()}

# 3.3 MOVES 轮胎排放因子表
TIRE_WEAR_RATES = {}
_tire_raw = _cfg.get("tire_wear_rates", {})

for cat, rates in _tire_raw.items():
    if not isinstance(rates, dict): # 跳过 _meta, _modes 等说明字段
        continue
    # OpMode key 可能是 "0", "35" 等字符串，统一转为 int
    TIRE_WEAR_RATES[cat] = {int(op_mode): val for op_mode, val in rates.items()}

# 3.4 车辆类型映射表
TYPE_MAP = {}
_type_map_raw = _cfg.get("type_map", {})

for key_str, type_val in _type_map_raw.items():
    if "," in key_str:
        try:
            sem_type, color = key_str.split(",")
            sem_type = sem_type.strip()
            color = color.strip()
            if sem_type in VEHICLE_SEMANTIC_MAP:
                cls_id = VEHICLE_SEMANTIC_MAP[sem_type]
                TYPE_MAP[(cls_id, color)] = type_val
        except ValueError:
            pass
    else:
        TYPE_MAP[key_str] = type_val

# ==============================================================================
# Phase 4: 兼容性保留
# ==============================================================================
# 保留 TIRE_WEAR_MODEL 变量以防旧代码引用，但不再用于新模型
TIRE_WEAR_MODEL = _cfg.get("tire_wear_model", {})
