import json
import os
import sys

"""
[配置模块] Settings
功能：负责读取外部 config.json 配置文件，并将其解析为 Python 原生数据类型。
      作为全局参数的单一真值来源 (Source of Truth)。

架构说明：
1. 基础资源与开关：直接读取 JSON 值。
2. YOLO 类别定义：定义系统识别的车辆 ID 常量。
3. 动态映射构建：基于 YOLO ID，解析 VSP 系数和车辆类型映射表，
   将配置文件中的语义字符串 (如 "car") 转换为程序内部使用的整数 ID。
"""

# 配置文件路径
CONFIG_FILE = "config.json"

# ==============================================================================
# Phase 0: 配置文件加载 (Config Loading)
# ==============================================================================
try:
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"配置文件 '{CONFIG_FILE}' 未找到，请确保其位于项目根目录。")
        
    with open(CONFIG_FILE, "r", encoding='utf-8') as f:
        _cfg = json.load(f)
        
except Exception as e:
    print(f"[Settings] Error: 配置文件加载失败 - {e}")
    sys.exit(1)

# ==============================================================================
# Phase 1: 基础参数提取 (Basic Parameters)
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
ENABLE_MOTION = _sw["enable_motion"]      # 是否开启运动感知
ENABLE_OCR = _sw["enable_ocr"]            # 是否开启车牌识别
ENABLE_EMISSION = _sw["enable_emission"]  # 是否开启排放计算

# 1.3 算法参数 (运动学 & OCR)
_k = _cfg["kinematics"]
SPEED_WINDOW = _k["speed_window"]
ACCEL_WINDOW = _k["accel_window"]
BORDER_MARGIN = _k["border_margin"]
MIN_TRACKING_FRAMES = _k["min_tracking_frames"]
MAX_PHYSICAL_ACCEL = _k["max_physical_accel"]
MIN_SURVIVAL_FRAMES = _k["min_survival_frames"]

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
# Phase 2: 核心常量定义 (Core Constants)
# ==============================================================================

# 2.1 YOLO 类别定义
# 这是整个系统的分类基准，从 config.json 中读取 ID
_y = _cfg["yolo_classes"]
YOLO_CLASS_CAR = _y["yolo_class_car"]
YOLO_CLASS_BUS = _y["yolo_class_bus"]
YOLO_CLASS_TRUCK = _y["yolo_class_truck"]

# 定义感兴趣的类别列表 (用于过滤检测结果)
YOLO_INTEREST_CLASSES = [YOLO_CLASS_CAR, YOLO_CLASS_BUS, YOLO_CLASS_TRUCK]

# 2.2 语义映射表 (Semantic to ID Map)
# 核心逻辑：定义配置文件中 "单词" 到 "数字ID" 的权威映射。
# 后续所有涉及 "car", "bus" 的解析都必须依赖此表，确保 ID 统一。
VEHICLE_SEMANTIC_MAP = {
    "car": YOLO_CLASS_CAR,
    "bus": YOLO_CLASS_BUS,
    "truck": YOLO_CLASS_TRUCK
}

# ==============================================================================
# Phase 3: 复杂数据结构构建 (Complex Structures)
# ==============================================================================

# 3.1 VSP 系数表 (VSP Coefficients)
# 输入: {"car": {...}, "bus": {...}}
# 输出: {2: {...}, 5: {...}}
VSP_COEFFS = {}
_vsp_raw = _cfg.get("vsp_coefficients", {})

# 默认系数兜底
VSP_COEFFS["default"] = _vsp_raw.get("default", {"a_m": 0.156, "b_m": 0.002, "c_m": 0.0005})

for sem_key, coeff_data in _vsp_raw.items():
    # 使用语义映射表转换 key
    if sem_key in VEHICLE_SEMANTIC_MAP:
        cls_id = VEHICLE_SEMANTIC_MAP[sem_key]
        VSP_COEFFS[cls_id] = coeff_data

# 3.2 MOVES 排放因子表
# 说明: JSON 中的 OpMode 键是字符串 ("0", "1")，需转换为整数 (0, 1) 以便查表
MOVES_BRAKE_WEAR_RATES = {}
for cat, rates in _cfg["moves_brake_wear_rates"].items():
    # cat 保持为 "CAR", "TRUCK", "BUS" 字符串，与 brake_model.py 中的 category 对应
    MOVES_BRAKE_WEAR_RATES[cat] = {int(op_mode): val for op_mode, val in rates.items()}

# 3.3 车辆类型映射表 (Type Map)
# 输入: "car,Blue": "Car-gasoline"
# 输出: (2, "Blue"): "Car-gasoline"
TYPE_MAP = {}
_type_map_raw = _cfg.get("type_map", {})

for key_str, type_val in _type_map_raw.items():
    if "," in key_str:
        # 处理组合键: "语义类型,颜色"
        try:
            sem_type, color = key_str.split(",")
            sem_type = sem_type.strip() # 去除可能存在的空格
            color = color.strip()
            
            # 处理语义键 (不支持数字硬编码)
            if sem_type in VEHICLE_SEMANTIC_MAP:
                cls_id = VEHICLE_SEMANTIC_MAP[sem_type]
                TYPE_MAP[(cls_id, color)] = type_val
            else:
                print(f"[Settings] Warning: type_map 中存在未定义的车型键 '{sem_type}'，已忽略。")
                
        except ValueError:
            print(f"[Settings] Warning: type_map 键格式错误 '{key_str}'，应为 'Type,Color'。")
    else:
        # 处理普通键 (如 "Default_Car", "Default_Heavy")
        TYPE_MAP[key_str] = type_val
