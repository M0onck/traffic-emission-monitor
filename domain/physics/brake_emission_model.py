import numpy as np
from domain.physics.opmode_calculator import MovesOpModeCalculator

class BrakeEmissionModel:
    """
    [业务层] 刹车磨损排放模型 (Brake Wear Emission Model)
    ===========================================================================
    
    【建模背景】
    ---------------------------------------------------------------------------
    刹车磨损是城市交通非尾气排放 (Non-exhaust Emissions) 的主要来源之一。
    本模型基于美国环保署 (EPA) MOVES 模型的方法论，通过车辆比功率 (VSP) 
    和操作工况 (OpMode) 来估算刹车颗粒物 (PM2.5/PM10) 的排放量。

    【核心逻辑】
    ---------------------------------------------------------------------------
    Emission = Base_EF(OpMode, VehicleType) * Correction_Factor
    
    1. VSP 计算 (Vehicle Specific Power):
       计算车辆单位质量的瞬时功率需求，涵盖动能变化、势能变化(坡度)、
       滚动阻力和空气阻力。
       
    2. OpMode 判定 (Operating Mode):
       利用注入的 opmode_calculator，根据 V, a, VSP 将车辆状态映射为
       标准的 MOVES Bin (如 Braking, Idling, Cruising)。
       
    3. 排放因子查表 (Look-up Table):
       根据 OpMode 和车型 (Car/Bus/Truck)，查询基准排放率 (mg/s)。
       *注: 刹车排放主要发生在 OpMode 0 (Braking) 和部分高负荷减速区。
       
    4. 新能源修正 (EV Correction):
       针对电动车/混动车，考虑再生制动 (Regenerative Braking) 对摩擦制动的
       替代效应，以及电池增重带来的负面效应。
    ===========================================================================
    """

    def __init__(self, config: dict):
        """
        初始化模型，加载参数并注入 OpMode 计算器。
        
        :param config: 包含排放参数的字典 (对应 config.json)
        """
        # --- 1. 基础物理/环境参数 ---
        # 道路坡度 (%)，正值代表上坡
        self.road_grade_percent = config.get("road_grade_percent", 0.0)
        # 质量因子 (EV相对于燃油车的增重比例，默认 1.25倍)
        self.mass_factor_ev = config.get("mass_factor_ev", 1.25)

        # --- 2. 查表数据 ---
        self.moves_rates = config.get("moves_brake_wear_rates", {})
                
        # --- 3. 依赖注入: OpMode 计算器 ---
        # 直接将 config 传给计算器，让其自行提取阈值 (braking_threshold 等)
        self.opmode_calculator = MovesOpModeCalculator(config)
        
        # --- 4. 内部常量 (YOLO Class ID) ---
        self.YOLO_CLASS_CAR = 2
        self.YOLO_CLASS_BUS = 5
        self.YOLO_CLASS_TRUCK = 7

    def _get_emission_factor(self, op_mode: int, vehicle_category: str) -> float:
        """根据工况和车型查询基准排放率 (mg/s)"""
        # 默认回退到 CAR
        rates = self.moves_rates.get(vehicle_category, self.moves_rates.get('CAR', {}))
        # 确保 op_mode 是 int (JSON key 可能是 str)
        return rates.get(str(op_mode), rates.get(op_mode, 0.0))

    def process(self, kinematics_data: dict, detections, plate_cache: dict, vehicle_classifier, vsp_map: dict) -> dict:
        """
        执行批处理计算
        
        :param kinematics_data: 运动学数据 {tid: {'speed':..., 'accel':...}}
        :param detections: YOLO 检测结果 (用于获取 class_id)
        :param plate_cache: 车牌颜色缓存
        :param vehicle_classifier: 车辆类型分类器实例
        :param vsp_map: VSP 字典
        :return: 包含 VSP, OpMode, Emission 的结果字典
        """
        results = {}
        # 构建 TID -> ClassID 映射
        id_to_class = {tid: cid for tid, cid in zip(detections.tracker_id, detections.class_id)}

        for tid, data in kinematics_data.items():
            # 1. 准备基础数据
            class_id = int(id_to_class.get(tid, self.YOLO_CLASS_CAR))
            plate_color = plate_cache.get(tid, "Unknown")
            v = data['speed']
            a = data['accel']
            
            # 2. 解析详细车型 (用于 EV 修正)
            _, type_str = vehicle_classifier.resolve_type(class_id, plate_color_override=plate_color)
            
            # 3. 计算 VSP (核心物理量)
            vsp = vsp_map.get(tid, 0.0)
            
            # 4. 调用 OpMode 计算器
            op_mode = self.opmode_calculator.get_opmode(v, a, vsp)
            
            # 5. 确定 MOVES 车型大类 (CAR/BUS/TRUCK)
            category = 'CAR'
            if class_id == self.YOLO_CLASS_BUS: category = 'BUS'
            elif class_id == self.YOLO_CLASS_TRUCK: category = 'TRUCK'
            
            # 6. 查表获取基准排放
            base_emission = self._get_emission_factor(op_mode, category)
            
            # 7. 应用修正因子 (EV Correction)
            final_factor = 1.0
            if "electric" in type_str:
                # 负面: 电池增重导致物理惯性增大
                mass_penalty = self.mass_factor_ev
                # 正面: 再生制动 (Regen) 减少摩擦制动的使用
                # OpMode 0 (急刹): Regen 贡献有限，主要靠摩擦 (系数 0.4)
                # 其他 (滑行): Regen 贡献大 (系数 0.1)
                regen_factor = 0.4 if op_mode == 0 else 0.1
                final_factor = mass_penalty * regen_factor
            
            emission_rate = base_emission * final_factor

            debug_info = {
                "v_ms": round(v, 2),
                "a_ms2": round(a, 2),
                "vsp": round(vsp, 2),
                "op_mode": op_mode,
                "base_rate_mg_s": base_emission, # 查表得到的基准值
                "is_ev": "electric" in type_str,
                "ev_factor": round(final_factor, 2), # 最终乘数
                "mass_penalty": self.mass_factor_ev if "electric" in type_str else 1.0,
                "regen_factor": regen_factor if "electric" in type_str else 1.0
            }

            results[tid] = {
                **data,
                "vsp": vsp,
                "op_mode": op_mode,
                "emission_rate": emission_rate,
                "type_str": type_str,
                "plate_color": plate_color,
                "debug_info": debug_info
            }
            
        return results
