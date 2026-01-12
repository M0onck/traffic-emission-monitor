import numpy as np

class BrakeEmissionModel:
    """
    [业务] 刹车磨损排放模型 (Based on MOVES)
    已重构：支持依赖注入
    """
    def __init__(self, config: dict):
        """
        :param config: 包含排放参数的字典 (通常对应 settings.py 中的 emission_params + vsp/moves 表)
        """
        # 基础阈值
        self.braking_decel_threshold = config.get("braking_decel_threshold", -0.89)
        self.idling_speed_threshold = config.get("idling_speed_threshold", 0.45)
        self.low_speed_threshold = config.get("low_speed_threshold", 11.17)
        self.mass_factor_ev = config.get("mass_factor_ev", 1.25)
        self.road_grade_percent = config.get("road_grade_percent", 0.0)
        
        # 查表数据
        self.moves_rates = config.get("moves_brake_wear_rates", {})
        self.vsp_coeffs = config.get("vsp_coefficients", {})
        
        # 常量定义
        self.YOLO_CLASS_CAR = 2
        self.YOLO_CLASS_BUS = 5
        self.YOLO_CLASS_TRUCK = 7

    def _determine_opmode(self, v, a, vsp):
        if a <= self.braking_decel_threshold: return 0
        if v < self.idling_speed_threshold: return 1
        if vsp < 0: return 11
        if v < self.low_speed_threshold: return 21 
        return 33 

    def _get_emission_factor(self, op_mode, vehicle_category):
        # 默认回退到 CAR
        rates = self.moves_rates.get(vehicle_category, self.moves_rates.get('CAR', {}))
        return rates.get(op_mode, 0.0)

    def _calculate_vsp(self, v, a, class_id):
        # 查找系数，回退到 default
        coeffs = self.vsp_coeffs.get(class_id, self.vsp_coeffs.get("default", 
                 {"a_m": 0.156, "b_m": 0.002, "c_m": 0.0005}))
        
        drag_rolling = coeffs["a_m"] * v + coeffs["b_m"] * (v**2) + coeffs["c_m"] * (v**3)
        grade_term = 9.81 * (self.road_grade_percent / 100.0)
        inertial_work = v * (1.1 * a + grade_term)
        return drag_rolling + inertial_work

    def process(self, kinematics_data, detections, plate_cache, vehicle_classifier):
        """
        :param vehicle_classifier: 传入 Classifier 类或实例用于解析类型
        """
        results = {}
        id_to_class = {tid: cid for tid, cid in zip(detections.tracker_id, detections.class_id)}

        for tid, data in kinematics_data.items():
            class_id = int(id_to_class.get(tid, self.YOLO_CLASS_CAR))
            plate_color = plate_cache.get(tid, "Unknown")
            
            # 使用传入的 classifier 解析类型
            _, type_str = vehicle_classifier.resolve_type(class_id, plate_color_override=plate_color)
            
            v, a = data['speed'], data['accel']
            vsp = self._calculate_vsp(v, a, class_id)
            op_mode = self._determine_opmode(v, a, vsp)
            
            category = 'CAR'
            if class_id == self.YOLO_CLASS_BUS: category = 'BUS'
            elif class_id == self.YOLO_CLASS_TRUCK: category = 'TRUCK'
            
            base_emission = self._get_emission_factor(op_mode, category)
            
            # EV 修正逻辑
            final_factor = 1.0
            if "electric" in type_str:
                mass_penalty = self.mass_factor_ev
                regen_factor = 0.4 if op_mode == 0 else 0.1
                final_factor = mass_penalty * regen_factor
            
            emission_rate = base_emission * final_factor

            results[tid] = {
                **data,
                "vsp": vsp,
                "op_mode": op_mode,
                "emission_rate": emission_rate,
                "type_str": type_str,
                "plate_color": plate_color
            }
        return results
