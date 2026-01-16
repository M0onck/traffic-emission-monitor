import numpy as np
from domain.physics.opmode_calculator import MovesOpModeCalculator

class BrakeEmissionModel:
    """
    [业务层] 刹车磨损排放模型 (查表法)
    """
    def __init__(self, config: dict):
        self.road_grade_percent = config.get("road_grade_percent", 0.0)
        self.mass_factor_ev = config.get("mass_factor_ev", 1.25)
        self.rates_map = config.get("brake_wear_coefficients", {})
        self.opmode_calculator = MovesOpModeCalculator(config)
        self.YOLO_CLASS_BUS = 5
        self.YOLO_CLASS_TRUCK = 7

    def _get_emission_factor(self, op_mode: int, vehicle_category: str) -> float:
        rates = self.rates_map.get(vehicle_category, self.rates_map.get('CAR', {}))
        return float(rates.get(op_mode, 0.0))

    def calculate_single_point(self, v_ms, a_ms2, vsp, vehicle_class_id, dt, type_str="") -> dict:
        # OpMode 计算包含加速工况 (35, 37)
        op_mode = self.opmode_calculator.get_opmode(v_ms, a_ms2, vsp)
        
        category = 'CAR'
        if vehicle_class_id == self.YOLO_CLASS_BUS: category = 'BUS'
        elif vehicle_class_id == self.YOLO_CLASS_TRUCK: category = 'TRUCK'
        
        # 查表 (对于 35/37 工况，配置表中系数为 0.0)
        base_emission = self._get_emission_factor(op_mode, category)
        
        final_factor = 1.0
        is_electric = "electric" in type_str
        if is_electric:
            # OpMode 0 (Braking) 时 Regen 贡献较小，其他工况贡献较大
            regen_factor = 0.4 if op_mode == 0 else 0.1
            final_factor = self.mass_factor_ev * regen_factor
        
        emission_rate = base_emission * final_factor
        emission_mass = emission_rate * dt

        return {
            "emission_mass": emission_mass,
            "emission_rate": emission_rate,
            "op_mode": op_mode,
            "debug_info": {
                "dt": dt,
                "op_mode": op_mode,
                "base_rate": base_emission,
                "is_ev": is_electric
            }
        }
        
    def process(self, kinematics_data, detections, plate_cache, vehicle_classifier, vsp_map, dt):
        results = {}
        id_to_class = {tid: cid for tid, cid in zip(detections.tracker_id, detections.class_id)}
        for tid, data in kinematics_data.items():
            class_id = int(id_to_class.get(tid, 2))
            plate_color = plate_cache.get(tid, "Unknown")
            _, type_str = vehicle_classifier.resolve_type(class_id, plate_color_override=plate_color)
            
            res = self.calculate_single_point(
                data['speed'], data['accel'], vsp_map.get(tid, 0.0), 
                class_id, dt, type_str
            )
            results[tid] = {
                **data, "vsp": vsp_map.get(tid, 0.0),
                "op_mode": res["op_mode"],
                "emission_rate": res["emission_rate"],
                "type_str": type_str,
                "plate_color": plate_color
            }
        return results
