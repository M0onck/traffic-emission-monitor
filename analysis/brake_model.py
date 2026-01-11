import config.settings as cfg
from core.classifier import VehicleClassifier

class BrakeEmissionModel:
    """
    [业务] 刹车磨损排放模型 (Based on MOVES)
    """
    def __init__(self):
        pass

    def _determine_opmode(self, v, a, vsp):
        """
        判定 MOVES 运行模态
        """
        # Bin 0: Braking (高排放)
        if a <= cfg.BRAKING_DECEL_THRESHOLD:
            return 0
        # Bin 1: Idling (零排放)
        if v < cfg.IDLING_SPEED_THRESHOLD:
            return 1
        # Bin 11: Coasting (滑行, VSP < 0)
        if vsp < 0:
            return 11
        # Bin 21/33: Cruise/Accel (根据速度区分)
        if v < cfg.LOW_SPEED_THRESHOLD:
            return 21 
        else:
            return 33 

    def _get_emission_factor(self, op_mode, vehicle_category):
        rates = cfg.MOVES_BRAKE_WEAR_RATES.get(vehicle_category, cfg.MOVES_BRAKE_WEAR_RATES['LDV'])
        return rates.get(op_mode, 0.0)

    def process(self, kinematics_data, detections, plate_cache):
        results = {}
        id_to_class = {tid: cid for tid, cid in zip(detections.tracker_id, detections.class_id)}

        for tid, data in kinematics_data.items():
            class_id = int(id_to_class.get(tid, cfg.YOLO_CLASS_CAR))
            
            # 如果 OCR 关闭或无记录，plate_color 为 Unknown
            plate_color = plate_cache.get(tid, "Unknown")
            
            # 1. 调用公共分类器进行类型判定
            _, type_str = VehicleClassifier.resolve_type(class_id, plate_color_override=plate_color)
            
            # 2. 物理参数修正
            # 只有明确标记为 electric 的才应用 EV 质量修正
            mass_correction = cfg.MASS_FACTOR_EV if "electric" in type_str else 1.0
            
            v, a = data['speed'], data['accel']
            adjusted_A = cfg.VSP_COEFF_A * mass_correction
            vsp = v * (adjusted_A * a + cfg.VSP_COEFF_B) + cfg.VSP_COEFF_C * (v**3)
            
            # 3. OpMode 判定
            op_mode = self._determine_opmode(v, a, vsp)
            
            # 4. 排放因子查表
            category = 'HDV' if 'HDV' in type_str else 'LDV'
            emission_rate = self._get_emission_factor(op_mode, category)
            
            # EV 动能回收修正 (仅在 OpMode 0 生效)
            if "electric" in type_str and op_mode == 0:
                emission_rate *= 0.4

            results[tid] = {
                **data,
                "vsp": vsp,
                "op_mode": op_mode,
                "emission_rate": emission_rate,
                "type_str": type_str,
                "plate_color": plate_color
            }
        return results
