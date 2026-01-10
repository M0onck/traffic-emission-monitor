import config.settings as cfg

class BrakeEmissionModel:
    """
    [业务] 刹车磨损排放模型 (Based on MOVES)
    """
    def __init__(self):
        pass

    def _determine_vehicle_type(self, yolo_class_id, plate_color):
        # ... (保留原有的类型判定逻辑) ...
        key = (yolo_class_id, plate_color)
        if key in cfg.TYPE_MAP: return cfg.TYPE_MAP[key]
        if yolo_class_id == 2: return cfg.TYPE_MAP['Default_Car']
        return "HDV-electric" # 大车兜底

    def _determine_opmode(self, v, a, vsp):
        """
        判定 MOVES 运行模态 (扩展版)
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
            return 21 # Low Speed
        else:
            return 33 # High Speed

    def _get_emission_factor(self, op_mode, vehicle_category):
        """
        根据车型(LDV/HDV)和工况查表
        """
        rates = cfg.MOVES_BRAKE_WEAR_RATES.get(vehicle_category, cfg.MOVES_BRAKE_WEAR_RATES['LDV'])
        return rates.get(op_mode, 0.0)

    def process(self, kinematics_data, detections, plate_cache):
        results = {}
        id_to_class = {tid: cid for tid, cid in zip(detections.tracker_id, detections.class_id)}

        for tid, data in kinematics_data.items():
            class_id = int(id_to_class.get(tid, 2))
            plate_color = plate_cache.get(tid, "Unknown")
            
            # 1. 类型与物理修正
            type_str = self._determine_vehicle_type(class_id, plate_color)
            mass_correction = cfg.MASS_FACTOR_EV if "electric" in type_str else 1.0
            
            # 2. VSP 计算
            v, a = data['speed'], data['accel']
            adjusted_A = cfg.VSP_COEFF_A * mass_correction
            vsp = v * (adjusted_A * a + cfg.VSP_COEFF_B) + cfg.VSP_COEFF_C * (v**3)
            
            # 3. OpMode 判定 (升级版)
            op_mode = self._determine_opmode(v, a, vsp)
            
            # 4. 排放因子查表
            # 判断是大车还是小车
            category = 'HDV' if 'HDV' in type_str else 'LDV'
            emission_rate = self._get_emission_factor(op_mode, category)
            
            # EV 动能回收修正 (仅在刹车工况 OpMode 0 生效)
            # 假设 EV 的机械刹车使用率减少 60%
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
