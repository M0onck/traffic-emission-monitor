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
        判定 MOVES 运行模态 (OpMode)
        """
        # Bin 0: Braking (高排放, 减速度超过阈值)
        if a <= cfg.BRAKING_DECEL_THRESHOLD:
            return 0
        # Bin 1: Idling (怠速, 零排放)
        if v < cfg.IDLING_SPEED_THRESHOLD:
            return 1
        # Bin 11: Coasting (滑行, VSP < 0 且未急刹)
        if vsp < 0:
            return 11
        # Bin 21/33: Cruise/Accel (巡航或加速, 根据速度区分)
        if v < cfg.LOW_SPEED_THRESHOLD:
            return 21 
        else:
            return 33 

    def _get_emission_factor(self, op_mode, vehicle_category):
        """查表获取基础排放因子"""
        rates = cfg.MOVES_BRAKE_WEAR_RATES.get(vehicle_category, cfg.MOVES_BRAKE_WEAR_RATES['LDV'])
        return rates.get(op_mode, 0.0)

    def _calculate_vsp(self, v, a, class_id):
        """
        基于物理分项的动态 VSP 计算
        公式: VSP = (A/M)v + (B/M)v^2 + (C/M)v^3 + v(1.1a + g*grade)
        """
        # 1. 查表获取该车型的物理系数 (A/M, B/M, C/M)
        # 如果 class_id 不在配置中，回退到 'default' (通常是轿车参数)
        coeffs = cfg.VSP_COEFFS.get(class_id, cfg.VSP_COEFFS["default"])
        
        # 2. 计算各项载荷
        # 滚动阻力 + 旋转损耗 + 空气阻力
        drag_rolling = coeffs["a_m"] * v + coeffs["b_m"] * (v**2) + coeffs["c_m"] * (v**3)
        
        # 3. 坡度项 (grade_percent / 100) -> sin(theta)
        grade_term = 9.81 * (cfg.ROAD_GRADE_PERCENT / 100.0)
        
        # 4. 惯性项 (包含旋转质量系数 1.1)
        inertial_work = v * (1.1 * a + grade_term)
        
        return drag_rolling + inertial_work

    def process(self, kinematics_data, detections, plate_cache):
        results = {}
        id_to_class = {tid: cid for tid, cid in zip(detections.tracker_id, detections.class_id)}

        for tid, data in kinematics_data.items():
            class_id = int(id_to_class.get(tid, cfg.YOLO_CLASS_CAR))
            
            # 如果 OCR 关闭或无记录，plate_color 为 Unknown
            plate_color = plate_cache.get(tid, "Unknown")
            
            # 1. 类型判定 (调用 classifier.py)
            _, type_str = VehicleClassifier.resolve_type(class_id, plate_color_override=plate_color)
            
            # 2. VSP 计算不再混入 EV 质量修正，而是使用纯物理公式
            v, a = data['speed'], data['accel']
            vsp = self._calculate_vsp(v, a, class_id)
            
            # 3. OpMode 判定
            op_mode = self._determine_opmode(v, a, vsp)
            
            # 4. 排放计算
            category = 'HDV' if 'HDV' in type_str else 'LDV'
            base_emission = self._get_emission_factor(op_mode, category)
            
            # 5. 针对 EV 的最终排放量修正
            # 逻辑：EV 虽然重 (MassFactor > 1)，但动能回收 (Regen) 极大减少了摩擦制动需求
            final_factor = 1.0
            
            if "electric" in type_str:
                # 质量惩罚: EV 较重，惯性大，基础物理磨损潜力增加 25%
                mass_penalty = cfg.MASS_FACTOR_EV # e.g. 1.25
                
                # 再生制动红利: 
                # OpMode 0 (急刹): 机械刹车介入较多，回收比例低 (例如仅省 30% -> 因子 0.7)
                # 其他 Mode (滑行/微刹): 电机几乎全包 (例如省 90% -> 因子 0.1)
                # 这里简化处理：OpMode 0 给 0.4 (省60%)，其他给 0.1 (省90%)
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
