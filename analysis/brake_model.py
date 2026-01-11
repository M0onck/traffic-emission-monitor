import config.settings as cfg

class BrakeEmissionModel:
    """
    [业务] 刹车磨损排放模型 (Based on MOVES)
    """
    def __init__(self):
        pass

    def _determine_vehicle_type(self, yolo_class_id, plate_color):
        """
        根据 YOLO 类别和车牌颜色判定车辆排放类型。
        策略：最大似然估计 (MLE)
        1. 有车牌颜色 -> 查表 (精确)
        2. 无车牌颜色 -> 基于车型先验概率 (Bus->电, Truck->油, Car->油)
        """
        # 1. 尝试精确匹配 (ClassID, Color)
        # 例如: (5, 'Green') -> HDV-electric
        key = (yolo_class_id, plate_color)
        if key in cfg.TYPE_MAP: 
            return cfg.TYPE_MAP[key]
            
        # 2. 兜底逻辑 (当 OCR 关闭 或 OCR 识别失败时)
        
        # 2.1 小型车 (Car/Taxi/SUV) -> 默认为汽油车
        if yolo_class_id == 2: 
            return cfg.TYPE_MAP.get('Default_Car', 'LDV-gasoline')
            
        # 2.2 大型车分流策略 (核心修改)
        if yolo_class_id == 5: # Bus (公交车/大巴)
            # 中国城市工况下，绝大多数公交为电动
            return "HDV-electric" 
            
        if yolo_class_id == 7: # Truck (卡车/货车)
            # 物流运输工况下，绝大多数卡车仍为柴油
            # (即使是 HDV-diesel，也比错误的归类为 electric 更安全，符合保守原则)
            return "HDV-diesel"
            
        # 2.3 极少数其他情况 (如 Person 误检等)，使用配置文件的总兜底
        return cfg.TYPE_MAP.get('Default_Heavy', 'HDV-diesel')

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
            class_id = int(id_to_class.get(tid, 2))
            
            # 如果 OCR 关闭或无记录，plate_color 为 Unknown
            plate_color = plate_cache.get(tid, "Unknown")
            
            # 1. 类型判定 (现在会正确返回 HDV-diesel)
            type_str = self._determine_vehicle_type(class_id, plate_color)
            
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
