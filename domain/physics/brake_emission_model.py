import numpy as np
from domain.physics.opmode_calculator import MovesOpModeCalculator

class BrakeEmissionModel:
    """
    [业务层] 刹车磨损排放模型 (Brake Wear Emission Model)

    该类基于 EPA MOVES 标准的 OpMode 方法，通过查表法计算车辆刹车磨损产生的颗粒物排放 (PM)。
    它不仅支持燃油车，还包含针对电动汽车 (EV) 再生制动特性的修正逻辑。

    Attributes:
        road_grade_percent (float): 道路坡度百分比 (默认 0.0)。
        mass_factor_ev (float): EV 质量惩罚系数 (通常 > 1.0，因为电池导致车重增加)。
        rates_map (dict): 基础排放率配置表 (mg/s)，按车型和 OpMode 索引。
        opmode_calculator (MovesOpModeCalculator): 工况计算器实例。
        YOLO_CLASS_BUS (int): 巴士在 YOLO 模型中的类别 ID。
        YOLO_CLASS_TRUCK (int): 卡车在 YOLO 模型中的类别 ID。
    """

    def __init__(self, config: dict):
        """
        初始化刹车排放模型。

        Args:
            config (dict): 包含模型参数的配置字典。
                - road_grade_percent: 道路坡度。
                - mass_factor_ev: EV 质量系数。
                - brake_wear_coefficients: 排放率系数表。
        """
        self.road_grade_percent = config.get("road_grade_percent", 0.0)
        self.mass_factor_ev = config.get("mass_factor_ev", 1.25)
        self.rates_map = config.get("brake_wear_coefficients", {})
        self.opmode_calculator = MovesOpModeCalculator(config)
        
        # YOLO Class ID 常量 (需与 config.json 或模型定义保持一致)
        self.YOLO_CLASS_BUS = 5
        self.YOLO_CLASS_TRUCK = 7

    def _get_emission_factor(self, op_mode: int, vehicle_category: str) -> float:
        """
        从配置表中查找基础排放因子 (Base Emission Factor)。

        Args:
            op_mode (int): 运行工况 ID (Operating Mode ID)。
            vehicle_category (str): 车辆大类 ('CAR', 'BUS', 'TRUCK')。

        Returns:
            float: 基础排放率 (mg/s)。如果未找到对应工况，返回 0.0。
        """
        # 优先查找指定车型，未找到则回退到 'CAR'
        rates = self.rates_map.get(vehicle_category, self.rates_map.get('CAR', {}))
        # 查找特定 OpMode 的排放率，默认为 0.0 (例如加速工况无刹车排放)
        return float(rates.get(op_mode, 0.0))

    def calculate_single_point(self, v_ms: float, a_ms2: float, vsp: float, 
                             vehicle_class_id: int, dt: float, type_str: str = "") -> dict:
        """
        计算单帧的刹车排放量。

        核心逻辑:
        1. 计算 OpMode。
        2. 查表获取基础排放率 (Base Rate)。
        3. 应用 EV 修正系数 (Regenerative Braking Correction)。

        Args:
            v_ms (float): 瞬时速度 (m/s)。
            a_ms2 (float): 瞬时加速度 (m/s²)。
            vsp (float): 车辆比功率 (Vehicle Specific Power)。
            vehicle_class_id (int): 车辆类别 ID。
            dt (float): 时间步长 (s)。
            type_str (str): 车辆细分类型描述 (用于判断是否为 EV)。

        Returns:
            dict: 包含排放质量、排放率、工况及调试信息的字典。
        """
        # 1. 计算运行工况 (OpMode)
        # 包含减速/刹车(0, 11) 和 加速(33, 35, 37) 等所有工况
        op_mode = self.opmode_calculator.get_opmode(v_ms, a_ms2, vsp)
        
        # 2. 映射车辆大类
        category = 'CAR'
        if vehicle_class_id == self.YOLO_CLASS_BUS: category = 'BUS'
        elif vehicle_class_id == self.YOLO_CLASS_TRUCK: category = 'TRUCK'
        
        # 3. 查表获取基础排放率
        # 注意：对于加速工况 (如 OpMode 33/35/37)，配置表中系数应为 0.0
        base_emission = self._get_emission_factor(op_mode, category)
        
        # 4. 应用修正系数
        final_factor = 1.0
        is_electric = "electric" in type_str
        
        if is_electric:
            # EV 再生制动 (Regen) 修正策略:
            # - OpMode 0 (Braking): 此时主要为机械刹车介入，但仍有部分能量回收，
            #   因此系数设为 0.4 (即机械刹车贡献 40% 的排放，相比燃油车减少 60%)。
            # - 其他工况 (如 OpMode 11 Coasting): 主要是滑行能量回收，机械刹车极少介入，
            #   系数设为 0.1 (大幅减少排放)。
            regen_factor = 0.4 if op_mode == 0 else 0.1
            
            # 最终系数 = 质量惩罚 * 再生制动减免
            # (EV 通常比同级燃油车重，会导致轮胎磨损增加，但再生制动显著减少刹车磨损)
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
        """
        批量处理当前帧的所有车辆 (实时计算)。
        
        注意：此方法主要用于实时 UI 展示或调试。
        高精度的最终排放计算通常在车辆离场后由 `monitor_engine` 统一调用 `calculate_single_point` 完成。

        Args:
            kinematics_data (dict): 车辆运动学数据 {tid: {'speed': ..., 'accel': ...}}.
            detections (sv.Detections): 当前帧的检测结果。
            plate_cache (dict): 车牌颜色缓存。
            vehicle_classifier: 车辆类型分类器实例。
            vsp_map (dict): 预计算的 VSP 映射表 {tid: vsp}.
            dt (float): 时间步长。

        Returns:
            dict: 处理结果映射表 {tid: {..., 'emission_rate': ..., 'op_mode': ...}}.
        """
        results = {}
        # 建立 track_id 到 class_id 的映射
        id_to_class = {tid: cid for tid, cid in zip(detections.tracker_id, detections.class_id)}
        
        for tid, data in kinematics_data.items():
            # 获取基础属性，默认为 CAR (ID 2)
            class_id = int(id_to_class.get(tid, 2))
            plate_color = plate_cache.get(tid, "Unknown")
            
            # 解析细分车型 (用于判断 EV)
            _, type_str = vehicle_classifier.resolve_type(class_id, plate_color_override=plate_color)
            
            # 执行单点计算
            res = self.calculate_single_point(
                data['speed'], data['accel'], vsp_map.get(tid, 0.0), 
                class_id, dt, type_str
            )
            
            # 打包结果
            results[tid] = {
                **data, 
                "vsp": vsp_map.get(tid, 0.0),
                "op_mode": res["op_mode"],
                "emission_rate": res["emission_rate"],
                "type_str": type_str,
                "plate_color": plate_color
            }
        return results
