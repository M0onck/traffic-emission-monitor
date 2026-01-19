class TireEmissionModel:
    """
    [业务层] 轮胎磨损排放模型 (Tire Wear Emission Model)

    该类基于 EPA MOVES 标准的 OpMode 方法，通过查表法计算车辆轮胎磨损产生的颗粒物排放 (PM)。
    与刹车磨损不同，轮胎磨损与车辆重量呈正相关。因此，模型包含了针对电动汽车 (EV) 
    因电池重量导致整车质量增加的“质量惩罚修正”逻辑。

    Attributes:
        rates_map (dict): 基础排放率配置表 (mg/s)，按车型 ('CAR', 'BUS', 'TRUCK') 和 OpMode 索引。
        opmode_calculator (object): 工况计算器实例 (依赖注入)。
        default_rate (float): 默认排放率 (0.15 mg/s)，用于处理查表缺失的情况。
    """

    def __init__(self, config: dict, opmode_calculator):
        """
        初始化轮胎排放模型。

        Args:
            config (dict): 包含模型参数的配置字典。
                - tire_wear_coefficients: 排放率系数表。
            opmode_calculator: 已初始化的 OpMode 计算器实例 (必须非空)。
        
        Raises:
            ValueError: 如果 opmode_calculator 未被正确注入。
        """
        self.rates_map = config.get("tire_wear_coefficients", {})
        
        if opmode_calculator is None:
            raise ValueError("[TireModel] 依赖注入失败: opmode_calculator 不能为空")
        
        self.opmode_calculator = opmode_calculator
        self.default_rate = 0.15 

    def _get_rate(self, vehicle_type: str, op_mode: int) -> float:
        """
        查表获取基础排放因子 (Base Emission Factor)。

        逻辑:
        1. 将输入的车型字符串标准化为大类 ('BUS', 'TRUCK', 'CAR')。
        2. 优先查找对应 OpMode 的排放率。
        3. 如果未找到，回退到 OpMode 21 (巡航) 的值或硬编码的 default_rate。

        Args:
            vehicle_type (str): 车辆类型字符串 (e.g., 'Car-electric', 'Bus-diesel')。
            op_mode (int): 运行工况 ID。

        Returns:
            float: 基础排放率 (mg/s)。
        """
        cat_key = vehicle_type.upper()
        if "BUS" in cat_key: cat_key = "BUS"
        elif "TRUCK" in cat_key: cat_key = "TRUCK"
        else: cat_key = "CAR"
        
        # 两级回退查找：指定车型 -> CAR (默认车型)
        rates = self.rates_map.get(cat_key, self.rates_map.get("CAR", {}))
        
        # 键值回退查找：指定工况 -> 工况21 (通用巡航) -> 默认值
        return float(rates.get(op_mode, rates.get("21", self.default_rate)))

    def process(self, vehicle_type: str, speed_ms: float, accel_ms2: float, dt: float, 
                mass_kg=None, vsp_kW_t=None, is_electric: bool = False, mass_factor: float = 1.0) -> dict:
        """
        计算单帧的轮胎排放量。

        核心逻辑:
        1. 计算 OpMode。
        2. 查表获取基础排放率。
        3. 应用 EV 质量修正 (Mass Penalty)。
           - 物理原理: EV 通常比同级燃油车重 (电池包)，较大的垂直载荷会导致轮胎磨损增加。
           - 修正方式: Emission = BaseRate * MassFactor (如果 is_electric 为 True)。

        Args:
            vehicle_type (str): 车辆类型描述。
            speed_ms (float): 瞬时速度 (m/s)。
            accel_ms2 (float): 瞬时加速度 (m/s²)。
            dt (float): 时间步长 (s)。
            mass_kg (float, optional): 车辆质量 (暂未使用，保留接口)。
            vsp_kW_t (float, optional): 预计算的 VSP。如果不传则内部计算。
            is_electric (bool): 是否为电动汽车 (触发质量惩罚)。
            mass_factor (float): 质量修正系数 (通常 > 1.0)。

        Returns:
            dict: 包含排放总量 (mg) 和调试信息的字典。
        """
        # 1. 计算工况
        op_mode = self.opmode_calculator.get_opmode(speed_ms, accel_ms2, vsp_kW_t)
        
        # 2. 获取基础率
        base_rate = self._get_rate(vehicle_type, op_mode)
        
        # 3. 应用质量修正
        # 燃油车系数为 1.0，电动车应用传入的 mass_factor (如 1.25)
        correction = mass_factor if is_electric else 1.0
        final_rate = base_rate * correction
        
        return {
            'pm10': final_rate * dt, # 排放质量 = 排放率 * 时间
            'debug_info': {
                'op_mode': op_mode,
                'base_rate': base_rate,
                'correction': correction
            }
        }
