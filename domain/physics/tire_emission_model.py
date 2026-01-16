import math

class TireEmissionModel:
    """
    [领域模型] 轮胎磨损排放模型 (查表法重构版)
    ===========================================================================
    
    【设计变更说明】
    ---------------------------------------------------------------------------
    鉴于单目视觉测速固有的加速度噪声会被传统的物理模型 (Archard 公式, F=ma) 
    放大，导致瞬态排放估算值出现非物理的剧烈波动。
    
    本模型已从 "受力分析模型" 重构为工程上更稳健的 "工况分类 + 查表映射" 
    (OpMode Lookup) 模式。
    
    【核心逻辑】
    ---------------------------------------------------------------------------
    1. 依赖注入 (Dependency Injection):
       不再内部实例化 OpModeCalculator，而是通过构造函数注入计算器实例。
       这实现了物理参数 (Rate Table) 与 逻辑判定 (OpMode Logic) 的解耦。
    
    2. 工况分类 (OpMode Binning):
       利用注入的计算器，将车辆运动状态归类为：
       - Braking (0): 急刹车
       - Idling (1): 怠速
       - Accel (35/37): 缓加速/急加速 (新增)
       - Cruising (21): 稳态巡航
       
    3. 排放计算 (Emission Calculation):
       Emission = Base_Rate(OpMode, VehicleType) * Weight_Correction * dt
       
       * Base_Rate: 从配置文件读取的经验排放率 (mg/s)。
       * Weight_Correction: 仅针对电动车因电池增重进行的线性修正。
    ===========================================================================
    """

    def __init__(self, config: dict, opmode_calculator):
        """
        初始化模型，加载排放率表并注入依赖。
        
        :param config: 模型配置字典，必须包含 "tire_wear_rates"。
        :param opmode_calculator: [依赖注入] 已实例化的 OpMode 计算器策略对象。
                                  必须实现 get_opmode(v, a, vsp) 和 get_description(id) 接口。
        """
        # 1. 加载轮胎磨损率表 (Look-up Table)
        # 结构示例: {"CAR": {"0": 0.6, "21": 0.15, ...}, "BUS": ...}
        self.rates_map = config.get("tire_wear_rates", {})
        
        # 2. 注入 OpMode 计算器
        if opmode_calculator is None:
            raise ValueError("[TireModel] 必须注入 opmode_calculator 实例")
        self.opmode_calculator = opmode_calculator
        
        # 3. 定义兜底默认值 (mg/s)
        # 当查不到具体工况或车型时，默认按轿车巡航标准处理
        self.default_rate = 0.15 
        
        print(f">>> [TireModel] 初始化完成。依赖注入: {opmode_calculator.__class__.__name__}")

    def _get_rate(self, vehicle_type: str, op_mode: int) -> float:
        """
        内部方法：根据车型和工况查表获取基准排放率 (mg/s)
        
        :param vehicle_type: 车辆类型字符串 (如 "Car-gasoline", "Bus-electric")
        :param op_mode: 工况 ID (如 0, 21, 37)
        :return: 排放率 (mg/s)
        """
        # 1. 归一化车型键名 (config key 通常为 CAR/BUS/TRUCK)
        # 逻辑：只要包含 "BUS" 就视为大巴，包含 "TRUCK" 视为卡车，否则默认为轿车
        cat_key = vehicle_type.upper()
        if "BUS" in cat_key: 
            cat_key = "BUS"
        elif "TRUCK" in cat_key: 
            cat_key = "TRUCK"
        else: 
            cat_key = "CAR"
        
        # 2. 获取该车型的费率表
        # 如果找不到该车型，回退到 CAR 表
        rates = self.rates_map.get(cat_key, self.rates_map.get("CAR", {}))
        
        # 3. 获取具体工况的费率
        # 注意：JSON key 可能是字符串，尝试 str(op_mode) 和 int(op_mode)
        rate = rates.get(str(op_mode), rates.get(op_mode))
        
        if rate is None:
            # 如果是未定义的工况 (比如 MOVES 的 33)，回退到稳态巡航 (21)
            # 确保不会因为键缺失导致 Crash
            rate = rates.get("21", self.default_rate)
            
        return float(rate)

    def process(self, vehicle_type: str, speed_ms: float, accel_ms2: float, dt: float, 
                mass_kg: float = None, vsp_kW_t: float = None,
                is_electric: bool = False, mass_factor: float = 1.0) -> dict:
        """
        执行单步排放计算
        
        :param vehicle_type: 车辆类型 (用于查表)
        :param speed_ms: 当前车速 (m/s)
        :param accel_ms2: 当前加速度 (m/s²)
        :param dt: 时间步长 (s)
        :param mass_kg: (可选) 车辆质量，本模型主要使用 mass_factor 进行相对修正
        :param vsp_kW_t: 车辆比功率 (用于辅助 OpMode 判定)
        :param is_electric: 是否为电动车 (影响重量修正)
        :param mass_factor: 质量修正系数 (如 EV = 1.25)
        :return: 包含 pm10 排放值和调试信息的字典
        """
        # 1. 判定工况 (委托给注入的计算器)
        # 这是核心步骤：将含噪的运动数据“定性”为具体的行驶状态
        op_mode = self.opmode_calculator.get_opmode(speed_ms, accel_ms2, vsp_kW_t)
        
        # 2. 查表获取基准排放率 (mg/s)
        base_rate_mg_s = self._get_rate(vehicle_type, op_mode)
        
        # 3. 应用物理修正
        # 逻辑：电动车因电池增重，轮胎磨损线性增加。
        # 相比旧模型 (F^2)，线性修正更能避免因参数不准导致的数值爆炸。
        weight_correction = mass_factor if is_electric else 1.0
        
        # 4. 计算最终排放量 (mg)
        # Emission = Rate * Correction * Time
        final_rate = base_rate_mg_s * weight_correction
        emission_mg = final_rate * dt

        return {
            'pm10': emission_mg,
            
            # 详细调试信息 (用于 Console Reporter 或 UI 显示)
            'debug_info': {
                'mode': self.opmode_calculator.get_description(op_mode),
                'op_mode': op_mode,
                'base_rate': base_rate_mg_s,
                'correction': weight_correction,
                'calc_method': "Lookup Table (DI)"
            }
        }
