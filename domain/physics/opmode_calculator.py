class MovesOpModeCalculator:
    """
    [业务层] MOVES OpMode 判定逻辑计算器 (Logic Calculator)
    ===========================================================================
    
    【设计模式】
    ---------------------------------------------------------------------------
    采用策略模式与依赖注入。本类封装了 EPA MOVES 标准中复杂的车辆操作工况
    (Operating Mode) 判定规则。通过在构造函数中注入阈值参数，实现了与全局
    配置文件的解耦，便于独立进行单元测试。

    【物理/业务逻辑 (扩展版)】
    ---------------------------------------------------------------------------
    MOVES 模型根据车辆的瞬时速度、加速度和比功率 (VSP) 将行驶状态划分为
    不同的 Bin (即 OpMode ID)。
    
    针对视觉测速噪声较大的问题，为了实现更稳健的轮胎排放估算 (查表法)，
    本计算器对标准 MOVES 逻辑进行了扩展，增加了对“加速行为”的显式分级判定，
    不再完全依赖 VSP 数值，而是直接捕捉运动趋势。
    
    工况定义与判定优先级 (Priority):
    1. Braking (0): 显著减速或刹车 (a <= 刹车阈值)。
    2. Idling (1): 车辆静止或类静止蠕行 (v < 怠速阈值)。
    3. Accel Hard (37): 急加速 (a >= 急加速阈值)。[新增]
       *注: 对应 MOVES 高负荷区间，此处作为定性判定。
    4. Accel Mild (35): 缓加速 (a >= 缓加速阈值)。[新增]
    5. Coasting (11): 车辆滑行 (VSP < 0 或 a < -0.1 且未刹车)。
    6. Cruising (21/33): 稳态巡航 (其他情况，根据速度区分低速/高速)。
    ===========================================================================
    """

    def __init__(self, config: dict):
        """
        初始化计算器，注入判定阈值。
        
        :param config: 包含排放阈值参数的字典 (通常来自 config.json 中的 emission_params)
        """
        # 1. 注入基础阈值 (默认值源自 MOVES 技术指南)
        self.braking_threshold = config.get("braking_decel_threshold", -0.89) # m/s²
        self.idling_speed = config.get("idling_speed_threshold", 0.45)        # m/s (1 mph)
        self.low_speed = config.get("low_speed_threshold", 11.17)             # m/s (25 mph)
        
        # 2. 注入加速阈值 (用于扩展的趋势判定逻辑)
        # 默认值: 缓加速 0.1 m/s², 急加速 1.5 m/s²
        self.accel_mild = config.get("accel_mild_threshold", 0.25)
        self.accel_hard = config.get("accel_hard_threshold", 1.5)
        
        # 3. 描述映射表 (用于调试输出)
        self.desc_map = {
            0: "Braking",
            1: "Idling", 
            11: "Coasting",
            21: "Cruising",
            33: "Cruising (High)",
            35: "Accel (Mild)",
            37: "Accel (Hard)"
        }

    def get_opmode(self, v_ms: float, a_ms2: float, vsp_kw_t: float = None) -> int:
        """
        根据 MOVES 标准及扩展逻辑判定当前工况 ID
        
        :param v_ms: 速度 (m/s)
        :param a_ms2: 加速度 (m/s²)
        :param vsp_kw_t: 车辆比功率 (kW/t)
        :return: OpMode ID (0, 1, 11, 21, 33, 35, 37)
        """
        # 1. 刹车判定 (Braking)
        # 优先级最高：只要减速度足够大，无论速度如何，都视为刹车工况
        if a_ms2 <= self.braking_threshold:
            return 0
            
        # 2. 怠速判定 (Idling)
        # 速度极低时，视为怠速 (OpMode 1)
        if v_ms < self.idling_speed:
            return 1
            
        # 3. 加速判定 (Acceleration) - [核心扩展逻辑]
        # 视觉测速中，若加速度持续大于阈值，定性为加速工况，
        # 以此直接查表获取排放率，规避物理公式中 a 被平方放大的误差。
        if a_ms2 >= self.accel_hard:
            return 37 # Hard Acceleration (OpMode 37)
        elif a_ms2 >= self.accel_mild:
            return 35 # Mild Acceleration (OpMode 35)
            
        # 4. 滑行/减速判定 (Coasting)
        # 未踩刹车(a > -0.89)，但 VSP < 0 或 a < -0.1，表示发动机未做正功
        # 车辆处于滑行阻力减速状态
        if vsp_kw_t is not None:
            if vsp_kw_t < 0: return 11
        elif a_ms2 < -0.1: # 如果上游未计算 VSP，使用加速度近似
            return 11
            
        # 5. 巡航判定 (Cruising)
        # VSP >= 0 且 加速度较小 (-0.1 ~ 0.1)，视为稳态巡航
        # 依然保留对高速巡航的区分 (虽然 tire model 查表时可能统一回退到 Cruise)
        if v_ms < self.low_speed:
            return 21 # Low Speed Cruise
        else:
            return 33 # High Speed Cruise

    def get_description(self, op_mode: int) -> str:
        """获取工况的文字描述"""
        return self.desc_map.get(op_mode, str(op_mode))
