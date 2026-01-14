class MovesOpModeCalculator:
    """
    [业务层] MOVES OpMode 判定逻辑计算器 (Logic Calculator)
    ===========================================================================
    
    【设计模式】
    ---------------------------------------------------------------------------
    采用策略模式与依赖注入。本类封装了 EPA MOVES 标准中复杂的车辆操作工况
    (Operating Mode) 判定规则。通过在构造函数中注入阈值参数，实现了与全局
    配置文件的解耦，便于独立进行单元测试。

    【物理/业务逻辑】
    ---------------------------------------------------------------------------
    MOVES 模型根据车辆的瞬时速度、加速度和比功率 (VSP) 将行驶状态划分为
    不同的 Bin (即 OpMode ID)，每个 Bin 对应一组特定的排放因子。
    
    主要工况定义:
    - Braking (0): 显著减速或刹车 (a < 阈值)。
    - Idling (1): 车辆静止或类静止蠕行 (v < 阈值)。
    - Coasting (11): 车辆滑行 (VSP < 0 且未刹车)。
    - Cruising/Accel (21/33): 巡航或加速，根据速度区分低速/高速区。
    ===========================================================================
    """

    def __init__(self, config: dict):
        """
        初始化计算器，注入判定阈值。
        
        :param config: 包含排放阈值参数的字典 (通常来自 config.json 中的 emission_params)
        """
        # 1. 注入阈值 (提供默认值以防配置缺失，默认值源自 MOVES 技术指南)
        self.braking_threshold = config.get("braking_decel_threshold", -0.89) # m/s²
        self.idling_speed = config.get("idling_speed_threshold", 0.45)        # m/s (1 mph)
        self.low_speed = config.get("low_speed_threshold", 11.17)             # m/s (25 mph)
        
        # 2. 描述映射表 (用于调试输出)
        self.desc_map = {
            0: "Braking",
            1: "Idling", 
            11: "Coast/Decel",
            21: "Cruise(Low)",
            33: "Cruise(High)"
        }

    def get_opmode(self, v_ms: float, a_ms2: float, vsp_kw_t: float = None) -> int:
        """
        根据 MOVES 标准判定当前工况 ID
        
        :param v_ms: 速度 (m/s)
        :param a_ms2: 加速度 (m/s²)
        :param vsp_kw_t: 车辆比功率 (kW/t)
        :return: OpMode ID (0, 1, 11, 21, 33)
        """
        # 1. 刹车判定 (Braking)
        # 优先级最高：只要减速度足够大，无论速度如何，都视为刹车工况
        if a_ms2 <= self.braking_threshold:
            return 0
            
        # 2. 怠速判定 (Idling)
        # 速度极低时，视为怠速 (OpMode 1)
        if v_ms < self.idling_speed:
            return 1
            
        # 3. 滑行/减速判定 (Coasting)
        # VSP < 0 表示发动机未做正功 (车辆在滑行或轻微减速)
        if vsp_kw_t is not None:
            if vsp_kw_t < 0: return 11
        elif a_ms2 < -0.1: # 如果上游未计算 VSP，使用加速度近似
            return 11
            
        # 4. 巡航/加速判定 (Cruising/Acceleration)
        # VSP >= 0，发动机做功。此时根据速度区间区分低速/高速工况。
        if v_ms < self.low_speed:
            return 21 # Low Speed Cruise/Accel
        else:
            return 33 # High Speed Cruise/Accel

    def get_description(self, op_mode: int) -> str:
        """获取工况的文字描述"""
        return self.desc_map.get(op_mode, str(op_mode))
