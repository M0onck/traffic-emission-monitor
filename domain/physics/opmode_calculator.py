class MovesOpModeCalculator:
    """
    [业务层] MOVES OpMode 判定逻辑计算器 (Logic Calculator)
    ===========================================================================
    
    【设计模式】
    ---------------------------------------------------------------------------
    采用策略模式与依赖注入。本类封装了 EPA MOVES 标准中复杂的车辆操作工况
    (Operating Mode) 判定规则。

    【物理/业务逻辑 (VSP-Based)】
    ---------------------------------------------------------------------------
    本计算器严格遵循 EPA MOVES 的 VSP (Vehicle Specific Power) 分箱逻辑。相比单纯
    基于运动学参数判断，这能有效区分“下坡加速”(加速度大但VSP低) 和“重载起步” (加速度
    小但VSP高) 等复杂工况。
    
    工况定义与判定优先级 (Priority):
    1. Braking (0): 显著减速 (a <= 刹车阈值)。
    2. Idling (1): 车辆静止或类静止 (v < 怠速阈值)。
    3. Running (基于 VSP 分箱):
       - Coasting (11): VSP < 0
       - Cruise (21/33): 0 <= VSP < 6 (区分低速/高速)
       - Accel Mild (35): 6 <= VSP < 12
       - Accel Hard (37): VSP >= 12
    ===========================================================================
    """

    def __init__(self, config: dict):
        """
        初始化计算器，注入判定阈值。
        
        :param config: 包含排放阈值参数的字典
        """
        # 1. 注入基础阈值 (默认值源自 MOVES 技术指南)
        self.braking_threshold = config.get("braking_decel_threshold", -0.89) # m/s²
        self.idling_speed = config.get("idling_speed_threshold", 0.45)        # m/s (1 mph)
        self.low_speed = config.get("low_speed_threshold", 11.17)             # m/s (25 mph)
        
        # 2. VSP 分箱阈值 (kW/tonne)
        # 源自 MOVES Running Exhaust/Wear Bins 的简化映射
        self.vsp_mild = 6.0
        self.vsp_hard = 12.0
        
        # 3. 兜底加速阈值 (仅在 VSP 不可用时使用)
        self.accel_mild = config.get("accel_mild_threshold", 0.25)
        self.accel_hard = config.get("accel_hard_threshold", 1.5)
        
        # 4. 描述映射表
        self.desc_map = {
            0: "Braking",
            1: "Idling", 
            11: "Coasting",
            21: "Cruise (Low Spd)",
            33: "Cruise (High Spd)",
            35: "Accel (Mild Load)",
            37: "Accel (High Load)"
        }

    def get_opmode(self, v_ms: float, a_ms2: float, vsp_kw_t: float = None) -> int:
        """
        根据 MOVES 标准判定当前工况 ID。
        
        逻辑流:
        Braking -> Idling -> (VSP available?) -> VSP Bins -> (Fallback) Accel Thresholds
        
        :param v_ms: 速度 (m/s)
        :param a_ms2: 加速度 (m/s²)
        :param vsp_kw_t: 车辆比功率 (kW/t)，推荐传入以获得高精度判定
        :return: OpMode ID
        """
        # 1. 刹车判定 (Braking)
        # 优先级最高：MOVES 定义中，减速度超过阈值强制归为 OpMode 0 (不管 VSP)
        if a_ms2 <= self.braking_threshold:
            return 0
            
        # 2. 怠速判定 (Idling)
        if v_ms < self.idling_speed:
            return 1
            
        # 3. 运行工况判定 (Running Mode)
        # 优先使用 VSP 进行准确分箱
        if vsp_kw_t is not None:
            if vsp_kw_t < 0:
                return 11 # Coasting (滑行/减速但未踩死刹车)
                
            elif vsp_kw_t < self.vsp_mild:
                # Cruise (0 <= VSP < 6)
                # 进一步根据速度细分
                return 21 if v_ms < self.low_speed else 33
                
            elif vsp_kw_t < self.vsp_hard:
                return 35 # Mild Accel / Moderate Load (6 <= VSP < 12)
                
            else:
                return 37 # Hard Accel / High Load (VSP >= 12)

        # 4. 兜底逻辑 (Fallback)
        # 当上游未能提供 VSP 时，回退到基于加速度的定性判断
        else:
            if a_ms2 < -0.1:
                return 11 # Coasting
            elif a_ms2 >= self.accel_hard:
                return 37
            elif a_ms2 >= self.accel_mild:
                return 35
            else:
                return 21 if v_ms < self.low_speed else 33

    def get_description(self, op_mode: int) -> str:
        """获取工况的文字描述"""
        return self.desc_map.get(op_mode, str(op_mode))
