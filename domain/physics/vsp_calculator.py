class VSPCalculator:
    """
    [业务层] 车辆比功率 (Vehicle Specific Power, VSP) 计算器

    **职责 (Responsibility):**
    根据车辆的瞬时运动状态（速度、加速度）和物理参数，计算车辆的比功率。
    VSP 是衡量车辆发动机负载的代理变量，广泛应用于 EPA MOVES 等排放模型中，
    用于查找对应的运行工况 (OpMode) 和排放因子。

    **物理公式 (Physical Formula):**
    VSP 定义为发动机输出功率与车辆质量的比值（单位: $kW/tonne$）。
    计算公式如下：

    $$ \text{VSP} = \frac{P_{load}}{m} = v \cdot [1.1 \cdot a + 9.81 \cdot \sin(\theta)] + (a_m \cdot v + b_m \cdot v^2 + c_m \cdot v^3) $$

    其中:
    * $v$: 车辆瞬时速度 ($m/s$)
    * $a$: 车辆瞬时加速度 ($m/s^2$)
    * $\theta$: 道路坡度角 (Road Grade)
    * $1.1$: 旋转质量系数 (Mass Factor)，用于补偿车轮、曲轴等旋转部件的转动惯量
    * $a_m, b_m, c_m$: 归一化的行驶阻力系数 ($kW \cdot s/m$ 等)，分别对应滚动阻力、旋转阻力和空气阻力

    **单位 (Units):**
    * 输入速度: $m/s$
    * 输入加速度: $m/s^2$
    * 输出 VSP: $kW/tonne$
    """

    def __init__(self, config: dict):
        """
        初始化 VSP 计算器。

        Args:
            config (dict): 配置字典，需包含以下键:
                - `vsp_coefficients` (dict): 车辆系数映射表。结构为 `{class_id: {'a_m': float, 'b_m': float, 'c_m': float}}`。
                - `road_grade_percent` (float): 道路坡度百分比 (例如 0.0 表示平路, 5.0 表示 5% 坡度)。
        """
        # 1. 加载 VSP 系数表 (通常来自 EPA MOVES 数据库或实测标定)
        self.coeffs_map = config.get("vsp_coefficients", {})
        
        # 2. 道路坡度 (默认为 0.0)
        # 注意: 这里存储的是百分比值，计算时需除以 100
        self.road_grade = config.get("road_grade_percent", 0.0)
        
        # 默认系数 (兜底策略: 使用标准轿车系数)
        # a_m: 滚动阻力系数 (Rolling Resistance)
        # b_m: 旋转阻力系数 (Rotational Resistance)
        # c_m: 空气阻力系数 (Aerodynamic Drag)
        self.default_coeffs = self.coeffs_map.get("default", 
            {"a_m": 0.156, "b_m": 0.002, "c_m": 0.0005}
        )

    def calculate(self, v_ms: float, a_ms2: float, class_id: int) -> float:
        """
        计算单帧的车辆比功率 (VSP)。

        Args:
            v_ms (float): 瞬时速度，单位 $m/s$。
            a_ms2 (float): 瞬时加速度，单位 $m/s^2$。
            class_id (int): 车辆类别 ID (用于查找对应的 A/B/C 阻力系数)。

        Returns:
            float: 计算出的 VSP 值，单位 $kW/tonne$。
        """
        # 查找系数 (支持 int ID 或 str key 的兼容查找，未找到则回退至 default)
        coeffs = self.coeffs_map.get(class_id, 
                 self.coeffs_map.get(str(class_id), self.default_coeffs))
        
        # 提取归一化阻力系数
        a_m = coeffs.get("a_m", 0.156)
        b_m = coeffs.get("b_m", 0.002)
        c_m = coeffs.get("c_m", 0.0005)

        # ---------------------------------------------------------
        # 第一部分: 克服行驶阻力所需的功率 (Drag & Rolling Power)
        # 公式: $P_{drag}/m = a_m \cdot v + b_m \cdot v^2 + c_m \cdot v^3$
        # ---------------------------------------------------------
        drag_term = a_m * v_ms + b_m * (v_ms**2) + c_m * (v_ms**3)
        
        # ---------------------------------------------------------
        # 第二部分: 克服惯性与重力所需的功率 (Inertial & Potential Power)
        # 公式: $P_{inert}/m = v \cdot (1.1 \cdot a + 9.81 \cdot \sin(\theta))$
        # 注意: 当坡度较小时，$\sin(\theta) \approx \tan(\theta) = \text{grade}/100$
        # ---------------------------------------------------------
        grade_term = 9.81 * (self.road_grade / 100.0)
        inertial_term = v_ms * (1.1 * a_ms2 + grade_term)
        
        return drag_term + inertial_term
