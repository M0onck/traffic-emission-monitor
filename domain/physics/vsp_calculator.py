class VSPCalculator:
    """
    [业务层] 车辆比功率 (VSP) 计算器
    职责: 根据车辆运动状态和物理参数计算 VSP (kW/tonne)。
    公式: VSP = (Av + Bv² + Cv³ + m*v*(a + g*sinθ)) / m
    """
    def __init__(self, config: dict):
        """
        :param config: 包含 'vsp_coefficients' 和 'road_grade_percent' 的配置字典
        """
        # 1. 加载 VSP 系数表
        self.coeffs_map = config.get("vsp_coefficients", {})
        # 2. 道路坡度 (默认为 0)
        self.road_grade = config.get("road_grade_percent", 0.0)
        
        # 默认系数 (轿车标准)
        self.default_coeffs = self.coeffs_map.get("default", 
            {"a_m": 0.156, "b_m": 0.002, "c_m": 0.0005}
        )

    def calculate(self, v_ms: float, a_ms2: float, class_id: int) -> float:
        """
        计算 VSP
        :param v_ms: 速度 (m/s)
        :param a_ms2: 加速度 (m/s²)
        :param class_id: 车辆类别 ID (用于查找 A/B/C 系数)
        :return: VSP (kW/t)
        """
        # 查找系数 (支持 int ID 或 str key 的兼容查找)
        coeffs = self.coeffs_map.get(class_id, 
                 self.coeffs_map.get(str(class_id), self.default_coeffs))
        
        # 提取系数
        a_m = coeffs.get("a_m", 0.156)
        b_m = coeffs.get("b_m", 0.002)
        c_m = coeffs.get("c_m", 0.0005)

        # 1. 阻力功率项 (Drag & Rolling) / mass
        # P_drag/m = A*v + B*v^2 + C*v^3
        drag_term = a_m * v_ms + b_m * (v_ms**2) + c_m * (v_ms**3)
        
        # 2. 惯性与重力功率项 (Inertial & Potential) / mass
        # P_inert/m = v * (1.1*a + 9.81*sin(theta))
        # *1.1 是旋转质量系数 (Mass Factor)
        grade_term = 9.81 * (self.road_grade / 100.0)
        inertial_term = v_ms * (1.1 * a_ms2 + grade_term)
        
        return drag_term + inertial_term
