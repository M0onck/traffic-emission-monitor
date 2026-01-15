import math
from domain.physics.opmode_calculator import MovesOpModeCalculator

class TireEmissionModel:
    """
    [领域模型] 轮胎磨损排放物理模型 (Physics-based Tire Wear Model)
    ===========================================================================
    
    【建模背景与目标】
    ---------------------------------------------------------------------------
    传统的轮胎磨损评估（如 EPA MOVES）主要基于行驶距离（mg/km），这种平均化方法
    无法反映微观驾驶行为（急加速、急刹车）以及不同车辆载荷（如新能源车增重）对
    轮胎磨损产生的非线性、瞬态影响。
    本模型旨在构建一个能够响应瞬时工况和车辆物理属性的精细化计算引擎。

    【模型架构: "基准-扰动" 二元模型 (Baseline-Perturbation Model)】
    ---------------------------------------------------------------------------
    本模型将总磨损解构为两个独立的物理过程：
    
    Total Emission = E_baseline (稳态基准) + E_transient (瞬态扰动)

    1. 稳态基准项 (Baseline Term):
       - 物理含义: 表征车辆在标准工况下，因滚动接触疲劳 (Rolling Contact Fatigue)
                   和稳态微滑移引起的背景磨损。
       - 建模方法: 采用 EPA MOVES / EMEP 的基于距离的排放因子。
       - 特性: 与行驶距离成正比，与车辆质量呈近似线性关系。
       
    2. 瞬态扰动项 (Transient Perturbation Term):
       - 物理含义: 表征高动态工况下，轮胎接地面因承受巨大纵向剪切力 (F_shear) 
                   而发生的剧烈摩擦磨损。
       - 建模方法: 基于摩擦学第一性原理 —— Archard 磨损定律。

    【物理推导: 瞬态项的第一性原理 (First Principles Derivation)】
    ---------------------------------------------------------------------------
    依据 Archard 定律，磨损体积 V 正比于载荷 W 和滑动距离 L，反比于硬度 H:
    V = K_wear * W * L / H

    转换为质量磨损率 (m_dot):
    (1) 磨损率正比于摩擦功率 (P_fric):
        m_dot ∝ P_fric = F_shear * v_slip
        
    (2) 引入轮胎动力学: 滑移速度 (v_slip) 与车速 (v) 和滑移率 (S) 有关:
        v_slip = v * S
        
    (3) 引入轮胎刚度: 在非抱死的线性区，滑移率 S 由纵向刚度 (C_k) 决定:
        S ≈ F_shear / C_k
        
    (4) 联立 (1)(2)(3)，得到瞬态磨损率公式:
        m_dot ∝ F_shear * (v * F_shear / C_k)
              = (物理系数 Ψ) * F_shear² * v
        *注: 磨损对剪切力表现出显著的平方级 (非线性) 敏感性。

    【物理参数的车型差异化 (Vehicle-Specific Physics)】
    ---------------------------------------------------------------------------
    物理系数 Ψ = (rho * k_spec) / C_k
    不同车型的轮胎物理属性差异巨大，必须分别计算：
    - 轿车 (Car): 轮胎较软 (C_k低)，比磨损率较高。
    - 卡车 (Truck): 轮胎极硬 (C_k高，通常是轿车的3-5倍)，耐磨配方 (k_spec低)。
    
    若不区分车型，使用轿车参数计算卡车（质量大、力大），会导致卡车排放被错误地
    计算为轿车的数十倍（因为 F² 效应）。引入高刚度 C_k 可正确抵消 F² 带来的
    过激增长，符合物理事实。
    ===========================================================================
    """

    def __init__(self, config: dict):
        """
        初始化模型，从配置中加载物理参数与基准因子。
        :param config: 完整的配置字典，程序将提取 "tire_wear_model" 子项。
        """
        # 1. 提取子配置
        self.cfg = config.get("tire_wear_model", {})
        
        # 2. 加载稳态基准参数 (来自 EPA/EMEP)
        self.base_ef = self.cfg.get("base_ef", {
            "car": 8.7,     # 轻型车
            "bus": 21.0,    # 公交车/客车
            "truck": 25.0,  # 重型卡车
            "default": 10.0
        })
        self.ref_mass = self.cfg.get("ref_mass", {
            "car": 1600.0,
            "bus": 13000.0,
            "truck": 24000.0
        })

        # 3. 加载并预计算瞬态物理系数 (Pre-calculate Physics Coeffs)
        # Coeff(Ψ) = (rho * k_spec) / C_k
        raw_physics = self.cfg.get("physics", {})
        self.physics_coeffs = {}

        # 定义 PM10 转化系数 (Source: Grigoratos et al., 2014)
        # 仅 1% 的总磨损质量会成为 PM10 悬浮物
        self.PM10_RATIO = 0.01
        
        # 定义需要处理的车型键值
        target_keys = set(raw_physics.keys()) | {'default'}
        
        for key in target_keys:
            # 获取单车型参数，缺省则回退到默认 (轿车标准)
            params = raw_physics.get(key, raw_physics.get('default', {
                "stiffness_N": 1.8e5,
                "specific_wear_rate": 2.5e-10,
                "rubber_density": 1200.0
            }))
            
            stiffness = params.get("stiffness_N", 1.8e5)
            k_spec = params.get("specific_wear_rate", 2.5e-10)
            rho = params.get("rubber_density", 1200.0)
            
            if stiffness > 0:
                coeff = (rho * k_spec) / stiffness
            else:
                coeff = 0.0
                
            self.physics_coeffs[key] = coeff
            
        print(f">>> [TireModel] 已加载物理系数表: {self.physics_coeffs}")

        # 从配置中提取道路坡度 (用于低速回退计算)
        # 注意：这里的 config 是 bootstrap 传入的 tire_emission_config 字典
        self.road_grade = config.get("emission_params", {}).get("road_grade_percent", 0.0)

        # 4. 实例化 OpMode 计算器
        emission_cfg = config.get("emission_params", {})
        self.opmode_calculator = MovesOpModeCalculator(emission_cfg)

    def process(self, vehicle_type: str, speed_ms: float, accel_ms2: float, dt: float, 
                mass_kg: float = None, vsp_kW_t: float = None,
                is_electric: bool = False, mass_factor: float = 1.0) -> dict:
        """
        计算单步时间 (dt) 内的轮胎磨损排放总量 (PM10)。
        
        :param vehicle_type: 车辆类型 (car, bus, truck)，用于查找对应的物理系数。
        :param speed_ms: 当前车速 (m/s)
        :param accel_ms2: 当前加速度 (m/s²)
        :param dt: 时间步长 (s)
        :param mass_kg: 当前车辆质量估算值 (kg)
        :param vsp_kW_t: [核心输入] 车辆比功率 (kW/t)，用于精确计算剪切力。
        :return: 包含排放值和调试信息的字典。
        """
        # 静止状态不产生磨损
        if speed_ms <= 0.1:
            return {'pm10': 0.0, 'debug_info': {'mode': 'Static (1)', 'op_mode': 1}}
        
        # 标准化 vehicle_type
        v_type_key = vehicle_type.lower()
        if v_type_key not in self.physics_coeffs:
            v_type_key = 'default'

        # 确定当前质量和参考质量
        ref_m = self.ref_mass.get(v_type_key, 1500.0)
        
        if mass_kg is not None:
            curr_m = mass_kg
        else:
            # 如果是电动车，应用外部传入的增重系数
            curr_m = ref_m * mass_factor if is_electric else ref_m

        # =================================================================
        # Part A: 计算稳态基准排放 (Baseline Emission)
        # =================================================================
        # 公式: E_base = EF_moves(mg/m) * Distance(m) * Mass_Correction
        distance_m = speed_ms * dt
        ef_per_m = self.base_ef.get(v_type_key, self.base_ef['default']) / 1000.0
        
        emission_base_mg = ef_per_m * distance_m * (curr_m / ref_m)

        # =================================================================
        # Part B: 计算瞬态剪切排放 (Transient Shear Emission)
        # =================================================================
        # 1. 确定物理系数 Ψ (根据车型)
        current_phys_coeff = self.physics_coeffs.get(v_type_key, self.physics_coeffs['default'])
        
        # 2. 计算剪切力 F_shear
        shear_force_N = 0.0
        calc_method = "accel_fallback" 

        if vsp_kW_t is not None and speed_ms > 1.0:
            # [策略 1: 基于 VSP 的全阻力反演]
            # 物理含义: VSP 包含了 惯性 + 风阻 + 滚阻 + 坡度
            # P_wheel ≈ Mass * VSP
            power_W = abs(vsp_kW_t) * curr_m
            shear_force_N = power_W / speed_ms
            calc_method = "vsp_inversion"
            
        elif abs(accel_ms2) > 0.3:
            # [策略 2: 基于加速度的惯性估算]
            # F_total ≈ F_inertial + F_grade
            force_inertial = curr_m * abs(accel_ms2)
            # 坡度力: F = m * g * sin(theta) ≈ m * g * (grade/100)
            g = 9.81
            force_grade = curr_m * g * (self.road_grade / 100.0)
            # 在坡道起步或刹车时，坡度力都会增加轮胎负担（简化处理取叠加）
            shear_force_N = force_inertial + max(0, force_grade)

        # 3. 应用物理公式: m_dot = Ψ * F_shear² * v
        emission_pm10_kg = 0.0
        # 设置最小力阈值 (10N) 以忽略数值噪声
        if shear_force_N > 10.0:
            # 1. 计算总橡胶磨损质量 (Total Rubber Mass Loss)
            # m_dot_total = Ψ * F² * v
            # current_phys_coeff 基于 k_spec_total (e-10)
            total_mass_rate_kg_s = current_phys_coeff * (shear_force_N ** 2) * speed_ms
            
            # 2. 转换为 PM10 排放 (应用 1% 比例)
            # Emission_PM10 = Emission_Total * Ratio
            emission_pm10_kg = total_mass_rate_kg_s * dt * self.PM10_RATIO

        emission_shear_mg = emission_pm10_kg * 1e6

        # =================================================================
        # Part C: 汇总输出与工况判定
        # =================================================================
        total_pm10 = emission_base_mg + emission_shear_mg
        
        # 调用注入的 OpMode 计算器
        op_mode_id = self.opmode_calculator.get_opmode(speed_ms, accel_ms2, vsp_kW_t)
        op_desc = self.opmode_calculator.get_description(op_mode_id)

        return {
            'pm10': total_pm10,
            
            # 调试信息
            'debug_info': {
                'mode': f"{op_desc} ({op_mode_id})",
                'op_mode': op_mode_id,
                'calc_method': calc_method,
                'mass_kg': int(curr_m),           # 检查是否误用了重型车质量
                'speed_ms': round(speed_ms, 2),   # 检查速度是否虚高
                'accel': round(accel_ms2, 2),     # 检查加速度是否抖动
                'base_mg': round(emission_base_mg, 4),
                'shear_mg': round(emission_shear_mg, 4),
                'force_N': int(shear_force_N),
                'phys_coeff': current_phys_coeff
            }
        }
