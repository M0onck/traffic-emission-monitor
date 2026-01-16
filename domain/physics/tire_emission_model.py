class TireEmissionModel:
    """
    [业务层] 轮胎磨损排放模型 (查表法)
    """
    def __init__(self, config: dict, opmode_calculator):
        self.rates_map = config.get("tire_wear_coefficients", {})
        
        if opmode_calculator is None:
            raise ValueError("[TireModel] 依赖注入失败")
        
        self.opmode_calculator = opmode_calculator
        self.default_rate = 0.15 

    def _get_rate(self, vehicle_type: str, op_mode: int) -> float:
        cat_key = vehicle_type.upper()
        if "BUS" in cat_key: cat_key = "BUS"
        elif "TRUCK" in cat_key: cat_key = "TRUCK"
        else: cat_key = "CAR"
        
        rates = self.rates_map.get(cat_key, self.rates_map.get("CAR", {}))
        return float(rates.get(op_mode, rates.get("21", self.default_rate)))

    def process(self, vehicle_type, speed_ms, accel_ms2, dt, mass_kg=None, vsp_kW_t=None, is_electric=False, mass_factor=1.0) -> dict:
        op_mode = self.opmode_calculator.get_opmode(speed_ms, accel_ms2, vsp_kW_t)
        base_rate = self._get_rate(vehicle_type, op_mode)
        
        correction = mass_factor if is_electric else 1.0
        final_rate = base_rate * correction
        
        return {
            'pm10': final_rate * dt,
            'debug_info': {
                'op_mode': op_mode,
                'base_rate': base_rate,
                'correction': correction
            }
        }
