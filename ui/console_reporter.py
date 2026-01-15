from collections import defaultdict

class Reporter:
    """
    [表现层] 调试报告生成器
    功能：负责格式化并打印车辆离场时的详细数据报告。
    """
    def __init__(self, config: dict):
        """
        :param config: 包含 debug_mode, fps, min_survival_frames 的字典
        """
        self.debug_mode = config.get('debug_mode', False)
        self.fps = config.get('fps', 30)
        self.min_survival_frames = config.get('min_survival_frames', 30)

    def print_exit_report(self, tid, record, kinematics_estimator, vehicle_classifier):
        """
        打印离场车辆的详细分析报告
        :param tid: 车辆追踪 ID
        :param record: 车辆档案字典 (来自 VehicleRegistry)
        :param kinematics_estimator: 用于获取运动历史 (可能为 None)
        """
        # 1. 检查配置开关
        if not self.debug_mode:
            return

        # 2. 噪点过滤 (存活时间过短)
        life_span = record.get('last_seen_frame', 0) - record.get('first_frame', 0)
        if life_span < self.min_survival_frames:
            return 

        history = record.get('plate_history', [])
        final_plate = "Unknown"
        vote_info = "无识别记录"
        
        # 3. 面积加权投票逻辑 (Area-Weighted Voting)
        if history:
            scores = defaultdict(float)
            total_weight = 0.0
            
            for entry in history:
                w = entry['area'] # 权重 = 像素面积
                scores[entry['color']] += w
                total_weight += w
                
            if scores:
                winner = max(scores, key=scores.get)
                # 计算胜出者的权重占比
                confidence = scores[winner] / total_weight if total_weight > 0 else 0
                final_plate = winner
                vote_info = f"得分 {int(scores[winner]/1000)}k (占比 {confidence:.1%})"

        # 4. 最终归类逻辑 (Type Classification)
        # 调用公共分类器
        final_plate, final_type = vehicle_classifier.resolve_type(
            record.get('class_id'), 
            plate_history=record.get('plate_history', [])
        )

        # 5. 运动统计
        speed_info = "N/A (Motion Off)"
        # 直接从 registry 的记录中读取最大速度
        max_spd = record.get('max_speed', 0.0)
        # 只有当最大速度有意义时才显示
        if max_spd > 0.1:
            speed_info = f"Max: {max_spd:.1f} m/s"

        # 6. 排放与工况统计
        op_stats = record.get('op_mode_stats', {})
        total_brake = record.get('brake_emission_mg', 0.0)
        total_tire = record.get('tire_emission_mg', 0.0)
        
        # 格式化 OpMode 时间 (帧数 -> 秒)
        op_summary = []
        for mode in sorted(op_stats.keys()):
            seconds = op_stats[mode] / self.fps
            mode_name = {
                0: "Braking", 1: "Idling", 11: "Coast", 
                21: "Cruise(L)", 33: "Cruise(H)"
            }.get(mode, str(mode))
            op_summary.append(f"{mode_name}: {seconds:.1f}s")
        
        op_str = " | ".join(op_summary) if op_summary else "无有效工况数据"

        # 7. 执行打印
        print("-" * 70)
        print(f"[离场] ID: {tid} | 存活: {life_span/self.fps:.1f}s | 类型: {final_type}")
        print(f"       车牌: {final_plate} [{vote_info}]")
        print(f"       速度: {speed_info}")
        print(f"       统计: {op_str}")
        print(f"       排放: 刹车 {total_brake:.2f} mg | 轮胎 {total_tire:.2f} mg (PM10)")
        print("-" * 70 + "\n")
