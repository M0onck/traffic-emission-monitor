import math
from collections import defaultdict

class Reporter:
    def __init__(self, config: dict):
        self.debug_mode = config.get('debug_mode', False)
        self.fps = config.get('fps', 30)
        self.min_survival_frames = config.get('min_survival_frames', 30)

    def print_exit_report(self, tid, record, kinematics_estimator, vehicle_classifier):
        """
        [输出] 车辆离场报告
        包含：基本信息、OCR结果、物理统计(速度/里程)、工况分布、排放总量及强度。
        """
        if not self.debug_mode: return

        # 1. 存活时间过滤
        life_span = record.get('last_seen_frame', 0) - record.get('first_frame', 0)
        if life_span < self.min_survival_frames: return 

        duration = life_span / self.fps

        # 2. 车牌/车型投票解析
        history = record.get('plate_history', [])
        final_plate = "Unknown"
        vote_info = "No OCR"
        
        if history:
            scores = defaultdict(float)
            total_weight = 0.0
            for entry in history:
                # 权重 = 置信度 * 面积开根号 (防止大框主导)
                w = entry.get('conf', 1.0) * math.sqrt(entry.get('area', 0.0))
                scores[entry['color']] += w
                total_weight += w
            if scores:
                winner = max(scores, key=scores.get)
                conf = scores[winner] / total_weight if total_weight > 0 else 0
                final_plate = winner
                vote_info = f"Score {int(scores[winner])} ({conf:.1%})"

        final_plate, final_type = vehicle_classifier.resolve_type(
            record.get('class_id'), record.get('plate_history', [])
        )

        # 3. 物理统计 (速度 & 里程) [新增]
        dist_m = record.get('total_distance_m', 0.0)
        max_spd = record.get('max_speed', 0.0)
        
        # 计算平均速度 (m/s -> km/h)
        # 使用 总里程/总时间 计算，比瞬时速度求平均更准确
        avg_spd = dist_m / duration if duration > 0 else 0
        avg_spd_kmh = avg_spd * 3.6
        
        speed_info = f"Avg: {avg_spd_kmh:.1f} km/h | Max: {max_spd:.1f} m/s | Dist: {dist_m:.1f} m"

        # 4. 工况分布 (OpModes)
        op_stats = record.get('op_mode_stats', {})
        op_summary = []
        for mode in sorted(op_stats.keys()):
            seconds = op_stats[mode] / self.fps
            mode_name = {
                0: "Brake", 1: "Idle", 11: "Coast", 
                21: "Cruise(L)", 33: "Cruise(H)",
                35: "Accel(L)", 37: "Accel(H)"
            }.get(mode, str(mode))
            op_summary.append(f"{mode_name}:{seconds:.1f}s")
        
        op_str = " | ".join(op_summary) if op_summary else "No Data"
        
        # 5. 排放统计 (总量 & 强度) [新增]
        total_brake = record.get('brake_emission_mg', 0.0)
        total_tire = record.get('tire_emission_mg', 0.0)

        # 计算排放强度 (mg/km)
        # 仅当行驶距离足够长 (>10m) 时计算，避免极短轨迹产生的除零或离群值
        dist_km = dist_m / 1000.0
        if dist_km > 0.01:
            brake_intensity = total_brake / dist_km
            tire_intensity = total_tire / dist_km
            intensity_str = f"Intensity: Brake {brake_intensity:.1f} | Tire {tire_intensity:.1f} (mg/km)"
        else:
            intensity_str = "Intensity: N/A (Dist < 10m)"

        # 6. 打印报告
        print("-" * 70)
        print(f"[Exit] ID: {tid} | Life: {duration:.1f}s | Type: {final_type}")
        print(f"       Plate: {final_plate} [{vote_info}]")
        print(f"       Physics: {speed_info}")
        print(f"       OpModes: {op_str}")
        print(f"       Total:     Brake {total_brake:.2f} mg | Tire {total_tire:.2f} mg")
        print(f"       {intensity_str}")
        print("-" * 70 + "\n")
