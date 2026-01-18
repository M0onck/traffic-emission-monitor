import math
import numpy as np
from collections import defaultdict
import plotext as plt

class Reporter:
    def __init__(self, config: dict):
        self.debug_mode = config.get('debug_mode', False)
        self.fps = config.get('fps', 30)
        self.min_survival_frames = config.get('min_survival_frames', 30)

    def print_exit_report(self, tid, record, kinematics_estimator, vehicle_classifier):
        """
        [输出] 车辆离场报告
        包含：基本信息、OCR结果、物理统计(速度/里程)、工况分布、排放总量及强度。
        [新增] 运动学曲线图 (速度 & 加速度)
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

        # 3. 物理统计 (速度 & 里程)
        dist_m = record.get('total_distance_m', 0.0)
        max_spd = record.get('max_speed', 0.0)
        
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
        
        # 5. 排放统计 (总量 & 强度)
        total_brake = record.get('brake_emission_mg', 0.0)
        total_tire = record.get('tire_emission_mg', 0.0)

        dist_km = dist_m / 1000.0
        if dist_km > 0.01:
            brake_intensity = total_brake / dist_km
            tire_intensity = total_tire / dist_km
            intensity_str = f"Intensity: Brake {brake_intensity:.1f} | Tire {tire_intensity:.1f} (mg/km)"
        else:
            intensity_str = "Intensity: N/A (Dist < 10m)"

        # 6. [新增] 绘制运动学曲线 (速度 & 加速度)
        trajectory = record.get('trajectory', [])
        
        # 打印报告头
        print("-" * 70)
        print(f"[Exit] ID: {tid} | Life: {duration:.1f}s | Type: {final_type}")
        print(f"       Plate: {final_plate} [{vote_info}]")
        print(f"       Physics: {speed_info}")
        print(f"       OpModes: {op_str}")
        print(f"       Total:     Brake {total_brake:.2f} mg | Tire {total_tire:.2f} mg")
        print(f"       {intensity_str}")
        
        # 嵌入图表
        if len(trajectory) > 5:
            # 提取数据
            speeds = [p['speed'] for p in trajectory]
            accels = [p['accel'] for p in trajectory]
            
            print("\n       [Kinematics Profile]")
            self._plot_kinematics_graph(speeds, accels)
            
        print("-" * 70 + "\n")

    def _plot_kinematics_graph(self, speeds, accels):
        """
        [可视化终极版 - 修复版] 绘制运动学曲线
        修复点：
        1. 解决了 yticks 传入整数导致的 TypeError 报错。
        2. 手动计算 13 个均匀分布的刻度值，强制填满每一行。
        """
        # 1. 获取终端尺寸
        term_w, term_h = plt.terminal_size()
        safe_w = min(term_w - 5, 100) 
        safe_h = 31 
        
        # 窗口过小检查
        if safe_w < 40 or term_h < 36:
            return 

        # 2. 清除画布
        plt.clear_figure()
        plt.plotsize(safe_w, safe_h)
        plt.subplots(2, 1)
        
        t = [i / self.fps for i in range(len(speeds))]
        
        # [核心参数] 刻度数量
        # 总高31 -> 单图高约15 -> 除去标题/X轴，实际绘图区约12-13行
        DENSE_TICKS_COUNT = 13

        # --- 子图 1: 速度曲线 (Top) ---
        plt.subplot(1, 1)
        plt.plot(t, speeds, marker="dot", color="cyan")
        plt.title("Speed (m/s)")
        plt.theme('dark')
        plt.ticks_color('white') 
        plt.grid(True, True)
        
        # 计算量程
        max_v = max(speeds) if speeds else 0
        limit_v = max(max_v * 1.05, 1.0) 
        plt.ylim(0, limit_v)
        
        # [修复] 手动生成刻度列表
        # linspace(start, stop, num) -> 生成包含 num 个点的等差数列
        v_ticks = np.linspace(0, limit_v, DENSE_TICKS_COUNT).tolist()
        plt.yticks(v_ticks) 

        # --- 子图 2: 加速度曲线 (Bottom) ---
        plt.subplot(2, 1)
        plt.plot(t, accels, marker="dot", color="magenta")
        plt.title("Accel (m/s^2)")
        plt.theme('dark')
        plt.ticks_color('white')
        plt.grid(True, True)
        
        # 计算量程
        max_abs_a = max([abs(x) for x in accels]) if accels else 0
        limit_a = max(max_abs_a * 1.05, 0.5) 
        plt.ylim(-limit_a, limit_a)
        
        # [修复] 手动生成刻度列表
        a_ticks = np.linspace(-limit_a, limit_a, DENSE_TICKS_COUNT).tolist()
        plt.yticks(a_ticks)
        
        # 红色零线
        plt.hline(0, color="red") 
        
        # 3. 显示
        plt.show()
