import math
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
        [可视化优化版] 绘制运动学曲线
        特性：
        1. 速度轴：阶梯式纵坐标，白色线框。
        2. 加速度轴：对称纵坐标(±N)，红线居中。
        3. [修复] 强制使用奇数行高度，解决零线显示问题。
        """
        # 1. 获取终端尺寸并自适应
        term_w, term_h = plt.terminal_size()
        safe_w = min(term_w - 5, 100) 
        
        # [核心修改] 将高度固定为奇数 (21行)
        # 原理：ASCII字符绘图中，只有奇数高度的画布才拥有物理上的"正中间那一行"。
        # 偶数高度会导致 y=0 落在两行字符之间，导致红线渲染不清晰或消失。
        safe_h = 21 
        
        # 窗口过小检查 (稍微调大需求，确保容纳21行)
        if safe_w < 40 or term_h < 26:
            return 

        # 2. 清除画布与配置
        plt.clear_figure()
        plt.plotsize(safe_w, safe_h)
        plt.subplots(2, 1)
        
        # 准备时间轴
        t = [i / self.fps for i in range(len(speeds))]
        
        # --- 子图 1: 速度曲线 (Top) ---
        plt.subplot(1, 1)
        plt.plot(t, speeds, marker="dot", color="cyan")
        plt.title("Speed (m/s)")
        plt.theme('dark')
        plt.ticks_color('white') 
        plt.grid(True, True)
        
        # 速度纵坐标: 0 ~ N (步长 5)
        max_v = max(speeds) if speeds else 0
        if max_v > 0:
            y_limit_v = math.ceil(max_v / 5.0) * 5.0
            y_limit_v = max(y_limit_v, 5.0)
            plt.ylim(0, y_limit_v)

        # --- 子图 2: 加速度曲线 (Bottom) ---
        plt.subplot(2, 1)
        plt.plot(t, accels, marker="dot", color="magenta")
        plt.title("Accel (m/s^2)")
        plt.theme('dark')
        plt.ticks_color('white')
        plt.grid(True, True)
        
        # 加速度纵坐标: 对称模式 ±N (步长 2.0)
        # 配合奇数行高，0 线将精确落在中间一行
        max_abs_a = max([abs(x) for x in accels]) if accels else 0
        a_limit = math.ceil(max_abs_a / 2.0) * 2.0
        a_limit = max(a_limit, 2.0) 
        plt.ylim(-a_limit, a_limit)
        
        # 红色零线
        plt.hline(0, color="red") 
        
        # 3. 显示
        plt.show()
