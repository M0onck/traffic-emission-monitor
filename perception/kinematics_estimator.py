import numpy as np
from collections import deque
from perception.math.kalman_filter import KalmanFilterCV

class SGEstimator:
    """
    [数学组件] Savitzky-Golay 滤波器 (两级级联版)
    功能：
    1. 位置 -> 速度：拟合位置曲线，求一阶导数得到速度。
    2. 速度 -> 加速度：拟合速度曲线，求一阶导数得到加速度。
    """
    def __init__(self, window_size=15, dt=0.033, poly_order=2):
        self.window_size = window_size
        self.dt = dt
        self.poly_order = poly_order
        
        # 历史数据容器
        self.x_history = deque(maxlen=window_size)
        self.y_history = deque(maxlen=window_size)
        
        # [Step 2] 速度序列容器 (v_seq)
        # 窗口可以独立设置，这里复用 window_size 以保持一致性
        self.speed_history = deque(maxlen=window_size)
        
        # 预计算时间轴 (以窗口中心为 0)
        self.time_axis = np.arange(window_size) * dt
        self.center_idx = window_size // 2

    def update(self, pos_x, pos_y):
        """
        更新观测并返回 (speed, accel)
        """
        self.x_history.append(pos_x)
        self.y_history.append(pos_y)
        
        # 预热期
        if len(self.x_history) < self.window_size:
            return 0.0, 0.0

        # --- Stage A: 计算速度 (Pos -> Speed) ---
        xs = np.array(self.x_history)
        ys = np.array(self.y_history)
        
        # SG 滤波：多项式拟合位置
        coeffs_x = np.polyfit(self.time_axis, xs, self.poly_order)
        coeffs_y = np.polyfit(self.time_axis, ys, self.poly_order)
        
        # 解析求导 (计算窗口中心的瞬时速度)
        # v = dx/dt = 2at + b
        t_mid = self.time_axis[self.center_idx]
        vx = 2 * coeffs_x[0] * t_mid + coeffs_x[1]
        vy = 2 * coeffs_y[0] * t_mid + coeffs_y[1]
        
        # 合成标量速度
        current_speed = np.sqrt(vx**2 + vy**2)
        
        # [Step 2] 存入速度序列
        self.speed_history.append(current_speed)

        # --- Stage B: 计算加速度 (Speed -> Accel) ---
        # [Step 3] 对 v_seq 应用 SG 滤波并求导 (deriv=1)
        accel = 0.0
        
        # 至少积累一定量的速度数据才开始计算加速度 (减少冷启动噪声)
        if len(self.speed_history) >= 5: 
            recent_speeds = np.array(self.speed_history)
            
            # 动态生成当前数据长度的时间轴
            n_points = len(recent_speeds)
            t_axis_v = np.arange(n_points) * self.dt
            
            # 使用线性拟合 (Order 1) 作为 SG 滤波器核
            # 理由：对于速度求导得到加速度，线性回归是最稳健的平滑微分算子，
            # 相当于 SG(window, poly_order=1, deriv=1)
            # v(t) = a*t + b -> dv/dt = a
            v_coeffs = np.polyfit(t_axis_v, recent_speeds, 1)
            
            # 斜率即为加速度
            raw_accel = v_coeffs[0]
            
            # 静止抑制：如果速度极小，强制加速度为0
            if current_speed < 0.1:
                accel = 0.0
            else:
                accel = raw_accel
            
        return current_speed, accel


class KinematicsEstimator:
    """
    [感知层] 运动学估算器 (KF + SG 级联版)
    """
    def __init__(self, config: dict):
        """
        :param config: 包含 kinematics 和 system fps 的字典
        """
        self.fps = config.get("fps", 30)
        dt = 1.0 / self.fps
        
        params = config.get("kinematics", {})
        self.sg_window = params.get("accel_window", 15)
        self.border_margin = params.get("border_margin", 20)
        self.min_tracking_frames = params.get("min_tracking_frames", 10)
        
        # [Step 4] 物理截断阈值 (默认 8.0 m/s², 约 0.8g)
        self.max_physical_accel = params.get("max_physical_accel", 8.0)
        
        self.trackers = {}     
        self.active_frames = {}
        self.dt = dt

    def update(self, detections, points_transformed, frame_shape):
        results = {}
        img_h, img_w = frame_shape[:2]
        
        for tid, box, point in zip(detections.tracker_id, detections.xyxy, points_transformed):
            if tid not in self.trackers:
                # 初始化：KF 用于位置，SG 用于速度/加速度
                self.trackers[tid] = {
                    'kf': KalmanFilterCV(point, dt=self.dt),
                    'sg': SGEstimator(window_size=self.sg_window, dt=self.dt, poly_order=2)
                }
                self.active_frames[tid] = 0
            
            self.active_frames[tid] += 1
            
            # --- [Step 1] 使用 Kalman Filter 对原始坐标去噪 ---
            kf = self.trackers[tid]['kf']
            state = kf.update(point)
            smooth_pos_x, smooth_pos_y = state[0], state[1]
            
            # --- [Step 2 & 3] 计算 v_seq 并应用 SG 滤波求导 ---
            sg = self.trackers[tid]['sg']
            speed, accel = sg.update(smooth_pos_x, smooth_pos_y)

            # Gate A: 边缘检测 (保持不变)
            x1, y1, x2, y2 = box
            if (x1 < self.border_margin or y1 < self.border_margin or 
                x2 > img_w - self.border_margin or y2 > img_h - self.border_margin):
                continue 

            # Gate B: 预热期
            if self.active_frames[tid] < max(self.min_tracking_frames, self.sg_window):
                continue 

            # Gate C: [Step 4] 物理截断 (Clamping)
            # 将加速度强制限制在合理范围内 (例如 -8.0 到 +8.0)
            # 相比置0，截断能保留“大急刹”的物理意义，同时去除 4g 这种离谱噪声
            accel = np.clip(accel, -self.max_physical_accel, self.max_physical_accel)

            results[tid] = {"speed": speed, "accel": accel}
            
        return results
