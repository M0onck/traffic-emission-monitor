import numpy as np
from collections import deque
from perception.math.kalman_filter import KalmanFilterCV

class SGEstimator:
    """
    [数学组件] Savitzky-Golay 滤波器
    功能：基于滑动窗口多项式拟合，计算平滑的速度和加速度。
    特点：通过拟合过去 N 帧的位置，解析求导得到中间时刻的物理量，极大抑制差分噪声。
    """
    def __init__(self, window_size=15, dt=0.033, poly_order=2):
        self.window_size = window_size
        self.dt = dt
        self.poly_order = poly_order
        # 存储历史平滑位置 (x, y)
        self.x_history = deque(maxlen=window_size)
        self.y_history = deque(maxlen=window_size)
        
        # 预计算时间轴 (以窗口中心为 0，单位: 秒)
        # 例如 window=15, indices=[-7, -6, ..., 0, ..., 7]
        # 但我们在实时流中只能拿到过去的数据: [-(N-1), ..., 0]
        # 为了获得最佳平滑度，我们计算 "延迟 N/2 帧" 的时刻（窗口中心）
        self.time_axis = np.arange(window_size) * dt
        self.center_idx = window_size // 2  # 取中间点索引

    def update(self, pos_x, pos_y):
        self.x_history.append(pos_x)
        self.y_history.append(pos_y)
        
        # 1. 预热期：数据不足时返回 0
        if len(self.x_history) < self.window_size:
            return 0.0, 0.0

        # 2. 准备数据
        xs = np.array(self.x_history)
        ys = np.array(self.y_history)
        
        # 3. 多项式拟合 (x = at^2 + bt + c)
        # np.polyfit 返回系数 [a, b, c] (最高次幂在前)
        coeffs_x = np.polyfit(self.time_axis, xs, self.poly_order)
        coeffs_y = np.polyfit(self.time_axis, ys, self.poly_order)
        
        # 4. 解析求导 (计算窗口中心时刻 t_mid 的导数)
        # 原因：窗口两端的拟合误差最大（龙格现象），中间最准。
        # 代价：引入了 (window_size/2 * dt) 的时间延迟。
        t_mid = self.time_axis[self.center_idx]
        
        # 速度 v(t) = 2at + b
        vx = 2 * coeffs_x[0] * t_mid + coeffs_x[1]
        vy = 2 * coeffs_y[0] * t_mid + coeffs_y[1]
        
        # 加速度 a(t) = 2a (常加速度模型下为常数)
        ax = 2 * coeffs_x[0]
        ay = 2 * coeffs_y[0]
        
        # 5. 合成标量
        speed = np.sqrt(vx**2 + vy**2)
        
        # 切向加速度 (Tangential Acceleration)
        # 将加速度向量投影到速度方向: a_tan = (v . a) / |v|
        if speed > 0.1:
            accel = (vx * ax + vy * ay) / speed
        else:
            accel = 0.0
            
        return speed, accel


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
        
        # 提取参数
        params = config.get("kinematics", {})
        # SG 窗口大小直接复用配置中的 accel_window (建议 15~30)
        self.sg_window = params.get("accel_window", 15)
        self.border_margin = params.get("border_margin", 20)
        self.min_tracking_frames = params.get("min_tracking_frames", 10)
        self.max_physical_accel = params.get("max_physical_accel", 6.0)
        
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
            
            # --- Stage 1: Kalman Filter 位置去抖 ---
            kf = self.trackers[tid]['kf']
            state = kf.update(point)
            # 获取 KF 平滑后的位置 (x, y)
            smooth_pos_x, smooth_pos_y = state[0], state[1]
            
            # --- Stage 2: Savitzky-Golay 速度/加速度估算 ---
            sg = self.trackers[tid]['sg']
            speed, accel = sg.update(smooth_pos_x, smooth_pos_y)

            # Gate A: 边缘检测
            x1, y1, x2, y2 = box
            if (x1 < self.border_margin or y1 < self.border_margin or 
                x2 > img_w - self.border_margin or y2 > img_h - self.border_margin):
                continue 

            # Gate B: 预热期 (需要等待 SG 窗口填满)
            # 只有当 SG 有输出了(speed不为0或帧数够了)，才开始输出数据
            if self.active_frames[tid] < max(self.min_tracking_frames, self.sg_window):
                continue 

            # Gate C: 物理极限过滤
            if abs(accel) > self.max_physical_accel:
                # 如果加速度异常大，可能意味着 SG 拟合过冲，保持上一帧或置0
                accel = 0.0 

            results[tid] = {"speed": speed, "accel": accel}
            
        return results
