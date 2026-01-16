import numpy as np
from collections import deque
from perception.math.kalman_filter import KalmanFilterCV

class SGEstimator:
    """
    [数学组件] 增强型运动学滤波器 (Dual-Window SG + Robust Slope + Phase Alignment)
    
    架构改进：
    1. 双窗口机制：分离 '位置平滑窗口' 和 '速度平滑窗口'。
    2. 鲁棒求导：使用 Theil-Sen (中位数斜率) 计算加速度。
    3. 惯性平滑：对最终加速度应用 EMA。
    4. 相位对齐：通过延迟速度输出，消除两级滤波带来的 v/a 时间错位。
    """
    def __init__(self, pos_window=15, speed_window=30, dt=0.033, poly_order=2, ema_alpha=0.2):
        """
        :param pos_window: 位置历史窗口 (计算速度用)
        :param speed_window: 速度历史窗口 (计算加速度用)
        :param dt: 采样时间间隔 (s)
        :param poly_order: 位置拟合的多项式阶数
        :param ema_alpha: EMA平滑系数
        """
        self.pos_window = pos_window
        self.speed_window = speed_window
        self.dt = dt
        self.poly_order = poly_order
        self.ema_alpha = ema_alpha
        
        # 1. 位置历史 (用于计算速度)
        self.x_history = deque(maxlen=pos_window)
        self.y_history = deque(maxlen=pos_window)
        
        # 2. 速度历史 (用于计算加速度趋势)
        self.speed_history = deque(maxlen=speed_window)
        
        # 3. [新增] 速度延时队列 (用于相位对齐)
        # 加速度计算使用的是 speed_window 的整体趋势，物理意义对应窗口中心。
        # 因此加速度比速度滞后了 speed_window / 2 帧。
        # 我们需要让输出的速度也“等一等”，滞后同样的帧数，以保证 v 和 a 物理时刻一致。
        self.lag_compensation = speed_window // 2
        self.speed_delay_queue = deque(maxlen=self.lag_compensation + 1)
        
        # 预计算位置拟合的时间轴
        self.time_axis_pos = np.arange(pos_window) * dt
        self.center_idx_pos = pos_window // 2
        
        # 状态记忆
        self.last_accel = 0.0

    def _robust_slope(self, y_series, dt):
        """
        [核心算法] 使用简化版 Theil-Sen 估算器计算斜率 (dv/dt)
        """
        n = len(y_series)
        if n < 3: return 0.0
        
        slopes = []
        # 采样策略：计算相邻点、跨度N/3点、跨度N/2点的斜率
        steps = list(set([1, max(1, n//3), max(1, n//2)]))
        
        for step in steps:
            if step >= n: continue
            for i in range(n - step):
                dy = y_series[i+step] - y_series[i]
                dx = step * dt
                slopes.append(dy / dx)
            
        if not slopes:
            return 0.0
            
        return np.median(slopes)

    def update(self, pos_x, pos_y):
        """
        更新观测并返回 (aligned_speed, accel)
        注意：返回的速度是经过对齐补偿的，比画面滞后约 0.5-0.7秒。
        """
        # --- Stage 1: 数据录入 ---
        self.x_history.append(pos_x)
        self.y_history.append(pos_y)
        
        # 预热期
        if len(self.x_history) < self.pos_window:
            return 0.0, 0.0

        # --- Stage 2: 计算速度 (Pos -> Speed) ---
        xs = np.array(self.x_history)
        ys = np.array(self.y_history)
        
        coeffs_x = np.polyfit(self.time_axis_pos, xs, self.poly_order)
        coeffs_y = np.polyfit(self.time_axis_pos, ys, self.poly_order)
        
        t_mid = self.time_axis_pos[self.center_idx_pos]
        
        if self.poly_order == 2:
            vx = 2 * coeffs_x[0] * t_mid + coeffs_x[1]
            vy = 2 * coeffs_y[0] * t_mid + coeffs_y[1]
        else: # poly_order = 3
            vx = 3 * coeffs_x[0] * t_mid**2 + 2 * coeffs_x[1] * t_mid + coeffs_x[2]
            vy = 3 * coeffs_y[0] * t_mid**2 + 2 * coeffs_y[1] * t_mid + coeffs_y[2]
        
        current_speed = np.sqrt(vx**2 + vy**2)
        self.speed_history.append(current_speed)

        # --- Stage 3: 计算加速度 (Speed -> Accel) ---
        raw_accel = 0.0
        min_accel_samples = max(5, self.speed_window // 3)
        
        if len(self.speed_history) >= min_accel_samples:
            recent_speeds = np.array(self.speed_history)
            raw_accel = self._robust_slope(recent_speeds, self.dt)
            
            if current_speed < 0.2: 
                raw_accel = 0.0

        # --- Stage 4: EMA 惯性平滑 ---
        final_accel = self.ema_alpha * raw_accel + (1 - self.ema_alpha) * self.last_accel
        self.last_accel = final_accel
        
        # --- Stage 5: [新增] 相位对齐补偿 ---
        # 将最新算出的速度放入延时队列
        self.speed_delay_queue.append(current_speed)
        
        # 取出滞后的速度，使其物理时刻与 final_accel 对齐
        # 如果队列未满(刚开始)，暂时用当前速度兜底
        aligned_speed = current_speed
        if len(self.speed_delay_queue) > self.lag_compensation:
            aligned_speed = self.speed_delay_queue[0]
            
        return aligned_speed, final_accel


class KinematicsEstimator:
    """
    [感知层] 运动学估算器 (KF + Robust SG)
    """
    def __init__(self, config: dict):
        self.fps = config.get("fps", 30)
        dt = 1.0 / self.fps
        
        params = config.get("kinematics", {})
        
        # 分离读取窗口配置
        self.speed_window = params.get("speed_window", 15)
        self.accel_window = params.get("accel_window", 30)
        
        self.border_margin = params.get("border_margin", 20)
        self.min_tracking_frames = params.get("min_tracking_frames", 10)
        self.max_physical_accel = params.get("max_physical_accel", 6.0)
        
        self.poly_order = params.get("poly_order", 3)
        
        self.trackers = {}     
        self.active_frames = {}
        self.dt = dt
        
        self.last_raw_pixels = {} 

    def update(self, detections, points_transformed, frame_shape):
        results = {}
        img_h, img_w = frame_shape[:2]
        
        raw_boxes = detections.xyxy
        if len(raw_boxes) > 0:
            raw_centers_x = (raw_boxes[:, 0] + raw_boxes[:, 2]) / 2
            raw_centers_y = (raw_boxes[:, 1] + raw_boxes[:, 3]) / 2
        else:
            raw_centers_x, raw_centers_y = [], []
        
        for i, (tid, point) in enumerate(zip(detections.tracker_id, points_transformed)):
            if tid not in self.trackers:
                self.trackers[tid] = {
                    'kf': KalmanFilterCV(point, dt=self.dt),
                    'sg': SGEstimator(
                        pos_window=self.speed_window, 
                        speed_window=self.accel_window, 
                        dt=self.dt, 
                        poly_order=self.poly_order,
                        ema_alpha=0.2
                    )
                }
                self.active_frames[tid] = 0
                self.last_raw_pixels[tid] = (raw_centers_x[i], raw_centers_y[i])
            
            self.active_frames[tid] += 1
            
            # --- 静态抑制 (Pixel-level Stationary Check) ---
            curr_pixel = (raw_centers_x[i], raw_centers_y[i])
            prev_pixel = self.last_raw_pixels.get(tid, curr_pixel)
            pixel_dist = np.hypot(curr_pixel[0]-prev_pixel[0], curr_pixel[1]-prev_pixel[1])
            self.last_raw_pixels[tid] = curr_pixel
            
            is_moving = pixel_dist > 0.5 

            # --- Stage 1: Kalman Filter ---
            kf = self.trackers[tid]['kf']
            state = kf.update(point)
            smooth_pos_x, smooth_pos_y = state[0], state[1]
            
            # --- Stage 2: SG Estimator (带对齐) ---
            sg = self.trackers[tid]['sg']
            speed, accel = sg.update(smooth_pos_x, smooth_pos_y)

            # 强制静态抑制
            if not is_moving and speed < 1.0:
                speed = 0.0
                accel = 0.0
                sg.last_accel = 0.0
                # 同时清空延时队列，防止滞后的速度在停车后继续输出
                sg.speed_delay_queue.clear()

            # Gate A: 边缘检测
            box = raw_boxes[i]
            x1, y1, x2, y2 = box
            if (x1 < self.border_margin or y1 < self.border_margin or 
                x2 > img_w - self.border_margin or y2 > img_h - self.border_margin):
                continue 

            # Gate B: 预热期
            max_window = max(self.speed_window, self.accel_window)
            if self.active_frames[tid] < max(self.min_tracking_frames, max_window):
                continue 

            # Gate C: 物理截断
            accel = np.clip(accel, -self.max_physical_accel, self.max_physical_accel)

            results[tid] = {"speed": speed, "accel": accel}
            
        return results
