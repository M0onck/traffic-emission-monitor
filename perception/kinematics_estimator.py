import numpy as np
from collections import deque
from perception.math.kalman_filter import KalmanFilterCV

class AdaptiveSGEstimator:
    """
    [数学组件] 透视自适应 SG 滤波器
    """
    def __init__(self, pos_window=15, speed_window=30, dt=0.033, poly_order=2):
        self.pos_window = pos_window
        self.speed_window = speed_window # 这里实际上被用作加速度计算窗口
        self.dt = dt
        self.poly_order = poly_order
        
        # 1. 位置历史
        self.x_history = deque(maxlen=pos_window)
        self.y_history = deque(maxlen=pos_window)
        self.t_axis_pos = np.arange(pos_window) * dt
        
        # 2. 速度历史
        self.speed_history = deque(maxlen=speed_window)
        # t_axis_spd 不需要了，我们将直接计算斜率
        
        # 3. 状态记忆
        self.last_speed = 0.0
        self.last_accel = 0.0
        
        self.lag_compensation = speed_window // 2
        self.speed_delay_queue = deque(maxlen=self.lag_compensation + 1)

    def _sg_derivative(self, y_data, t_axis):
        """Savitzky-Golay 多项式拟合求导 (仅用于位置->速度)"""
        n = len(y_data)
        if n < self.poly_order + 1: return 0.0
        
        t = t_axis[:n]
        coeffs = np.polyfit(t, y_data, self.poly_order)
        t_mid = t[n // 2]
        
        if self.poly_order == 2:
            deriv = 2 * coeffs[0] * t_mid + coeffs[1]
        else:
            deriv = 3 * coeffs[0] * t_mid**2 + 2 * coeffs[1] * t_mid + coeffs[2]
            
        return deriv

    def _calculate_confidence(self, y_norm):
        if y_norm > 0.4: return 1.0
        elif y_norm < 0.1: return 0.1 
        else: return 0.1 + (y_norm - 0.1) / (0.4 - 0.1) * 0.9

    def update(self, pos_x, pos_y, y_norm):
        # --- Stage 1: 数据录入 ---
        self.x_history.append(pos_x)
        self.y_history.append(pos_y)
        
        if len(self.x_history) < self.pos_window:
            return 0.0, 0.0

        # --- Stage 2: 速度计算 (保持 SG 滤波以获得平滑速度) ---
        vx_meas = self._sg_derivative(list(self.x_history), self.t_axis_pos)
        vy_meas = self._sg_derivative(list(self.y_history), self.t_axis_pos)
        speed_meas = np.sqrt(vx_meas**2 + vy_meas**2)
        
        self.speed_history.append(speed_meas)
        
        # --- Stage 3: 加速度计算 (修改为长窗口线性斜率) ---
        # 策略：不使用 SG 拟合曲线，而是直接计算过去 ~0.5s 的整体速度变化率
        accel_meas = 0.0
        min_accel_window = 10 # 至少积累10帧才开始算加速度
        
        if len(self.speed_history) >= min_accel_window:
            # 取当前速度与 N 帧前的速度
            v_now = self.speed_history[-1]
            v_old = self.speed_history[0] # 这里的 0 就是队列头部，即 window 帧之前
            
            # 计算时间跨度
            dt_span = (len(self.speed_history) - 1) * self.dt
            
            # 计算平均加速度 (Linear Slope)
            accel_meas = (v_now - v_old) / dt_span

        # --- Stage 4: 惯性预测 ---
        speed_pred = self.last_speed + self.last_accel * self.dt
        accel_pred = self.last_accel

        # --- Stage 5: 动态融合 ---
        w = self._calculate_confidence(y_norm)
        
        if abs(accel_meas - self.last_accel) > 2.0:
            w *= 0.5

        final_speed = w * speed_meas + (1 - w) * speed_pred
        final_accel = w * accel_meas + (1 - w) * accel_pred
        
        if final_speed < 0.5:
            final_accel = 0.0
            
        self.last_speed = final_speed
        self.last_accel = final_accel

        self.speed_delay_queue.append(final_speed)
        aligned_speed = final_speed
        if len(self.speed_delay_queue) > self.lag_compensation:
            aligned_speed = self.speed_delay_queue[0]
            
        return aligned_speed, final_accel

class KinematicsEstimator:
    def __init__(self, config: dict):
        self.fps = config.get("fps", 30)
        dt = 1.0 / self.fps
        
        params = config.get("kinematics", {})
        self.speed_window = params.get("speed_window", 15)
        self.accel_window = params.get("accel_window", 15)
        self.border_margin = params.get("border_margin", 20)
        self.min_tracking_frames = params.get("min_tracking_frames", 10)
        self.max_physical_accel = params.get("max_physical_accel", 6.0)
        self.poly_order = params.get("poly_order", 2)
        
        self.trackers = {}     
        self.active_frames = {}
        self.dt = dt
        self.last_raw_pixels = {} 

    def update(self, detections, points_transformed, frame_shape):
        results = {}
        img_h, img_w = frame_shape[:2]
        
        raw_boxes = detections.xyxy
        if len(raw_boxes) > 0:
            raw_bottom_y = raw_boxes[:, 3] 
            raw_centers_x = (raw_boxes[:, 0] + raw_boxes[:, 2]) / 2
            raw_centers_y = (raw_boxes[:, 1] + raw_boxes[:, 3]) / 2
        else:
            raw_bottom_y = []
            raw_centers_x, raw_centers_y = [], []
        
        for i, (tid, point) in enumerate(zip(detections.tracker_id, points_transformed)):
            if tid not in self.trackers:
                self.trackers[tid] = {
                    'kf': KalmanFilterCV(point, dt=self.dt),
                    'sg': AdaptiveSGEstimator(
                        pos_window=self.speed_window, 
                        speed_window=self.accel_window, # 使用配置的加速度窗口长度
                        dt=self.dt,
                        poly_order=self.poly_order
                    )
                }
                self.active_frames[tid] = 0
                self.last_raw_pixels[tid] = (raw_centers_x[i], raw_centers_y[i])
            
            self.active_frames[tid] += 1
            
            curr_pixel = (raw_centers_x[i], raw_centers_y[i])
            prev_pixel = self.last_raw_pixels.get(tid, curr_pixel)
            pixel_dist = np.hypot(curr_pixel[0]-prev_pixel[0], curr_pixel[1]-prev_pixel[1])
            self.last_raw_pixels[tid] = curr_pixel
            is_moving = pixel_dist > 0.8 

            kf = self.trackers[tid]['kf']
            state = kf.update(point)
            smooth_pos_x, smooth_pos_y = state[0], state[1]
            
            y_pixel = raw_bottom_y[i]
            y_norm = y_pixel / img_h
            
            sg = self.trackers[tid]['sg']
            speed, accel = sg.update(smooth_pos_x, smooth_pos_y, y_norm)

            if not is_moving and speed < 1.0:
                speed = 0.0
                accel = 0.0
                sg.last_accel = 0.0
                sg.speed_delay_queue.clear()

            box = raw_boxes[i]
            x1, y1, x2, y2 = box
            if (x1 < self.border_margin or y1 < self.border_margin or 
                x2 > img_w - self.border_margin or y2 > img_h - self.border_margin):
                continue 

            max_window = max(self.speed_window, self.accel_window)
            if self.active_frames[tid] < max(self.min_tracking_frames, max_window):
                continue 

            accel = np.clip(accel, -self.max_physical_accel, self.max_physical_accel)

            results[tid] = {
                "speed": speed, 
                "accel": accel,
                "curr_x": smooth_pos_x,
                "curr_y": smooth_pos_y
            }
            
        return results
