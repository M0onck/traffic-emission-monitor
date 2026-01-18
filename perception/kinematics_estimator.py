import numpy as np
from collections import deque
from perception.math.kalman_filter import KalmanFilterCV

class AdaptiveSGEstimator:
    def __init__(self, pos_window=11, speed_window=1, dt=0.033, poly_order=2):
        self.pos_window = pos_window
        self.speed_window = speed_window 
        self.dt = dt
        self.poly_order = poly_order
        self.x_history = deque(maxlen=pos_window)
        self.y_history = deque(maxlen=pos_window)
        self.t_axis_pos = np.arange(pos_window) * dt
        self.speed_history = deque(maxlen=speed_window)
        self.last_speed = 0.0
        self.last_accel = 0.0
        self.lag_compensation = speed_window // 2
        self.speed_delay_queue = deque(maxlen=self.lag_compensation + 1)

    def _sg_derivative(self, y_data, t_axis):
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
        """
        [修改版] 计算测量置信度 (Cap Confidence Strategy)
        策略：强制降低对视觉测量的信任上限，增强惯性。
        """
        TOP_EDGE = 0.15     
        BOTTOM_EDGE = 0.85  
        MIN_CONF = 0.1      
        
        # [核心修改] 设置最大置信度上限
        # 0.3 意味着：最终速度 = 0.3 * 观测值 + 0.7 * 预测值
        MAX_CONF = 0.3      
        
        if TOP_EDGE <= y_norm <= BOTTOM_EDGE:
            # 中间区域返回设定的上限，而不是 1.0
            return MAX_CONF
        
        elif y_norm < TOP_EDGE:
            ratio = max(0.0, y_norm) / TOP_EDGE
            return MIN_CONF + ratio * (MAX_CONF - MIN_CONF)
            
        else: 
            dist_to_bottom = 1.0 - y_norm
            range_len = 1.0 - BOTTOM_EDGE
            ratio = max(0.0, dist_to_bottom) / range_len
            return MIN_CONF + ratio * (MAX_CONF - MIN_CONF)

    def update(self, pos_x, pos_y, y_norm):
        # ... (Stage 1 ~ 4 保持不变) ...
        self.x_history.append(pos_x)
        self.y_history.append(pos_y)
        if len(self.x_history) < self.pos_window: return 0.0, 0.0

        vx_meas = self._sg_derivative(list(self.x_history), self.t_axis_pos)
        vy_meas = self._sg_derivative(list(self.y_history), self.t_axis_pos)
        speed_meas = np.sqrt(vx_meas**2 + vy_meas**2)
        self.speed_history.append(speed_meas)
        
        accel_meas = 0.0
        min_accel_window = 10
        if len(self.speed_history) >= min_accel_window:
            v_now = self.speed_history[-1]
            v_old = self.speed_history[0]
            dt_span = (len(self.speed_history) - 1) * self.dt
            accel_meas = (v_now - v_old) / dt_span

        speed_pred = self.last_speed + self.last_accel * self.dt
        accel_pred = self.last_accel

        # --- Stage 5: 动态融合 ---
        w = self._calculate_confidence(y_norm)
        
        # [额外修改] 如果测量加速度剧烈跳变，进一步降低权重
        if abs(accel_meas - self.last_accel) > 1.5:
            w *= 0.2

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
        
        # [核心修改] 增大默认平滑窗口
        # 将默认窗口从 15 提升到 31 (约 1秒)，大幅平滑高频抖动
        self.speed_window = params.get("speed_window", 31)
        self.accel_window = params.get("accel_window", 31)
        
        self.border_margin = params.get("border_margin", 20)
        self.min_tracking_frames = params.get("min_tracking_frames", 10)
        self.max_physical_accel = params.get("max_physical_accel", 6.0)
        self.poly_order = params.get("poly_order", 2)
        
        self.trackers = {}     
        self.active_frames = {}
        self.dt = dt
        self.last_raw_pixels = {} 

    def update(self, detections, points_transformed, frame_shape, roi_y_range=None):
        # ... (update 逻辑保持不变) ...
        results = {}
        img_h, img_w = frame_shape[:2]
        roi_min_y, roi_max_y = 0.0, float(img_h)
        if roi_y_range: roi_min_y, roi_max_y = roi_y_range
        roi_height = max(1.0, roi_max_y - roi_min_y)

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
                        speed_window=self.accel_window,
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
            y_norm = (y_pixel - roi_min_y) / roi_height
            
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
                "speed": speed, "accel": accel,
                "curr_x": smooth_pos_x, "curr_y": smooth_pos_y
            }
        return results
