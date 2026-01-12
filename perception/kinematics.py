import numpy as np
from collections import deque
from core.filters import KalmanFilterCV

class KinematicsEstimator:
    """
    [感知] 运动学估算器
    已重构：支持依赖注入
    """
    def __init__(self, config: dict):
        """
        :param config: 包含 kinematics 和 system fps 的字典
        """
        self.fps = config.get("fps", 30)
        
        # 提取参数
        params = config.get("kinematics", {})
        self.speed_window = params.get("speed_window", 15)
        self.accel_window = params.get("accel_window", 15)
        self.border_margin = params.get("border_margin", 20)
        self.min_tracking_frames = params.get("min_tracking_frames", 10)
        self.max_physical_accel = params.get("max_physical_accel", 6.0)
        
        self.trackers = {}     
        self.active_frames = {} 

    def update(self, detections, points_transformed, frame_shape):
        results = {}
        img_h, img_w = frame_shape[:2]
        
        for tid, box, point in zip(detections.tracker_id, detections.xyxy, points_transformed):
            if tid not in self.trackers:
                self.trackers[tid] = {
                    'kf': KalmanFilterCV(point, dt=1.0/self.fps),
                    'speed_history': deque(maxlen=self.accel_window)
                }
                self.active_frames[tid] = 0
            
            self.active_frames[tid] += 1
            
            kf = self.trackers[tid]['kf']
            state = kf.update(point)
            vx, vy = state[2], state[3]
            smooth_speed = np.sqrt(vx**2 + vy**2)
            if smooth_speed < 0.2: smooth_speed = 0.0
            
            self.trackers[tid]['speed_history'].append(smooth_speed)
            accel = self._calc_accel(self.trackers[tid]['speed_history'], smooth_speed)

            # Gate A: 边缘检测 (使用注入的参数)
            x1, y1, x2, y2 = box
            if (x1 < self.border_margin or y1 < self.border_margin or 
                x2 > img_w - self.border_margin or y2 > img_h - self.border_margin):
                continue 

            # Gate B: 预热期
            if self.active_frames[tid] < self.min_tracking_frames:
                continue 

            # Gate C: 物理极限
            if abs(accel) > self.max_physical_accel:
                continue 

            results[tid] = {"speed": smooth_speed, "accel": accel}
            
        return results

    def _calc_accel(self, history, current_speed):
        if len(history) < self.accel_window:
            return 0.0
        v_old = history[0]
        time_span = (self.accel_window - 1) / self.fps
        if time_span <= 0: return 0.0
        accel = (current_speed - v_old) / time_span
        if current_speed == 0 or abs(accel) < 0.2: return 0.0
        return accel
