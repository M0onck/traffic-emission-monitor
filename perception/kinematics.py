import numpy as np
from collections import deque
from core.filters import KalmanFilterCV
import config.settings as cfg

class KinematicsEstimator:
    """
    [感知] 运动学估算器。
    功能：维护多目标跟踪的卡尔曼滤波器状态，计算速度和加速度，并执行数据门控。
    """
    def __init__(self, fps: int, speed_window: int, accel_window: int):
        self.fps = fps
        self.accel_window = accel_window
        
        self.trackers = {}      # {tid: {'kf': filter, 'speed_history': deque}}
        self.active_frames = {} # {tid: frame_count}

    def update(self, detections, points_transformed, frame_shape):
        """
        更新追踪状态，输出清洗后的运动数据。
        """
        results = {}
        img_h, img_w = frame_shape[:2]
        
        for tid, box, point in zip(detections.tracker_id, detections.xyxy, points_transformed):
            # 1. 初始化
            if tid not in self.trackers:
                self.trackers[tid] = {
                    'kf': KalmanFilterCV(point, dt=1.0/self.fps, 
                                         process_noise_scale=0.5, 
                                         measurement_noise_scale=50.0),
                    'speed_history': deque(maxlen=self.accel_window)
                }
                self.active_frames[tid] = 0
            
            self.active_frames[tid] += 1
            
            # 2. KF 迭代 (始终执行以保持连续性)
            kf = self.trackers[tid]['kf']
            state = kf.update(point)
            vx, vy = state[2], state[3]
            
            smooth_speed = np.sqrt(vx**2 + vy**2)
            if smooth_speed < 0.2: smooth_speed = 0.0 # 死区归零
            
            # 3. 计算加速度
            self.trackers[tid]['speed_history'].append(smooth_speed)
            accel = self._calc_accel(self.trackers[tid]['speed_history'], smooth_speed)

            # 4. 数据门控 (Gating)
            # Gate A: 边缘检测
            x1, y1, x2, y2 = box
            if (x1 < cfg.BORDER_MARGIN or y1 < cfg.BORDER_MARGIN or 
                x2 > img_w - cfg.BORDER_MARGIN or y2 > img_h - cfg.BORDER_MARGIN):
                continue 

            # Gate B: 预热期
            if self.active_frames[tid] < cfg.MIN_TRACKING_FRAMES:
                continue 

            # Gate C: 物理极限过滤
            if abs(accel) > cfg.MAX_PHYSICAL_ACCEL:
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
