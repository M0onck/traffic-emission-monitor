import numpy as np
from collections import deque
from perception.math.kalman_filter import KalmanFilterCV

class LinearAccelEstimator:
    """
    [数学组件] 简易加速度计算器 (原 AdaptiveSGEstimator 的简化版)
    
    变更说明：
    根据 v5.4 需求，实时层不再进行复杂的磁性融合。
    本类现在仅负责维护速度历史，用于计算平滑的实时加速度。
    """
    def __init__(self, speed_window=31, dt=0.033):
        """
        :param speed_window: 加速度计算窗口大小
        :param dt: 采样时间间隔
        """
        self.speed_window = speed_window
        self.dt = dt
        
        # 仅维护速度历史，用于微分计算加速度
        self.speed_history = deque(maxlen=speed_window)

    def update(self, kf_speed):
        """
        :param kf_speed: 来自卡尔曼滤波器的稳态速度
        :return: (speed, accel)
        """
        # 1. 直接采纳 KF 速度
        current_speed = kf_speed
        self.speed_history.append(current_speed)
        
        # 2. 计算加速度 (长窗口线性斜率)
        accel = 0.0
        min_accel_window = 5 # 降低门槛，使其在启动初期也能有数值
        
        if len(self.speed_history) >= min_accel_window:
            # 取队首和队尾计算平均斜率
            v_now = self.speed_history[-1]
            v_old = self.speed_history[0]
            dt_span = (len(self.speed_history) - 1) * self.dt
            
            if dt_span > 1e-4:
                accel = (v_now - v_old) / dt_span

        return current_speed, accel


class KinematicsEstimator:
    """
    [感知层] 运动学估算器 (实时版 - v5.4 Simplified)
    """
    def __init__(self, config: dict):
        self.fps = config.get("fps", 30)
        dt = 1.0 / self.fps
        
        params = config.get("kinematics", {})
        # 依然保持较大的窗口以获得平滑的加速度
        self.speed_window = params.get("speed_window", 31) 
        self.border_margin = params.get("border_margin", 20)
        self.min_tracking_frames = params.get("min_tracking_frames", 10)
        self.max_physical_accel = params.get("max_physical_accel", 6.0)
        
        self.trackers = {}     
        self.active_frames = {}
        self.dt = dt
        self.last_raw_pixels = {} 

    def update(self, detections, points_transformed, frame_shape, roi_y_range=None):
        results = {}
        img_h, img_w = frame_shape[:2]
        
        raw_boxes = detections.xyxy
        if len(raw_boxes) > 0:
            raw_centers_x = (raw_boxes[:, 0] + raw_boxes[:, 2]) / 2
            raw_centers_y = (raw_boxes[:, 1] + raw_boxes[:, 3]) / 2
        else:
            raw_centers_x, raw_centers_y = [], []
        
        for i, (tid, point) in enumerate(zip(detections.tracker_id, points_transformed)):
            # 1. 初始化
            if tid not in self.trackers:
                self.trackers[tid] = {
                    'kf': KalmanFilterCV(point, dt=self.dt), # KF 参数保持 Q=1e-4 即可
                    'sg': LinearAccelEstimator(
                        speed_window=self.speed_window, 
                        dt=self.dt
                    )
                }
                self.active_frames[tid] = 0
                self.last_raw_pixels[tid] = (raw_centers_x[i], raw_centers_y[i])
            
            self.active_frames[tid] += 1
            
            # 2. 静止检测
            curr_pixel = (raw_centers_x[i], raw_centers_y[i])
            prev_pixel = self.last_raw_pixels.get(tid, curr_pixel)
            pixel_dist = np.hypot(curr_pixel[0]-prev_pixel[0], curr_pixel[1]-prev_pixel[1])
            self.last_raw_pixels[tid] = curr_pixel
            is_moving = pixel_dist > 0.5 

            # 3. 卡尔曼滤波 (核心)
            kf = self.trackers[tid]['kf']
            state = kf.update(point)
            smooth_pos_x, smooth_pos_y = state[0], state[1]
            
            # 提取 KF 速度 (这是最"稳"的速度)
            kf_vel_x, kf_vel_y = state[2], state[3]
            kf_speed = np.sqrt(kf_vel_x**2 + kf_vel_y**2)
            
            # 4. 加速度计算
            sg = self.trackers[tid]['sg']
            speed, accel = sg.update(kf_speed) # 直接传入 KF 速度

            # 5. 静止归零
            if not is_moving and speed < 1.0:
                speed = 0.0
                accel = 0.0
                # 清空历史以防下一次启动时有残留
                sg.speed_history.clear()

            # 6. 过滤与限幅
            box = raw_boxes[i]
            x1, y1, x2, y2 = box
            if (x1 < self.border_margin or y1 < self.border_margin or 
                x2 > img_w - self.border_margin or y2 > img_h - self.border_margin):
                continue 

            if self.active_frames[tid] < self.min_tracking_frames:
                continue 

            accel = np.clip(accel, -self.max_physical_accel, self.max_physical_accel)

            results[tid] = {
                "speed": speed, 
                "accel": accel,
                "curr_x": smooth_pos_x,
                "curr_y": smooth_pos_y
            }
            
        return results
