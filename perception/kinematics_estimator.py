import numpy as np
from collections import deque
from perception.math.kalman_filter import KalmanFilterCV

class AdaptiveSGEstimator:
    """
    [数学组件] 透视自适应 SG 滤波器 (Perspective-Adaptive SG Estimator)
    
    核心策略：
    1. 经典架构: 使用 Kalman 平滑位置，SG 多项式拟合计算速度/加速度。
    2. 透视抑制: 根据检测框在画面中的 Y 轴位置，动态调整测量置信度。
       - 近处 (Bottom): 置信度高，响应灵敏。
       - 远处 (Top): 置信度低，依赖惯性预测，抑制透视噪声。
    """
    def __init__(self, pos_window=15, speed_window=30, dt=0.033, poly_order=2):
        self.pos_window = pos_window
        self.speed_window = speed_window
        self.dt = dt
        self.poly_order = poly_order
        
        # 1. 位置历史 (用于拟合速度)
        self.x_history = deque(maxlen=pos_window)
        self.y_history = deque(maxlen=pos_window)
        self.t_axis_pos = np.arange(pos_window) * dt
        self.center_idx_pos = pos_window // 2
        
        # 2. 速度历史 (用于拟合加速度)
        self.speed_history = deque(maxlen=speed_window)
        self.t_axis_spd = np.arange(speed_window) * dt
        
        # 3. 状态记忆 (用于惯性预测)
        self.last_speed = 0.0
        self.last_accel = 0.0
        
        # 4. 相位对齐队列
        self.lag_compensation = speed_window // 2
        self.speed_delay_queue = deque(maxlen=self.lag_compensation + 1)

    def _sg_derivative(self, y_data, t_axis):
        """Savitzky-Golay 多项式拟合求导"""
        n = len(y_data)
        if n < self.poly_order + 1: return 0.0
        
        # 截取对应长度的时间轴
        t = t_axis[:n]
        # 拟合多项式: y = c0*t^2 + c1*t + c2
        coeffs = np.polyfit(t, y_data, self.poly_order)
        
        # 计算窗口中心的导数
        # t_mid = t[n // 2]  <-- 传统 SG 取中心
        # 为了更贴近实时性，这里取稍靠后的点，或者直接取末端(但这会引入噪声)
        # 这种架构下，我们还是取中心点以保证平滑，然后通过外部队列做相位对齐
        t_mid = t[n // 2]
        
        if self.poly_order == 2:
            # v = 2*c0*t + c1
            deriv = 2 * coeffs[0] * t_mid + coeffs[1]
        else:
            # v = 3*c0*t^2 + 2*c1*t + c2
            deriv = 3 * coeffs[0] * t_mid**2 + 2 * coeffs[1] * t_mid + coeffs[2]
            
        return deriv

    def _calculate_confidence(self, y_norm):
        """
        计算透视置信度
        :param y_norm: 归一化 Y 坐标 (0.0=Top/Far, 1.0=Bottom/Near)
        :return: confidence (0.0 ~ 1.0)
        """
        # 设定“危险区”：画面上部 30% 区域 (y < 0.3) 噪声极大
        # 设定“安全区”：画面下部 60% 区域 (y > 0.4) 较为可信
        
        if y_norm > 0.4:
            return 1.0
        elif y_norm < 0.1:
            return 0.1 # 极远处保留最低限度的更新
        else:
            # 中间区域线性过渡
            return 0.1 + (y_norm - 0.1) / (0.4 - 0.1) * 0.9

    def update(self, pos_x, pos_y, y_norm):
        """
        :param pos_x, pos_y: 卡尔曼平滑后的物理坐标
        :param y_norm: 原始检测框底部的归一化 Y 坐标 (用于判断远近)
        """
        # --- Stage 1: 数据录入 ---
        self.x_history.append(pos_x)
        self.y_history.append(pos_y)
        
        if len(self.x_history) < self.pos_window:
            return 0.0, 0.0

        # --- Stage 2: 纯视觉测量 (Measurement) ---
        # 使用 SG 滤波器从位置历史计算“测量速度”
        vx_meas = self._sg_derivative(list(self.x_history), self.t_axis_pos)
        vy_meas = self._sg_derivative(list(self.y_history), self.t_axis_pos)
        speed_meas = np.sqrt(vx_meas**2 + vy_meas**2)
        
        # 存入历史用于计算加速度
        self.speed_history.append(speed_meas)
        
        # 使用 SG 滤波器从速度历史计算“测量加速度”
        accel_meas = 0.0
        if len(self.speed_history) >= max(5, self.speed_window // 3):
            # 这里可以用线性回归(order=1)或者二次拟合(order=2)
            # 为了稳定性，计算加速度趋势建议用线性
            accel_meas = self._sg_derivative(list(self.speed_history), self.t_axis_spd)

        # --- Stage 3: 惯性预测 (Prediction) ---
        # 基于上一帧的状态，预测当前帧可能的速度和加速度
        # 模型：匀加速运动 v = v0 + a*t
        speed_pred = self.last_speed + self.last_accel * self.dt
        # 模型：加速度维持 (惯性)
        accel_pred = self.last_accel

        # --- Stage 4: 动态融合 (Fusion) ---
        # 获取置信度权重 w
        # 靠近画面边缘(y_norm -> 0)时，w -> 0，主要信赖 Prediction
        # 靠近画面底部(y_norm -> 1)时，w -> 1，主要信赖 Measurement
        w = self._calculate_confidence(y_norm)
        
        # 额外的物理约束：如果测量出的加速度极其离谱(例如透视跳变导致的 > 3m/s^2)，
        # 即使在近处也应该降低权重
        if abs(accel_meas - self.last_accel) > 2.0:
            w *= 0.5

        # 融合公式：State = w * Meas + (1-w) * Pred
        final_speed = w * speed_meas + (1 - w) * speed_pred
        final_accel = w * accel_meas + (1 - w) * accel_pred
        
        # 低速强制归零 (防止静止漂移)
        if final_speed < 0.5:
            final_accel = 0.0
            
        # 更新状态记忆
        self.last_speed = final_speed
        self.last_accel = final_accel

        # --- Stage 5: 相位对齐 (Output) ---
        # 将融合后的速度放入延时队列，使其与“计算过程滞后”的加速度在时间上对齐
        self.speed_delay_queue.append(final_speed)
        
        aligned_speed = final_speed
        if len(self.speed_delay_queue) > self.lag_compensation:
            aligned_speed = self.speed_delay_queue[0]
            
        return aligned_speed, final_accel


class KinematicsEstimator:
    """
    [感知层] 运动学估算器 (KF + Adaptive SG)
    """
    def __init__(self, config: dict):
        self.fps = config.get("fps", 30)
        dt = 1.0 / self.fps
        
        params = config.get("kinematics", {})
        self.speed_window = params.get("speed_window", 15)
        self.accel_window = params.get("accel_window", 30)
        self.border_margin = params.get("border_margin", 20)
        self.min_tracking_frames = params.get("min_tracking_frames", 10)
        self.max_physical_accel = params.get("max_physical_accel", 6.0)
        # 恢复使用 poly_order 参数
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
            # 计算检测框底边中心点 (Bottom-Center) 的 Y 坐标
            # 用于判断车辆在画面中的位置 (远/近)
            raw_bottom_y = raw_boxes[:, 3] 
            
            # 同时保留中心点用于静态检测
            raw_centers_x = (raw_boxes[:, 0] + raw_boxes[:, 2]) / 2
            raw_centers_y = (raw_boxes[:, 1] + raw_boxes[:, 3]) / 2
        else:
            raw_bottom_y = []
            raw_centers_x, raw_centers_y = [], []
        
        for i, (tid, point) in enumerate(zip(detections.tracker_id, points_transformed)):
            if tid not in self.trackers:
                self.trackers[tid] = {
                    'kf': KalmanFilterCV(point, dt=self.dt),
                    # 使用新的自适应 SG 滤波器
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
            
            # --- 静态抑制 ---
            curr_pixel = (raw_centers_x[i], raw_centers_y[i])
            prev_pixel = self.last_raw_pixels.get(tid, curr_pixel)
            pixel_dist = np.hypot(curr_pixel[0]-prev_pixel[0], curr_pixel[1]-prev_pixel[1])
            self.last_raw_pixels[tid] = curr_pixel
            is_moving = pixel_dist > 0.8 

            # --- Stage 1: KF 平滑 ---
            kf = self.trackers[tid]['kf']
            state = kf.update(point)
            smooth_pos_x, smooth_pos_y = state[0], state[1]
            
            # --- Stage 2: Adaptive SG ---
            # 计算归一化 Y 坐标 (0.0~1.0)
            # y 越大越靠近底部(近处)，y 越小越靠近顶部(远处)
            y_pixel = raw_bottom_y[i]
            y_norm = y_pixel / img_h
            
            sg = self.trackers[tid]['sg']
            # 传入 y_norm 供滤波器内部判断置信度
            speed, accel = sg.update(smooth_pos_x, smooth_pos_y, y_norm)

            # 强制静态清零
            if not is_moving and speed < 1.0:
                speed = 0.0
                accel = 0.0
                sg.last_accel = 0.0
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

            results[tid] = {
                "speed": speed, 
                "accel": accel,
                "curr_x": smooth_pos_x,
                "curr_y": smooth_pos_y
            }
            
        return results
