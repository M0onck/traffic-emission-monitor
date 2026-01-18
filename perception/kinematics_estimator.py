import numpy as np
from collections import deque
from perception.math.kalman_filter import KalmanFilterCV

class AdaptiveSGEstimator:
    """
    [数学组件] 透视自适应 SG 滤波器 (Adaptive Savitzky-Golay Estimator)
    
    功能：
    结合视觉观测的几何特征（如车辆在画面中的位置），动态融合"测量值"与"预测值"。
    在 v5.2 版本中，本组件已演变为"惯性增强型"滤波器，主要负责将 KF 输出的稳态速度
    与长窗口加速度进行融合，并根据视差风险动态调整权重。
    """
    def __init__(self, pos_window=31, speed_window=31, dt=0.033, poly_order=2):
        """
        :param pos_window: 位置历史窗口大小 (不再主要依赖，仅作 backup)
        :param speed_window: 加速度计算窗口大小 (用于计算长时斜率)
        :param dt: 采样时间间隔 (秒)
        :param poly_order: SG 拟合阶数
        """
        self.pos_window = pos_window
        self.speed_window = speed_window 
        self.dt = dt
        self.poly_order = poly_order
        
        # 1. 位置历史 (仅作为 KF 失效时的备用)
        self.x_history = deque(maxlen=pos_window)
        self.y_history = deque(maxlen=pos_window)
        self.t_axis_pos = np.arange(pos_window) * dt
        
        # 2. 速度历史 (用于计算宽窗加速度)
        self.speed_history = deque(maxlen=speed_window)
        
        # 3. 状态记忆 (上一帧状态，用于惯性预测)
        self.last_speed = 0.0
        self.last_accel = 0.0
        
        # 4. 滞后补偿队列
        self.lag_compensation = speed_window // 2
        self.speed_delay_queue = deque(maxlen=self.lag_compensation + 1)

    def _sg_derivative(self, y_data, t_axis):
        """
        [备用] Savitzky-Golay 多项式拟合求导
        仅在未传入 kf_speed 时启用，用于从位置历史反推速度。
        """
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
        [核心策略] 计算测量置信度 (Cap Confidence Strategy)
        
        根据车辆在画面中的垂直位置 (y_norm) 评估视差风险。
        策略更新：强制设定置信度上限 (MAX_CONF=0.3)，无论视觉观测多么稳定，
        绝大部分权重 (70%+) 都强制分配给惯性预测值，以实现"强匀速"效果。
        
        :param y_norm: 归一化垂直坐标 (0.0=ROI顶边, 1.0=ROI底边)
        """
        TOP_EDGE = 0.15     # 远端视差高风险区
        BOTTOM_EDGE = 0.85  # 近端透视拉伸高风险区
        MIN_CONF = 0.1      # 最小信任度
        MAX_CONF = 0.3      # [关键] 最大信任度被钳制在 0.3
        
        if TOP_EDGE <= y_norm <= BOTTOM_EDGE:
            # 中间区域 (最佳观测区) -> 返回上限值
            return MAX_CONF
        
        elif y_norm < TOP_EDGE:
            # 远端衰减区 (线性上升)
            ratio = max(0.0, y_norm) / TOP_EDGE
            return MIN_CONF + ratio * (MAX_CONF - MIN_CONF)
            
        else: # y_norm > BOTTOM_EDGE
            # 近端衰减区 (线性下降)
            dist_to_bottom = 1.0 - y_norm
            range_len = 1.0 - BOTTOM_EDGE
            ratio = max(0.0, dist_to_bottom) / range_len
            return MIN_CONF + ratio * (MAX_CONF - MIN_CONF)

    def update(self, pos_x, pos_y, y_norm, kf_speed=None):
        """
        执行单帧更新
        :param pos_x, pos_y: 平滑后的位置坐标
        :param y_norm: 归一化垂直坐标
        :param kf_speed: [新增] 从 Kalman Filter 状态向量直接提取的速度 (推荐)
        """
        # Stage 1: 数据录入
        self.x_history.append(pos_x)
        self.y_history.append(pos_y)
        if len(self.x_history) < self.pos_window: return 0.0, 0.0

        # Stage 2: 速度计算 (磁性融合策略)
        
        # A. 计算SG瞬时速度 (Raw Measurement) - 即使有噪声，但它是最真实的反应
        vx_sg = self._sg_derivative(list(self.x_history), self.t_axis_pos)
        vy_sg = self._sg_derivative(list(self.y_history), self.t_axis_pos)
        speed_raw = np.sqrt(vx_sg**2 + vy_sg**2)
        
        # B. 获取KF稳态速度 (Ideal Model)
        speed_model = kf_speed if kf_speed is not None else speed_raw
        
        # C. [核心修改] 洛伦兹加权融合
        # 计算测量值相对于模型的偏差
        deviation = abs(speed_raw - speed_model)
        
        # 定义"逃逸阈值" (sigma): 超过 1.5 m/s 的偏差被视为潜在的真实变速
        MAGNET_SIGMA = 1.5 
        
        # 计算模型权重: 偏差越小，权重越接近 1.0 (吸附)；偏差越大，权重趋向 0.0 (逃逸)
        # 这是一个软阈值，类似平方反比定律
        model_weight = 1.0 / (1.0 + (deviation / MAGNET_SIGMA) ** 2)
        
        # 融合速度：大部分时候是 model，急刹车时混入 raw
        speed_fusion = model_weight * speed_model + (1.0 - model_weight) * speed_raw
        
        self.speed_history.append(speed_fusion)
        
        # Stage 3: 加速度计算 (宽窗线性斜率)
        accel_meas = 0.0
        min_accel_window = 10
        if len(self.speed_history) >= min_accel_window:
            v_now = self.speed_history[-1]
            v_old = self.speed_history[0]
            dt_span = (len(self.speed_history) - 1) * self.dt
            accel_meas = (v_now - v_old) / dt_span

        # Stage 4: 惯性预测
        speed_pred = self.last_speed + self.last_accel * self.dt
        accel_pred = self.last_accel

        # Stage 5: 动态融合
        w = self._calculate_confidence(y_norm)
        
        # 如果偏差已经大到让模型失效 (model_weight < 0.5)，说明发生了真实变速
        # 此时应适当放开对惯性的依赖，让滤波器响应变化
        if model_weight < 0.5:
             w = min(w * 2.0, 0.8) # 提升观测权重

        final_speed = w * speed_fusion + (1 - w) * speed_pred
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
    """
    [感知层] 运动学估算器 (主控类)
    负责管理所有车辆的追踪状态，调度 KF 和 SG 滤波器。
    """
    def __init__(self, config: dict):
        self.fps = config.get("fps", 30)
        dt = 1.0 / self.fps
        
        params = config.get("kinematics", {})
        # [默认参数调优] 增大窗口以适应强平滑需求
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
        """
        更新所有车辆的运动学状态
        :param detections: Supervision Detections 对象
        :param points_transformed: 透视变换后的物理坐标 (N, 2)
        :param frame_shape: 图像尺寸 (h, w)
        :param roi_y_range: ROI 垂直范围 (min_y, max_y)，用于归一化计算
        """
        results = {}
        img_h, img_w = frame_shape[:2]
        
        # 解析 ROI 边界
        roi_min_y, roi_max_y = 0.0, float(img_h)
        if roi_y_range:
            roi_min_y, roi_max_y = roi_y_range
        roi_height = max(1.0, roi_max_y - roi_min_y)

        # 提取检测框信息
        raw_boxes = detections.xyxy
        if len(raw_boxes) > 0:
            raw_bottom_y = raw_boxes[:, 3] 
            raw_centers_x = (raw_boxes[:, 0] + raw_boxes[:, 2]) / 2
            raw_centers_y = (raw_boxes[:, 1] + raw_boxes[:, 3]) / 2
        else:
            raw_bottom_y = []
            raw_centers_x, raw_centers_y = [], []
        
        for i, (tid, point) in enumerate(zip(detections.tracker_id, points_transformed)):
            # 1. 初始化新车辆
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
            
            # 2. 静止检测 (基于像素移动距离)
            curr_pixel = (raw_centers_x[i], raw_centers_y[i])
            prev_pixel = self.last_raw_pixels.get(tid, curr_pixel)
            pixel_dist = np.hypot(curr_pixel[0]-prev_pixel[0], curr_pixel[1]-prev_pixel[1])
            self.last_raw_pixels[tid] = curr_pixel
            is_moving = pixel_dist > 0.8 

            # 3. 卡尔曼滤波 (Position Smoothing & Velocity Estimation)
            kf = self.trackers[tid]['kf']
            state = kf.update(point)
            smooth_pos_x, smooth_pos_y = state[0], state[1]
            
            # [核心逻辑] 直接提取 KF 状态向量中的速度分量
            # state = [pos_x, pos_y, vel_x, vel_y]
            # KF 已经在其内部通过 Q/R 矩阵平衡了匀速模型与观测值，
            # 这里的 vel_x/vel_y 是最符合"强匀速"假设的速度估计。
            kf_vel_x, kf_vel_y = state[2], state[3]
            kf_speed = np.sqrt(kf_vel_x**2 + kf_vel_y**2)
            
            # 4. 计算相对于 ROI 的归一化坐标
            y_pixel = raw_bottom_y[i]
            y_norm = (y_pixel - roi_min_y) / roi_height
            
            # 5. SG 滤波与动态融合
            # 将 KF 计算出的稳态速度 (kf_speed) 传递给 SG 滤波器
            sg = self.trackers[tid]['sg']
            speed, accel = sg.update(smooth_pos_x, smooth_pos_y, y_norm, kf_speed=kf_speed)

            # 6. 静止抑制
            if not is_moving and speed < 1.0:
                speed = 0.0
                accel = 0.0
                sg.last_accel = 0.0
                sg.speed_delay_queue.clear()

            # 7. 边缘安全区过滤
            box = raw_boxes[i]
            x1, y1, x2, y2 = box
            if (x1 < self.border_margin or y1 < self.border_margin or 
                x2 > img_w - self.border_margin or y2 > img_h - self.border_margin):
                continue 

            # 8. 最小追踪帧数过滤
            max_window = max(self.speed_window, self.accel_window)
            if self.active_frames[tid] < max(self.min_tracking_frames, max_window):
                continue 

            # 9. 物理限幅
            accel = np.clip(accel, -self.max_physical_accel, self.max_physical_accel)

            results[tid] = {
                "speed": speed, 
                "accel": accel,
                "curr_x": smooth_pos_x,
                "curr_y": smooth_pos_y
            }
            
        return results
