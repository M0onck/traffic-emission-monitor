import numpy as np

class KalmanFilterCV:
    """
    [感知层] 基于常速度模型 (Constant Velocity Model) 的卡尔曼滤波器。
    用于平滑视觉检测带来的位置抖动，并估算速度。
    
    状态向量 State x: [pos_x, pos_y, vel_x, vel_y].T
    观测向量 Measurement z: [pos_x, pos_y].T
    """
    def __init__(self, init_pos: np.ndarray, dt: float = 1/30.0, 
                 process_noise_scale: float = 1e-4, measurement_noise_scale: float = 1.0):
        """
        :param init_pos: 初始位置 [x, y]
        :param dt: 时间步长 (1/FPS)
        :param process_noise_scale: 过程噪声 Q (代表物体真实运动的突变程度)
        :param measurement_noise_scale: 观测噪声 R (代表检测器的抖动程度)
        """
        self.dt = dt
        
        # 1. 初始化状态 [x, y, 0, 0]
        self.x = np.array([init_pos[0], init_pos[1], 0, 0], dtype=np.float32)
        
        # 2. 状态转移矩阵 F (x' = x + v*dt)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # 3. 观测矩阵 H (仅观测位置)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # 4. 协方差矩阵 P (初始不确定性)
        self.P = np.eye(4, dtype=np.float32) * 100.0
        
        # 5. 噪声矩阵 Q, R
        self.Q = np.eye(4, dtype=np.float32) * process_noise_scale
        self.R = np.eye(2, dtype=np.float32) * measurement_noise_scale

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        执行 预测(Predict) + 更新(Correct) 步骤
        :param measurement: 当前帧观测坐标 [x, y]
        :return: 更新后的状态向量
        """
        z = np.array(measurement, dtype=np.float32)

        # Predict
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Update
        y = z - self.H @ x_pred                  # 残差
        S = self.H @ P_pred @ self.H.T + self.R  # 创新协方差
        K = P_pred @ self.H.T @ np.linalg.inv(S) # 卡尔曼增益
        
        self.x = x_pred + K @ y
        I = np.eye(self.F.shape[0])
        self.P = (I - K @ self.H) @ P_pred
        
        return self.x
