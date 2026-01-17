import cv2
import numpy as np

class CameraPreprocessor:
    """
    [感知层] 相机前处理器
    功能：利用内参矩阵和畸变系数，对输入的原始视频帧进行【去畸变 (Undistort)】处理。
    这能将广角镜头产生的"桶形畸变"拉直，确保后续的透视变换和测速是基于线性几何的。
    """
    def __init__(self, config: dict = None):
        # 1. 硬编码标定结果 (来自于您的标定输出)
        #    也可以改为从 config.json 读取，但为了方便，直接填入您刚测出的精确值
        self.mtx = np.array([
            [1.74715922e+03, 0.00000000e+00, 1.27452216e+03],
            [0.00000000e+00, 1.74910293e+03, 6.86205212e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ])
        
        self.dist = np.array([
            [ 0.00867745,  0.05071857, -0.02574871,  0.00441062, -0.11931698]
        ])
        
        # 缓存计算出的新内参和 ROI，避免每一帧重复计算
        self.new_camera_mtx = None
        self.roi = None
        self.img_shape = None # (width, height)

    def preprocess(self, frame):
        """
        对单帧图像进行去畸变处理
        """
        if frame is None: return frame
        
        h, w = frame.shape[:2]
        
        # 2. 初始化/更新新内参 (仅在第一帧或分辨率变化时计算)
        if self.new_camera_mtx is None or self.img_shape != (w, h):
            self.img_shape = (w, h)
            # alpha=0: 裁剪掉所有黑色无效像素 (画面会变小，但全是有效信息)
            # alpha=1: 保留所有像素 (画面边缘可能有黑色弯曲边框)
            # 建议 alpha=0，因为 YOLO 不需要看黑边
            alpha = 0 
            self.new_camera_mtx, self.roi = cv2.getOptimalNewCameraMatrix(
                self.mtx, self.dist, (w, h), alpha, (w, h)
            )

        # 3. 执行去畸变
        # 这一步会消耗少量算力 (1080P下约 5-10ms)，如果帧率吃紧，可考虑 GPU 加速 (cv2.cuda)
        dst = cv2.undistort(frame, self.mtx, self.dist, None, self.new_camera_mtx)
        
        # 4. 裁剪图像 (配合 alpha=0 使用)
        x, y, w, h = self.roi
        dst = dst[y:y+h, x:x+w]
        
        return dst
