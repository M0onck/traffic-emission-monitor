import cv2
import numpy as np

class ViewTransformer:
    """
    [感知层] 视图变换模块。
    功能：利用单应性矩阵 (Homography)，将图像坐标转换为物理世界坐标。提供 ROI (感兴趣区域) 判定功能。
    """
    def __init__(self, source: np.ndarray, target: np.ndarray):
        """
        :param source: 图像上的4个源点 (Pixel)
        :param target: 物理世界对应的4个目标点 (Meters)
        """
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m, _ = cv2.findHomography(source, target)
        
        # 保存 ROI 轮廓 (转换为 int32 以适配 cv2.pointPolygonTest)
        # 输入 source 的形状通常是 (4, 2)
        self.roi_contour = source.astype(np.int32)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        批量转换坐标点
        :param points: (N, 2) 像素坐标
        :return: (N, 2) 物理坐标
        """
        if points is None or points.size == 0:
            return points
            
        # Reshape 为 (N, 1, 2) 以适配 cv2.perspectiveTransform
        reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(reshaped, self.m)
        
        return transformed.reshape(-1, 2)

    def is_in_roi(self, point: np.ndarray) -> bool:
        """
        判断像素点是否在标定区域内
        :param point: [x, y] 像素坐标
        :return: True if inside or on edge
        """
        return cv2.pointPolygonTest(self.roi_contour, tuple(point), False) >= 0

    def get_roi_vertical_bounds(self):
        """
        [修复版] 获取 ROI 的垂直边界 (用于归一化坐标)
        兼容 (N, 2) 和 (N, 1, 2) 两种数据形状。
        :return: (min_y, max_y)
        """
        if self.roi_contour is None or len(self.roi_contour) == 0:
            return 0.0, 1080.0 # 兜底默认值
        
        # 自动判断维度以进行正确切片
        # 情况 A: Shape is (N, 1, 2) -> 常见于 cv2.findContours 输出
        if self.roi_contour.ndim == 3:
            ys = self.roi_contour[:, 0, 1]
        # 情况 B: Shape is (N, 2) -> 常见于手动定义的点集 (本项目情况)
        else:
            ys = self.roi_contour[:, 1]
            
        return float(np.min(ys)), float(np.max(ys))
