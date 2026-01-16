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
        # pointPolygonTest 返回值: +1(内部), -1(外部), 0(边缘)
        # measureDist=False 表示只检测位置关系，不计算距离
        return cv2.pointPolygonTest(self.roi_contour, tuple(point), False) >= 0
