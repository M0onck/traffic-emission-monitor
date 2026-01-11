import cv2
import hyperlpr3
import numpy as np

class LicensePlateRecognizer:
    """
    [感知] 车牌识别包装器 (Wrapper for HyperLPR3)
    功能：识别车辆图片中的车牌，并返回标准化的颜色类别。
    """
    def __init__(self):
        # 初始化 HyperLPR3，detect_level=1 精度优先
        self.catcher = hyperlpr3.LicensePlateCatcher(detect_level=1)
        
        # HyperLPR3 类型码映射表
        self.color_map = {
            0: "Blue",    # 蓝牌
            1: "Yellow",  # 黄牌
            2: "White",   # 白牌
            3: "Green",   # 绿牌 (涵盖小型新能源)
            4: "Black",   # 黑牌
            5: "Green",   # 部分版本将大型新能源归为此类
        }

    def predict(self, vehicle_image: np.ndarray):
        """
        :param vehicle_image: 车辆 ROI 图像 (BGR)
        :return: (plate_code, color_string, confidence)
        """
        if vehicle_image is None or vehicle_image.size == 0:
            return None, "Unknown", 0.0

        results = self.catcher(vehicle_image)
        if not results:
            return None, "Unknown", 0.0

        # 获取置信度最高的结果: [code, conf, type_idx, box]
        code, conf, type_idx, _ = results[0]
        
        color_str = self.color_map.get(type_idx, "Unknown")
        
        return code, color_str, conf
