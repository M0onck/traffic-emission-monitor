import cv2
import numpy as np
from perception.plate_classifier import LightPlateTypeClassifier

class LicensePlateRecognizer:
    """
    [感知层] 轻量化车牌类型识别包装器
    功能：检测车辆图片中的车牌，并返回标准化的颜色类别（专为新能源/黄绿双拼优化）。
    注：已剥离耗时的 OCR 字符识别模块。
    """
    def __init__(self):
        # 实例化我们的轻量级分类器
        self.catcher = LightPlateTypeClassifier()
        
        # 基础类型码映射表 (对应 plate_classifier/core/typedef.py)
        self.color_map = {
            0: "Blue",    # 蓝牌
            1: "Yellow",  # 黄牌单层
            2: "White",   # 白牌
            3: "Green",   # 绿牌 (普通新能源)
            4: "Black",   # 黑牌
            9: "Yellow"   # 黄牌双层
        }

    def predict(self, vehicle_image: np.ndarray):
        """
        :param vehicle_image: 车辆 ROI 图像 (BGR)
        :return: (plate_code, color_string, confidence)
        """
        # 返回默认的空文本 "", 以兼容旧接口
        if vehicle_image is None or vehicle_image.size == 0:
            return "", "Unknown", 0.0

        results = self.catcher(vehicle_image)
        if not results:
            return "", "Unknown", 0.0

        # 获取置信度最高的结果 (通常一张截图中只有一个车牌)
        res = results[0]
        conf = res['confidence']
        type_idx = res['plate_type']
        
        # --- 核心优化逻辑 ---
        # 利用新模块底层的兜底逻辑，只要底层认定是新能源（包含黄绿双拼），
        # 我们就强制对外输出 "Green"，完美契合业务层 classifier.py 的期待。
        if res.get('is_new_energy', False):
            color_str = "Green"
        else:
            color_str = self.color_map.get(type_idx, "Unknown")
        
        return "", color_str, conf
