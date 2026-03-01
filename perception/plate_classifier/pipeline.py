import numpy as np
import cv2
from .core.typedef import *
from .core.tools_process import get_rotate_crop_image

class PlateTypeClassifierPipeline:
    def __init__(self, detector, classifier):
        # 移除了 recognizer
        self.detector = detector
        self.classifier = classifier

    def run(self, image: np.ndarray) -> list:
        result = list()
        if len(image.shape) != 3 or image is None:
            return result
            
        # 1. 执行目标检测，寻找车牌
        outputs = self.detector(image)
        for out in outputs:
            rect = out[:4].astype(int)
            score = out[4]
            land_marks = out[5:13].reshape(4, 2).astype(int)
            layer_num = int(out[13])
            
            # 2. 根据关键点裁剪并矫正车牌图像
            pad = get_rotate_crop_image(image, land_marks)
            
            # 3. 直接跳过 OCR，进入分类器
            cls_result = self.classifier(pad)
            
            # 4. 优化后的分类逻辑（降序寻找合法类型 + 兜底）
            plate_type = UNKNOWN
            sorted_indices = np.argsort(cls_result.flatten())[::-1]
            
            for idx in sorted_indices:
                idx = int(idx)
                if idx == PLATE_TYPE_YELLOW:
                    plate_type = YELLOW_DOUBLE if layer_num == DOUBLE else YELLOW_SINGLE
                    break
                elif idx == PLATE_TYPE_BLUE:
                    plate_type = BLUE
                    break
                elif idx == PLATE_TYPE_GREEN:
                    plate_type = GREEN
                    break
                    
            # 针对黄绿牌/新能源的绝对兜底逻辑
            if plate_type == UNKNOWN:
                plate_type = GREEN
                
            # 保存结果 (只包含边界框、检测置信度、车牌类型)
            result.append({
                "bbox": rect.tolist(),
                "confidence": float(score),
                "plate_type": plate_type,
                "is_new_energy": plate_type == GREEN # 提供一个直观的布尔值
            })

        return result
