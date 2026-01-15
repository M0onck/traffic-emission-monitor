import math
from collections import defaultdict

class VehicleClassifier:
    """
    [业务层] 车辆类型分类器
    职责：根据 YOLO 类别、车牌历史和颜色，判定最终的业务车型 (如 'Bus-electric')。
    """
    def __init__(self, type_map: dict, yolo_classes: dict):
        """
        :param type_map: 映射表 {(class_id, color_str): type_str}
        :param yolo_classes: 类别ID配置 {'car': 2, 'bus': 5, 'truck': 7}
        """
        self.type_map = type_map
        self.cls_car = yolo_classes.get('car', 2)
        self.cls_bus = yolo_classes.get('bus', 5)
        self.cls_truck = yolo_classes.get('truck', 7)

    def resolve_type(self, class_id, plate_history=None, plate_color_override=None):
        """
        统一的车辆类型判定逻辑
        :param class_id: YOLO class ID (int)
        :param plate_history: 历史识别记录列表 (list of dict)
        :param plate_color_override: 强制指定的颜色 (str, optional)
        :return: (final_color, final_type_string)
        """
        # 1. 确定颜色
        final_color = "Unknown"
        if plate_color_override and plate_color_override != "Unknown":
            final_color = plate_color_override
        elif plate_history:
            # 加权投票逻辑： Conf * sqrt(Area)
            scores = defaultdict(float)
            for e in plate_history:
                # 兼容旧代码，如果没有 conf 字段则默认为 1.0
                conf = e.get('conf', 1.0)
                area = e.get('area', 0.0)
                
                weight = conf * math.sqrt(area) # [核心修改]
                scores[e['color']] += weight
                
            if scores: 
                final_color = max(scores, key=scores.get)
            
        # 2. 查表匹配 (优先)
        key = (class_id, final_color)
        if key in self.type_map:
            return final_color, self.type_map[key]
            
        # 3. 兜底逻辑 (MLE策略)
        suffix = "(Default)" # 可以根据是否开启OCR传入不同后缀，这里简化处理
        
        if class_id == self.cls_bus:   # Bus
            return final_color, f"Bus-electric {suffix}"
        elif class_id == self.cls_truck: # Truck
            return final_color, f"Truck-diesel {suffix}"
        elif class_id == self.cls_car:   # Car
            return final_color, f"Car-gasoline {suffix}"
            
        return final_color, self.type_map.get('Default_Heavy', 'HDV-diesel')
