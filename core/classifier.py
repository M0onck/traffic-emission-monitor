import config.settings as cfg
from collections import defaultdict

class VehicleClassifier:
    @staticmethod
    def resolve_type(class_id, plate_history=None, plate_color_override=None):
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
            # 加权投票逻辑
            scores = defaultdict(float)
            for e in plate_history: scores[e['color']] += e['area']
            if scores: final_color = max(scores, key=scores.get)
            
        # 2. 查表匹配 (优先)
        key = (class_id, final_color)
        if key in cfg.TYPE_MAP:
            return final_color, cfg.TYPE_MAP[key]
            
        # 3. 兜底逻辑 (MLE策略)
        suffix = "(Default)" # 可以根据是否开启OCR传入不同后缀，这里简化处理
        
        if class_id == cfg.YOLO_CLASS_BUS:   # Bus
            return final_color, f"HDV-electric {suffix}"
        elif class_id == cfg.YOLO_CLASS_TRUCK: # Truck
            return final_color, f"HDV-diesel {suffix}"
        elif class_id == cfg.YOLO_CLASS_CAR:   # Car
            return final_color, f"LDV-gasoline {suffix}"
            
        return final_color, cfg.TYPE_MAP.get('Default_Heavy', 'HDV-diesel')
