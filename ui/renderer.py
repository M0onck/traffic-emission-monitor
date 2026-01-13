import cv2
import numpy as np
import supervision as sv
from dataclasses import dataclass

"""
[表现层] 视频渲染器
功能：负责将检测结果、轨迹、车辆信息及统计数据绘制到视频帧上。
职责：
1. LabelFormatter: 负责将业务数据 (Data Objects) 格式化为人类可读的字符串。
2. Visualizer: 负责调用 OpenCV/Supervision 进行实际的图形绘制（框、标签、轨迹）。
依赖：仅依赖数据对象，不包含业务计算逻辑。
"""

@dataclass
class LabelData:
    """传输给显示层的数据对象"""
    track_id: int
    class_id: int
    speed: float = None
    emission_info: dict = None
    display_type: str = None

class LabelFormatter:
    """
    标签格式化器
    负责将业务数据转换为屏幕显示的字符串
    """
    def __init__(self, show_emission: bool = True):
        self.show_emission = show_emission

    def format(self, data: LabelData) -> str:
        label = f"#{data.track_id}"
        
        # 1. 车型显示
        if data.display_type:
            label += f" {data.display_type}"
            
        # 2. 速度与状态显示
        if data.speed is not None:
            label += f" | {data.speed:.1f}m/s"
            
        # 3. 排放状态显示
        if self.show_emission and data.emission_info:
            op_mode = data.emission_info.get('op_mode')
            if op_mode == 0:
                label += " [BRAKE]"
            elif op_mode == 1:
                label += " [IDLE]"
            # op_mode > 1 (GO) 保持简洁不显示
            
        return label

class Visualizer:
    """
    核心渲染器
    """
    def __init__(self, calibration_points: np.ndarray, trace_length: int = 30):
        self.calibration_points = calibration_points.astype(np.int32)
        # 注入 LabelFormatter
        self.formatter = LabelFormatter()
        
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(
            text_scale=0.5, text_thickness=1, text_padding=5,
            text_position=sv.Position.BOTTOM_CENTER
        )
        self.trace_annotator = sv.TraceAnnotator(
            thickness=2, trace_length=trace_length, position=sv.Position.BOTTOM_CENTER
        )

    def render(self, frame: np.ndarray, detections: sv.Detections, label_data_list: list) -> np.ndarray:
        """
        绘制单帧画面
        :param frame: 原始视频帧
        :param detections: 目标检测结果 (Supervision Detections)
        :param label_data_list: 对应每个检测目标的标签数据列表
        :return: 绘制完成的图像
        """
        scene = frame.copy()
        
        # 1. 转换数据对象为字符串
        labels = [self.formatter.format(d) for d in label_data_list]

        # 2. 绘制基础图层
        cv2.polylines(scene, [self.calibration_points], True, (255, 255, 0), 1)
        cv2.putText(scene, "Analysis Zone", tuple(self.calibration_points[3]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # 3. 绘制车辆
        scene = self.trace_annotator.annotate(scene=scene, detections=detections)
        scene = self.box_annotator.annotate(scene=scene, detections=detections)
        scene = self.label_annotator.annotate(scene=scene, detections=detections, labels=labels)
        
        return scene

def resize_with_pad(image: np.ndarray, target_size: tuple) -> np.ndarray:
    """
    工具函数：保持纵横比缩放并填充黑边
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off, y_off = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized_img
    return canvas
