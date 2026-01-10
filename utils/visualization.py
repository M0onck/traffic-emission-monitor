import cv2
import numpy as np
import supervision as sv

class Visualizer:
    """
    [工具] 可视化渲染器。
    封装了 Supervision 的绘图功能，负责将检测结果绘制到视频帧上。
    """
    def __init__(self, calibration_points: np.ndarray, trace_length: int = 30):
        self.calibration_points = calibration_points.astype(np.int32)
        
        self.box_annotator = sv.BoxAnnotator(thickness=2, color_lookup=sv.ColorLookup.TRACK)
        self.label_annotator = sv.LabelAnnotator(
            text_scale=0.5, text_thickness=1, text_padding=5,
            text_position=sv.Position.BOTTOM_CENTER, color_lookup=sv.ColorLookup.TRACK
        )
        self.trace_annotator = sv.TraceAnnotator(
            thickness=2, trace_length=trace_length, position=sv.Position.BOTTOM_CENTER
        )

    def render(self, frame: np.ndarray, detections: sv.Detections, labels: list) -> np.ndarray:
        scene = frame.copy()
        
        # 1. 绘制标定区
        cv2.polylines(scene, [self.calibration_points], True, (255, 255, 0), 1)
        cv2.putText(scene, "Analysis Zone", tuple(self.calibration_points[3]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # 2. 绘制车辆信息
        scene = self.trace_annotator.annotate(scene=scene, detections=detections)
        scene = self.box_annotator.annotate(scene=scene, detections=detections)
        scene = self.label_annotator.annotate(scene=scene, detections=detections, labels=labels)
        return scene

def resize_with_pad(image: np.ndarray, target_size: tuple) -> np.ndarray:
    """Letterbox 自适应缩放，保持长宽比并填充黑边"""
    h, w = image.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    
    new_w, new_h = int(w * scale), int(h * scale)
    resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off, y_off = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized_img
    return canvas
