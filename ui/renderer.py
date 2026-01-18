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
        [修改版] 绘制单帧画面
        特性：
        1. 标签内容简化：直接构建包含 OpMode 的文本，不再依赖外部 formatter。
        2. 视觉优化：隐藏实时速度，仅显示工况代码和状态 (如 Op:33 [Accel])。
        """
        scene = frame.copy()
        
        # 1. 构造自定义标签文本 (替代原有的 formatter.format)
        labels = []
        for data in label_data_list:
            # --- 第一行：ID 和 车型 ---
            # 如果 display_type 为空，显示默认值 "Vehicle"
            type_str = data.display_type if data.display_type else "Vehicle"
            line1 = f"#{data.track_id} {type_str}"
            
            # --- 第二行：优先显示 OpMode ---
            line2 = ""
            if data.emission_info and 'op_mode' in data.emission_info:
                op = data.emission_info['op_mode']
                
                # 简单的状态映射表，让数字代码更具可读性
                state_str = "Run"
                if op == 0: state_str = "Brake"      # 刹车/怠速
                elif op == 1: state_str = "Idle"     # 怠速
                elif op in [11, 21]: state_str = "Cruise" # 巡航
                elif op >= 33: state_str = "Accel"   # 加速 (33,35,37...)
                
                line2 = f"Op:{op} [{state_str}]"
                
            elif data.speed is not None:
                # 备用逻辑：如果没有算出工况但有速度（极少情况），显示速度
                line2 = f"{data.speed:.1f} km/h"
            else:
                # 初始状态
                line2 = "Detecting..."

            # 合并两行文本
            labels.append(f"{line1}\n{line2}")

        # 2. 绘制基础图层 (ROI 区域)
        # 保持原有的标定框绘制逻辑
        if self.calibration_points is not None and len(self.calibration_points) > 0:
            cv2.polylines(scene, [self.calibration_points], True, (255, 255, 0), 1)
            
            # 确保索引不越界，原代码使用第4个点 (index 3) 作为文字锚点
            text_anchor = tuple(self.calibration_points[3]) if len(self.calibration_points) > 3 else (50, 50)
            cv2.putText(scene, "Analysis Zone", text_anchor, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # 3. 绘制车辆注记 (利用 Supervision 库)
        if self.trace_annotator:
            scene = self.trace_annotator.annotate(scene=scene, detections=detections)
        
        if self.box_annotator:
            scene = self.box_annotator.annotate(scene=scene, detections=detections)
            
        if self.label_annotator:
            # 传入刚才构造好的新 labels 列表
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
