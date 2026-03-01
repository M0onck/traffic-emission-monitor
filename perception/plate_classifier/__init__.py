import os
from .core.multitask_detect import MultiTaskDetectorORT
from .core.classification import ClassificationORT
from .pipeline import PlateTypeClassifierPipeline

# 获取当前文件所在目录，从而定位 models 文件夹
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, "models")

class LightPlateTypeClassifier:
    def __init__(self):
        # 初始化 ONNX Runtime 推理引擎
        det_path = os.path.join(MODEL_DIR, "y5fu_320x_sim.onnx")
        cls_path = os.path.join(MODEL_DIR, "litemodel_cls_96x_r1.onnx")
        
        det = MultiTaskDetectorORT(det_path, input_size=(320, 320))
        cls = ClassificationORT(cls_path, input_size=(96, 96))
        
        self.pipeline = PlateTypeClassifierPipeline(detector=det, classifier=cls)

    def __call__(self, image):
        # 传入 cv2.imread 读取的图像
        return self.pipeline.run(image)
