import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict

# 本地模块导入
import config.settings as cfg
from core.geometry import ViewTransformer
from core.vehicle_registry import VehicleRegistry
from core.database import DatabaseManager
from perception.kinematics import KinematicsEstimator
from perception.plate_reader import LicensePlateRecognizer
from analysis.brake_model import BrakeEmissionModel
from utils.visualization import Visualizer, resize_with_pad
from utils.calibration_ui import CalibrationUI
from utils.reporter import Reporter

def main():
    # =========================================================================
    # Phase 0: 模式判定与初始化 (Mode Detection & Init)
    # =========================================================================
    print(f"\n>>> [系统启动] 正在检查工作模式...")
    
    motion_on = cfg.ENABLE_MOTION
    ocr_on = cfg.ENABLE_OCR
    emission_req = cfg.ENABLE_EMISSION
    
    # 自动纠正：如果没有运动数据，绝对无法计算排放
    if emission_req and not motion_on:
        print("    [警告] 排放计算依赖运动感知，但 Motion 已关闭。排放功能将强制禁用！")
        emission_req = False

    # 判定四大模式
    mode_id = 0
    mode_desc = ""
    
    if not motion_on and not ocr_on:
        mode_id = 1
        mode_desc = "基础流量统计 (仅分车型)"
    elif not motion_on and ocr_on:
        mode_id = 2
        mode_desc = "流量+能源统计 (车型+能源类型)"
    elif motion_on and not ocr_on:
        mode_id = 3
        mode_desc = "流量+运动统计 (车型+运动属性)"
    elif motion_on and ocr_on:
        mode_id = 4
        mode_desc = "全功能模式 (车型+运动+能源+排放)"

    print(f">>> 当前工作模式: [Mode {mode_id}] {mode_desc}")
    print(f"    - 运动感知: {'[ON]' if motion_on else '[OFF]'}")
    print(f"    - 车牌识别: {'[ON]' if ocr_on else '[OFF]'}")
    print(f"    - 排放计算: {'[ON]' if emission_req else '[OFF]'}")

    # Mode 3 特殊提示
    if mode_id == 3 and emission_req:
        print("\n    [提示] 处于无 OCR 隐私模式。排放计算将使用默认能源类型假设：")
        print("          * 小型车 (Car)   -> 默认 Gasoline")
        print("          * 大型车 (Bus/Truck) -> 默认 Diesel")

    # 1. 交互式标定
    calibrator = CalibrationUI(cfg.VIDEO_PATH)
    source_points, target_points = calibrator.run()
    
    # 2. 核心算法模块初始化
    model = YOLO("model/yolov8n.pt") 
    view_transformer = ViewTransformer(source_points, target_points)
    
    # 按需加载模块
    kinematics_estimator = KinematicsEstimator(cfg.FPS, cfg.SPEED_WINDOW, cfg.ACCEL_WINDOW) if motion_on else None
    plate_recognizer = LicensePlateRecognizer() if ocr_on else None
    brake_model = BrakeEmissionModel() if emission_req else None
    
    visualizer = Visualizer(calibration_points=source_points, trace_length=cfg.FPS)
    byte_tracker = sv.ByteTrack(frame_rate=cfg.FPS)
    smoother = sv.DetectionsSmoother(length=3)

    plate_cache = {}    
    plate_retry = {}    
    
    registry = VehicleRegistry()
    db_manager = DatabaseManager(cfg.DB_PATH)
    print(f">>> 数据库已连接: {cfg.DB_PATH}\n")
    
    # =========================================================================
    # Phase 1: 视频流处理循环
    # =========================================================================
    video_info = sv.VideoInfo.from_video_path(cfg.VIDEO_PATH)
    WINDOW_NAME = "Traffic Emission Monitor"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    frame_id = 0
    with sv.VideoSink(cfg.TARGET_VIDEO_PATH, video_info=video_info) as sink:
        for frame in sv.get_video_frames_generator(cfg.VIDEO_PATH):
            frame_id += 1
            img_h, img_w = frame.shape[:2]
            
            # --- Sub-Phase 1.1: 基础感知 ---
            result = model(frame, conf=0.3, iou=0.5, agnostic_nms=True, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = detections[np.isin(detections.class_id, [2, 5, 7])] 
            detections = byte_tracker.update_with_detections(detections)
            detections = smoother.update_with_detections(detections)

            # --- Sub-Phase 1.2: 注册表更新 ---
            registry.update(detections, frame_id, model)

            # --- Sub-Phase 1.3: 离场检测 ---
            for tid, record in registry.check_exits(frame_id):
                if cfg.DEBUG_MODE:
                    Reporter.print_exit_report(tid, record, kinematics_estimator)
                
                # 传入当前模式配置，辅助生成最终状态
                final_plate, final_type = _resolve_final_status(record, ocr_on)
                db_manager.insert_macro(tid, record, final_type, final_plate)

            # --- Sub-Phase 1.4: 车牌识别 (OCR) ---
            if ocr_on and (frame_id % cfg.OCR_INTERVAL == 0):
                for tid, xyxy in zip(detections.tracker_id, detections.xyxy):
                    if frame_id - plate_retry.get(tid, -999) < cfg.OCR_RETRY_COOLDOWN: continue
                    
                    cx, cy = (xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2
                    x1, y1, x2, y2 = map(int, xyxy)
                    bbox_area = (x2 - x1) * (y2 - y1)

                    if (0.1 * img_w < cx < 0.9 * img_w) and \
                       (0.4 * img_h < cy < 0.98 * img_h) and \
                       (bbox_area > cfg.MIN_PLATE_AREA):
                        
                        crop = frame[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)]
                        code, color, conf = plate_recognizer.predict(crop)
                        plate_retry[tid] = frame_id

                        if color != "Unknown":
                            registry.add_plate_history(tid, color, bbox_area, conf)
                        
                        if color != "Unknown" and conf > cfg.OCR_CONF_THRESHOLD:
                            plate_cache[tid] = color
                            if tid in plate_retry: del plate_retry[tid]

            # ------------------------------------------------------
            # Phase 2: 物理估算 (Motion)
            # ------------------------------------------------------
            kinematics_data = {}
            if motion_on:
                points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
                points_transformed = view_transformer.transform_points(points)
                kinematics_data = kinematics_estimator.update(detections, points_transformed, frame.shape)
            
            # ------------------------------------------------------
            # Phase 3: 排放分析 (Emission)
            # ------------------------------------------------------
            emission_data = {}
            # 只有在 (需要排放计算) AND (有运动数据) 时才执行
            if emission_req and motion_on:
                # 注意：如果 ocr_on 为 False，plate_cache 将为空，
                # BrakeModel 内部会自动 fallback 到 Default_Car/Default_Heavy
                emission_data = brake_model.process(kinematics_data, detections, plate_cache)
                
                for tid, d in emission_data.items():
                    registry.update_emission_stats(tid, d['op_mode'], d['emission_rate'])
                    record = registry.get_record(tid)
                    if record and (frame_id - record['first_frame'] > 5):
                        db_manager.insert_micro(frame_id, tid, d)
            
            # ------------------------------------------------------
            # Phase 4: 标签生成 (Label Generation)
            # ------------------------------------------------------
            labels = []
            # [修正] 同时遍历 tid 和 class_id，以便在缺少高级数据时回退显示 YOLO 类别
            for tid, class_id in zip(detections.tracker_id, detections.class_id):
                label_text = f"#{tid}"
                
                # 情况 A: 排放数据可用 (显示最详细的推断类型)
                if tid in emission_data:
                    d = emission_data[tid]
                    display_type = d['type_str'] # e.g., "HDV-diesel", "LDV-gasoline"
                    
                    # [修复] 如果 OCR 关闭，仅追加 (Def) 标记，保留 LDV/HDV 前缀
                    if not ocr_on: 
                        # 避免重复添加后缀 (BrakeModel 有时会返回带 Default 的字串)
                        if "(Def" not in display_type and "(Fall" not in display_type:
                             display_type += "(Def)"
                    # 如果 OCR 开启但识别失败，尝试用历史推断优化显示
                    elif tid not in plate_cache:
                        hist = registry.get_history(tid)
                        if hist:
                            scores = defaultdict(float)
                            for h in hist: scores[h['color']] += h['area']
                            best = max(scores, key=scores.get)
                            display_type = f"{best}?" # e.g. "Blue?"

                    # 状态标签
                    state_tag = "[BRAKE]" if d['op_mode'] == 0 else \
                                "[IDLE]" if d['op_mode'] == 1 else "[GO]"
                    
                    label_text += f" {display_type} | {d['speed']:.1f}m/s {state_tag}"
                
                # 情况 B: 仅有运动数据 (显示 YOLO 类别 + 速度)
                elif tid in kinematics_data:
                    yolo_class = model.names[class_id]
                    spd = kinematics_data[tid]['speed']
                    label_text += f" {yolo_class} | {spd:.1f}m/s"
                
                # 情况 C: 仅有 YOLO 检测 (显示 YOLO 类别)
                else:
                    yolo_class = model.names[class_id]
                    # 如果 OCR 开启且有缓存，补充显示车牌颜色
                    if ocr_on and tid in plate_cache:
                        label_text += f" {yolo_class} ({plate_cache[tid]})"
                    else:
                        label_text += f" {yolo_class}"
                
                labels.append(label_text)
            
            # ------------------------------------------------------
            # Phase 5: 渲染
            # ------------------------------------------------------
            annotated_frame = visualizer.render(frame, detections, labels)
            sink.write_frame(annotated_frame)
            
            display_frame = resize_with_pad(annotated_frame, (1280, 720))
            cv2.imshow(WINDOW_NAME, display_frame)
            if cv2.waitKey(1) == ord("q"): break

    cv2.destroyAllWindows()
    
    # Phase 6: 收尾
    print("\n[处理结束] 正在处理剩余在场车辆...")
    for tid, record in registry.check_exits(frame_id + 1000):
        if cfg.DEBUG_MODE:
            Reporter.print_exit_report(tid, record, kinematics_estimator if motion_on else None)
        final_plate, final_type = _resolve_final_status(record, ocr_on)
        db_manager.insert_macro(tid, record, final_type, final_plate)

    db_manager.close()

# =========================================================================
# Helper: 解析最终车辆状态
# =========================================================================
def _resolve_final_status(record, ocr_enabled):
    """
    根据历史记录解析最终的车牌颜色和车辆类型
    增加 ocr_enabled 标志，用于处理 Mode 3 的默认值逻辑
    """
    if not ocr_enabled:
        # Mode 1 & 3: 无 OCR，直接返回默认值
        final_plate = "N/A (OCR Off)"
        if record.get('class_id') in [5, 7]: # Bus/Truck
            final_type = "HDV-diesel (Default)"
        else: # Car
            final_type = "LDV-gasoline (Default)"
        return final_plate, final_type

    # Mode 2 & 4: 正常 OCR 逻辑
    history = record.get('plate_history', [])
    final_plate = "Unknown"
    
    if history:
        scores = defaultdict(float)
        for e in history: 
            scores[e['color']] += e['area']
        if scores: 
            final_plate = max(scores, key=scores.get)

    final_type = "Calculating..."
    if final_plate != "Unknown":
        if final_plate == "Green": final_type = "LDV-electric"
        elif final_plate == "Yellow": final_type = "HDV-diesel"
        elif final_plate == "Blue": final_type = "LDV-gasoline"
        
        if record.get('class_id') in [5, 7] and final_plate == "Green":
            final_type = "HDV-electric (Large EV)"
    else:
        # OCR 开启但未识别到车牌的兜底
        if record.get('class_id') in [5, 7]:
            final_type = "HDV-diesel (Fallback)" # 同样对齐到 Diesel
        else:
            final_type = "LDV-gasoline (Fallback)"
            
    return final_plate, final_type

if __name__ == "__main__":
    main()
