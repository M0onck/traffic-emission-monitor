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
    # Phase 0: 初始化与标定 (Initialization)
    # =========================================================================
    # 1. 交互式标定
    calibrator = CalibrationUI(cfg.VIDEO_PATH)
    source_points, target_points = calibrator.run()
    
    # 2. 核心算法模块初始化
    model = YOLO("model/yolov8n.pt") 
    view_transformer = ViewTransformer(source_points, target_points)
    kinematics_estimator = KinematicsEstimator(cfg.FPS, cfg.SPEED_WINDOW, cfg.ACCEL_WINDOW)
    brake_model = BrakeEmissionModel()
    plate_recognizer = LicensePlateRecognizer()
    
    # 3. 可视化与追踪模块
    visualizer = Visualizer(calibration_points=source_points, trace_length=cfg.FPS)
    byte_tracker = sv.ByteTrack(frame_rate=cfg.FPS)
    smoother = sv.DetectionsSmoother(length=3)

    # 4. 状态管理初始化
    # plate_cache: 仅用于 UI 实时显示的简易缓存
    plate_cache = {}    
    # plate_retry: 控制 OCR 调用频率的冷却字典
    plate_retry = {}    
    RETRY_COOLDOWN = 5 
    
    # [核心] 车辆注册表 (接管所有数据管理)
    registry = VehicleRegistry()
    
    # [核心] 数据库管理器 (自动处理建表与连接)
    db_manager = DatabaseManager(cfg.DB_PATH)
    print(f">>> 数据库已连接: {cfg.DB_PATH}")
    
    # =========================================================================
    # Phase 1: 视频流处理循环 (Processing Loop)
    # =========================================================================
    video_info = sv.VideoInfo.from_video_path(cfg.VIDEO_PATH)
    print(f"\n>>> 系统启动 | 视频源: {cfg.VIDEO_PATH}")
    print(f">>> 调试模式: {'[开启]' if cfg.DEBUG_MODE else '[关闭]'} (包含 OpMode 统计与排放计算)\n")
    
    WINDOW_NAME = "Traffic Emission Monitor"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    frame_id = 0
    with sv.VideoSink(cfg.TARGET_VIDEO_PATH, video_info=video_info) as sink:
        for frame in sv.get_video_frames_generator(cfg.VIDEO_PATH):
            frame_id += 1
            img_h, img_w = frame.shape[:2]
            
            # ------------------------------------------------------
            # Sub-Phase 1.1: 基础感知 (YOLO + Tracking)
            # ------------------------------------------------------
            result = model(frame, conf=0.3, iou=0.5, agnostic_nms=True, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = detections[np.isin(detections.class_id, [2, 5, 7])] # 2:Car, 5:Bus, 7:Truck
            detections = byte_tracker.update_with_detections(detections)
            detections = smoother.update_with_detections(detections)

            # ------------------------------------------------------
            # Sub-Phase 1.2: 注册表更新 (Registry Update)
            # ------------------------------------------------------
            # 将最新的检测结果同步到注册表，更新车辆存活状态
            registry.update(detections, frame_id, model)

            # ------------------------------------------------------
            # Sub-Phase 1.3: 离场检测 (Exit Check)
            # ------------------------------------------------------
            # 检查哪些车辆已经超时离开画面
            for tid, record in registry.check_exits(frame_id):
                # 1. 打印控制台报告
                Reporter.print_exit_report(tid, record, kinematics_estimator)
                
                # 2. [新增] 写入数据库宏观表 (Macro Table)
                final_plate, final_type = _resolve_final_status(record)
                db_manager.insert_macro(tid, record, final_type, final_plate)

            # ------------------------------------------------------
            # Sub-Phase 1.4: 车牌识别 (OCR)
            # ------------------------------------------------------
            if frame_id % 5 == 0:
                for tid, xyxy in zip(detections.tracker_id, detections.xyxy):
                    # 冷却检查
                    if frame_id - plate_retry.get(tid, -999) < RETRY_COOLDOWN: continue
                    
                    # 几何参数计算
                    cx, cy = (xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2
                    x1, y1, x2, y2 = map(int, xyxy)
                    bbox_area = (x2 - x1) * (y2 - y1)

                    # 门控: 必须在画面中心区域且面积足够大
                    if (0.1 * img_w < cx < 0.9 * img_w) and \
                       (0.4 * img_h < cy < 0.98 * img_h) and \
                       (bbox_area > cfg.MIN_PLATE_AREA):
                        
                        crop = frame[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)]
                        code, color, conf = plate_recognizer.predict(crop)
                        plate_retry[tid] = frame_id

                        # 将有效识别结果存入注册表历史 (用于加权投票)
                        if color != "Unknown":
                            registry.add_plate_history(tid, color, bbox_area, conf)

                        # 实时缓存仅用于视频标签显示
                        if color != "Unknown" and conf > 0.3:
                            plate_cache[tid] = color
                            if tid in plate_retry: del plate_retry[tid]

            # ------------------------------------------------------
            # Phase 2: 物理估算 (Kinematics)
            # ------------------------------------------------------
            points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            points_transformed = view_transformer.transform_points(points)
            kinematics_data = kinematics_estimator.update(detections, points_transformed, frame.shape)
            
            # ------------------------------------------------------
            # Phase 3: 排放分析 (Emission Analysis)
            # ------------------------------------------------------
            emission_data = brake_model.process(kinematics_data, detections, plate_cache)
            
            for tid, d in emission_data.items():
                # 1. 更新注册表中的累积统计 (OpMode计数, 总排放量)
                registry.update_emission_stats(
                    tid=tid, 
                    op_mode=d['op_mode'], 
                    emission_rate_mg_s=d['emission_rate']
                )
                
                # 2. [新增] 写入数据库微观表 (Micro Table)
                # 过滤噪点: 只有存活一定时间的车辆才记录微观数据
                record = registry.get_record(tid)
                if record and (frame_id - record['first_frame'] > 5):
                    db_manager.insert_micro(frame_id, tid, d)
            
            # ------------------------------------------------------
            # Phase 4: 标签生成 (Label Generation)
            # ------------------------------------------------------
            labels = []
            for tid in detections.tracker_id:
                if tid in emission_data:
                    d = emission_data[tid]
                    
                    # UI 显示优化：尝试显示当前加权投票的“预测”结果
                    display_type = d['type_str']
                    if tid not in plate_cache:
                        hist = registry.get_history(tid)
                        if hist:
                            scores = defaultdict(float)
                            for h in hist: scores[h['color']] += h['area']
                            best = max(scores, key=scores.get)
                            display_type = f"{best}?"

                    state_tag = "[BRAKE]" if d['op_mode'] == 0 else \
                                "[IDLE]" if d['op_mode'] == 1 else "[GO]"
                    labels.append(f"#{tid} {display_type} | {d['speed']:.1f}m/s {state_tag}")
                else:
                    labels.append(f"#{tid} Init...")
            
            # ------------------------------------------------------
            # Phase 5: 渲染 (Rendering)
            # ------------------------------------------------------
            annotated_frame = visualizer.render(frame, detections, labels)
            sink.write_frame(annotated_frame)
            
            display_frame = resize_with_pad(annotated_frame, (1280, 720))
            cv2.imshow(WINDOW_NAME, display_frame)
            if cv2.waitKey(1) == ord("q"): break

    cv2.destroyAllWindows()
    
    # ------------------------------------------------------
    # Phase 6: 收尾工作 (Finalize)
    # ------------------------------------------------------
    print("\n[处理结束] 正在处理剩余在场车辆...")
    # 强制检查所有剩余车辆 (模拟超时)
    for tid, record in registry.check_exits(frame_id + 1000):
        Reporter.print_exit_report(tid, record, kinematics_estimator)
        # 写入残留车辆到宏观表
        final_plate, final_type = _resolve_final_status(record)
        db_manager.insert_macro(tid, record, final_type, final_plate)

    # 关闭数据库连接 (确保缓冲区数据写入磁盘)
    db_manager.close()

# =========================================================================
# Helper: 解析最终车辆状态 (用于写入数据库)
# =========================================================================
def _resolve_final_status(record):
    """
    根据历史记录解析最终的车牌颜色和车辆类型
    (逻辑与 Reporter 保持一致，用于数据库存储)
    """
    history = record.get('plate_history', [])
    final_plate = "Unknown"
    
    # 面积加权投票
    if history:
        scores = defaultdict(float)
        for e in history: 
            scores[e['color']] += e['area']
        if scores: 
            final_plate = max(scores, key=scores.get)

    # 类型推导
    final_type = "Calculating..."
    if final_plate != "Unknown":
        if final_plate == "Green": final_type = "LDV-electric"
        elif final_plate == "Yellow": final_type = "HDV-diesel"
        elif final_plate == "Blue": final_type = "LDV-gasoline"
        
        # 特殊逻辑修正
        if record.get('class_id') in [5, 7] and final_plate == "Green":
            final_type = "HDV-electric (Large EV)"
    else:
        # 兜底逻辑
        if record.get('class_id') in [5, 7]:
            final_type = "HDV-electric (Fallback)"
        else:
            final_type = "LDV-gasoline (Fallback)"
            
    return final_plate, final_type

if __name__ == "__main__":
    main()
