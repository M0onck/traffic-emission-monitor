import cv2
import numpy as np
import supervision as sv
from collections import defaultdict
from ui.renderer import resize_with_pad, LabelData

class TrafficMonitorEngine:
    """
    [应用层] 交通监测引擎
    封装主循环逻辑，协调各模块工作。
    """
    def __init__(self, config, components):
        """
        :param config: 全局配置对象 (settings module)
        :param components: 包含已初始化的组件字典
               {
                 'model': YOLO, 'tracker': ByteTrack, 'registry': VehicleRegistry,
                 'db': DatabaseManager, 'visualizer': Visualizer,
                 'kinematics': KinematicsEstimator (opt), 'brake_model': BrakeEmissionModel (opt),
                 'ocr_worker': AsyncOCRManager (opt), 'transformer': ViewTransformer
               }
        """
        self.cfg = config
        self.comps = components
        
        # 提取默认使用的组件
        self.model = components['model']
        self.tracker = components['tracker']
        self.registry = components['registry']
        self.visualizer = components['visualizer']
        self.smoother = components.get('smoother')
        self.db = components['db']
        self.classifier = components['classifier']
        
        # 状态缓存
        self.plate_cache = {}
        self.plate_retry = {}
        
        # 功能开关
        self.debug_mode = config.DEBUG_MODE
        self.motion_on = config.ENABLE_MOTION
        self.ocr_on = config.ENABLE_OCR
        self.emission_req = config.ENABLE_EMISSION

    def run(self):
        """执行主循环"""
        video_info = sv.VideoInfo.from_video_path(self.cfg.VIDEO_PATH)
        print(f">>> [Engine] 开始处理视频: {self.cfg.VIDEO_PATH}")
        
        with sv.VideoSink(self.cfg.TARGET_VIDEO_PATH, video_info=video_info) as sink:
            for frame_idx, frame in enumerate(sv.get_video_frames_generator(self.cfg.VIDEO_PATH)):
                frame_id = frame_idx + 1
                
                # 1. 处理单帧
                annotated_frame = self.process_frame(frame, frame_id)
                
                # 2. 写入视频
                sink.write_frame(annotated_frame)
                
                # 3. 屏幕显示
                display = resize_with_pad(annotated_frame, (1280, 720))
                cv2.imshow("Traffic Monitor", display)
                if cv2.waitKey(1) == ord('q'):
                    break
        
        self.cleanup(frame_id)

    def process_frame(self, frame, frame_id):
        img_h, img_w = frame.shape[:2]
        
        # --- 1. 感知 (Detection & Tracking) ---
        res = self.model(frame, conf=0.3, iou=0.5, agnostic_nms=True, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(res)
        detections = detections[np.isin(detections.class_id, self.cfg.YOLO_INTEREST_CLASSES)]
        detections = self.tracker.update_with_detections(detections)
        if self.smoother:
            detections = self.smoother.update_with_detections(detections)

        # --- 2. 业务更新 (Registry) ---
        self.registry.update(detections, frame_id, self.model)
        self._handle_exits(frame_id)
        
        # --- 3. 异步 OCR ---
        if self.ocr_on:
            self._handle_ocr(frame, frame_id, detections)

        # --- 4. 物理估算 (Kinematics) ---
        kinematics_data = {}
        if self.motion_on and self.comps.get('kinematics'):
            points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            transformed = self.comps['transformer'].transform_points(points)
            kinematics_data = self.comps['kinematics'].update(detections, transformed, frame.shape)

        # --- 5. 排放计算 (Emission) ---
        emission_data = {}
        if self.emission_req and self.motion_on and self.comps.get('brake_model'):
            emission_data = self.comps['brake_model'].process(
                kinematics_data, detections, self.plate_cache, self.classifier
            )
            self._save_micro_logs(frame_id, emission_data)

        # --- 6. 数据打包与渲染 ---
        label_data_list = self._prepare_labels(detections, kinematics_data, emission_data)
        return self.visualizer.render(frame, detections, label_data_list)

    def _handle_exits(self, frame_id):
        """处理车辆离场"""
        for tid, record in self.registry.check_exits(frame_id):
            if self.debug_mode and self.comps.get('reporter'):
                self.comps['reporter'].print_exit_report(
                    tid,
                    record,
                    self.comps.get('kinematics'),
                    self.classifier
                )
            final_plate, final_type = self.classifier.resolve_type(
                record['class_id'], plate_history=record.get('plate_history', [])
            )
            self.db.insert_macro(tid, record, final_type, final_plate)

    def _handle_ocr(self, frame, frame_id, detections):
        """OCR 任务分发与结果回收"""
        worker = self.comps.get('ocr_worker')
        if not worker: return

        img_h, img_w = frame.shape[:2]
        
        # 生产任务
        if frame_id % self.cfg.OCR_INTERVAL == 0:
            for tid, box, cid in zip(detections.tracker_id, detections.xyxy, detections.class_id):
                if frame_id - self.plate_retry.get(tid, -999) < self.cfg.OCR_RETRY_COOLDOWN:
                    continue
                
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1+x2)/2, (y1+y2)/2
                if not (0.1*img_w < cx < 0.9*img_w and 0.4*img_h < cy < 0.98*img_h):
                    continue
                
                if (x2-x1)*(y2-y1) > self.cfg.MIN_PLATE_AREA:
                    crop = frame[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)].copy()
                    if worker.push_task(tid, crop, cid):
                        self.plate_retry[tid] = frame_id

        # 回收结果
        for (tid, color, conf, area) in worker.get_results():
            self.registry.add_plate_history(tid, color, area, conf)
            if conf > self.cfg.OCR_CONF_THRESHOLD:
                self.plate_cache[tid] = color
                if tid in self.plate_retry: del self.plate_retry[tid]

    def _save_micro_logs(self, frame_id, emission_data):
        """保存微观数据"""
        for tid, d in emission_data.items():
            self.registry.update_emission_stats(
                tid,
                d['op_mode'],
                d['emission_rate'],
                d['speed']
            )
            rec = self.registry.get_record(tid)
            if rec and (frame_id - rec['first_frame'] > 5):
                self.db.insert_micro(frame_id, tid, d)

    def _prepare_labels(self, detections, kinematics_data, emission_data):
        """构建 LabelData 对象列表"""
        labels = []
        for tid, class_id in zip(detections.tracker_id, detections.class_id):
            data = LabelData(track_id=tid, class_id=class_id)
            
            # 填充速度
            if tid in kinematics_data:
                data.speed = kinematics_data[tid]['speed']
            
            # 填充排放与类型
            if tid in emission_data:
                d = emission_data[tid]
                data.emission_info = d
                data.display_type = d['type_str']
                if not self.ocr_on and "(Def)" not in data.display_type:
                    data.display_type += "(Def)"
            else:
                # 回退类型推断
                hist = self.registry.get_history(tid)
                color = self.plate_cache.get(tid)
                _, data.display_type = self.classifier.resolve_type(
                    class_id, plate_history=hist, plate_color_override=color
                )
            
            labels.append(data)
        return labels

    def cleanup(self, final_frame_id):
        print("\n[Engine] 正在清理资源...")
        if self.comps.get('ocr_worker'):
            self.comps['ocr_worker'].stop()
        cv2.destroyAllWindows()
        
        # 处理剩余车辆
        print("[Engine] 保存剩余车辆数据...")
        self._handle_exits(final_frame_id + 1000)
        self.db.close()
