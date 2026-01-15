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
        brake_data = {}
        tire_data = {}
        emission_data = {}
        
        if self.emission_req and self.motion_on and kinematics_data:
            # [步骤 A] 统一计算 VSP
            vsp_map = {} # {tid: vsp_value}
            vsp_calc = self.comps.get('vsp_calculator')
            
            if vsp_calc:
                for tid, k_data in kinematics_data.items():
                    # 查找 class_id
                    mask = detections.tracker_id == tid
                    if not np.any(mask): continue
                    class_id = int(detections.class_id[mask][0])
                    
                    # 统一计算
                    vsp = vsp_calc.calculate(
                        v_ms=k_data['speed'],
                        a_ms2=k_data['accel'],
                        class_id=class_id
                    )
                    vsp_map[tid] = vsp

            # [步骤 B] 运行刹车模型 (传入 vsp_map)
            if self.comps.get('brake_model'):
                brake_data = self.comps['brake_model'].process(
                    kinematics_data, detections, self.plate_cache, self.classifier, 
                    vsp_map=vsp_map  # 注入统一的 VSP
                )
                emission_data = brake_data

            # [步骤 C] 运行轮胎模型 (传入 vsp_map)
            if self.comps.get('tire_model'):
                for tid, k_data in kinematics_data.items():
                    mask = detections.tracker_id == tid
                    if not np.any(mask): continue
                    
                    class_id = int(detections.class_id[mask][0])
                    
                    # 1. 修复硬编码：使用 self.cfg (注入的配置对象)
                    category = 'car'
                    if class_id == self.cfg.YOLO_CLASS_BUS:     # 使用注入的常量
                        category = 'bus'
                    elif class_id == self.cfg.YOLO_CLASS_TRUCK: # 使用注入的常量
                        category = 'truck'
                    
                    # 2. 准备 EV 参数
                    # 尝试从 brake_data 获取详细车型
                    type_str = "Unknown"
                    if tid in brake_data:
                        type_str = brake_data[tid].get('type_str', '')
                    else:
                        # 回退逻辑：手动解析
                        color = self.plate_cache.get(tid, "Unknown")
                        _, type_str = self.classifier.resolve_type(class_id, plate_color_override=color)

                    # 3. 运行模型 (传入 EV 修正参数)
                    t_res = self.comps['tire_model'].process(
                        vehicle_type=category,
                        speed_ms=k_data['speed'],
                        accel_ms2=k_data['accel'],
                        dt=1.0/self.cfg.FPS,
                        mass_kg=None,  # 保持 None，由内部基准质量决定
                        vsp_kW_t=vsp_map.get(tid, 0.0),
                        # 新增参数：通过注入的配置传递 EV 因子
                        is_electric=("electric" in type_str),
                        mass_factor=self.cfg.MASS_FACTOR_EV 
                    )
                    tire_data[tid] = t_res

            # [步骤 D] 保存日志
            self._save_micro_logs(frame_id, brake_data, tire_data, kinematics_data)
        
        # [调试] 实时监控异常排放峰值
        if self.debug_mode:
            for tid, t_res in tire_data.items():
                # 设定一个敏感阈值，例如单帧轮胎排放 > 1.0 mg (这在物理上很大)
                tire_val = t_res.get('pm10', 0)
                if tire_val > 1.0: 
                    dbg = t_res.get('debug_info', {})
                    print(f"\n[WARN] High Tire Emission ID:{tid} | Val:{tire_val:.2f}mg")
                    print(f"   -> Cause: Mass={dbg.get('mass_kg')}kg | Force={dbg.get('force_N')}N | Method={dbg.get('calc_method')}")
                    print(f"   -> Input: V={dbg.get('speed_ms')}m/s | A={dbg.get('accel')}m/s²")

            for tid, b_res in brake_data.items():
                # 设定刹车阈值，例如单帧排放 > 0.5 mg (对应 rate > 15 mg/s)
                brake_rate = b_res.get('emission_rate', 0)
                # 转化为单帧质量: rate * dt
                brake_val_mg = brake_rate * (1.0 / self.cfg.FPS)
                
                if brake_val_mg > 0.5:
                    dbg = b_res.get('debug_info', {})
                    print(f"\n[WARN] High Brake Emission ID:{tid} | Val:{brake_val_mg:.2f}mg (Rate: {brake_rate:.1f}mg/s)")
                    print(f"   -> State: OpMode={dbg.get('op_mode')} | BaseEF={dbg.get('base_rate_mg_s')}")
                    print(f"   -> Input: V={dbg.get('v_ms')} | A={dbg.get('a_ms2')} | VSP={dbg.get('vsp')}")

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

    def _save_micro_logs(self, frame_id, brake_data, tire_data, kinematics_data):
        """
        保存微观数据
        [修复] 增加了 kinematics_data 参数，用于在排放数据缺失时回退获取速度信息
        """
        # 获取所有涉及的 TID 集合
        all_tids = set(brake_data.keys()) | set(tire_data.keys())
        
        for tid in all_tids:
            b_info = brake_data.get(tid, {})
            t_info = tire_data.get(tid, {})
            
            # 更新 Registry
            # 1. 刹车
            if b_info:
                self.registry.update_emission_stats(
                    tid, b_info['op_mode'], b_info['emission_rate'], b_info['speed']
                )
            # 2. 轮胎
            if t_info:
                self.registry.update_tire_stats(tid, t_info['pm10'])
            
            # 写入数据库 (合并数据)
            rec = self.registry.get_record(tid)
            if rec and (frame_id - rec['first_frame'] > 5):
                # [修复] 速度获取逻辑：优先从 brake_data 取，没有则从 kinematics_data 取，防止 KeyError
                current_speed = b_info.get('speed', 0)
                if current_speed == 0 and tid in kinematics_data:
                    current_speed = kinematics_data[tid]['speed']

                # 构造合并后的数据包
                merged_data = {
                    'type_str': b_info.get('type_str', 'Unknown'),
                    'plate_color': b_info.get('plate_color', 'Unknown'),
                    'speed': current_speed,
                    'accel': b_info.get('accel', 0),
                    'vsp': b_info.get('vsp', 0),
                    'op_mode': b_info.get('op_mode', -1),
                    'brake_emission': b_info.get('emission_rate', 0),
                    'tire_emission': t_info.get('pm10', 0) 
                }
                self.db.insert_micro(frame_id, tid, merged_data)

    def _prepare_labels(self, detections, kinematics_data, emission_data):
        """构建 LabelData 对象列表"""
        labels = []
        for tid, raw_class_id in zip(detections.tracker_id, detections.class_id):
            # 优先从 Registry 获取经过平滑投票的 class_id
            # 这样屏幕上的车型显示会非常稳定，不会在 Car/Truck 之间闪烁
            record = self.registry.get_record(tid)
            voted_class_id = int(raw_class_id) # 默认回退
            if record:
                voted_class_id = record['class_id']

            data = LabelData(track_id=tid, class_id=voted_class_id)
            
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
                # 传入 voted_class_id 而不是 raw_class_id
                _, data.display_type = self.classifier.resolve_type(
                    voted_class_id, plate_history=hist, plate_color_override=color
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
