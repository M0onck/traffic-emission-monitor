import cv2
import numpy as np
import supervision as sv
from collections import defaultdict
from ui.renderer import resize_with_pad, LabelData

class TrafficMonitorEngine:
    """
    [应用层] 交通监测引擎
    封装主循环逻辑，协调各模块工作。
    
    [架构模式]
    1. 在线阶段 (On-line): 仅记录运动学轨迹 (Trajectory)，不做最终排放结算。
    2. 离场阶段 (On-Exit): 基于最终确定的车型 (Voted Class)，回放轨迹并计算排放，统一入库。
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
        # 更新车辆档案，计算加权投票，但此时不锁定车型
        self.registry.update(detections, frame_id, self.model)
        
        # 检查是否有车辆离场，触发结算
        self._handle_exits(frame_id)
        
        # --- 3. 异步 OCR ---
        if self.ocr_on:
            self._handle_ocr(frame, frame_id, detections)

        # --- 4. 物理估算 (Kinematics) ---
        kinematics_data = {}
        if self.motion_on and self.comps.get('kinematics'):
            # 获取底部中心点 (像素坐标)
            points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            
            # 坐标变换
            transformed = self.comps['transformer'].transform_points(points)
            
            # 计算运动学数据 (速度/加速度) - 注意：这里仍然对所有目标计算，以维持滤波器的连续性
            kinematics_data = self.comps['kinematics'].update(detections, transformed, frame.shape)
            
            # ROI 过滤逻辑
            # 建立 TID -> 像素坐标 的映射，用于快速查找
            tid_to_pixel = {tid: pt for tid, pt in zip(detections.tracker_id, points)}
            transformer = self.comps['transformer']

            for tid, k_data in kinematics_data.items():
                raw_point = tid_to_pixel.get(tid)
                
                # 仅当车辆中心点位于标定区域内时，才记录数据
                if raw_point is not None and transformer.is_in_roi(raw_point):
                    self.registry.append_kinematics(tid, frame_id, k_data['speed'], k_data['accel'])

        # --- 5. 排放计算 (仅用于 UI 展示) ---
        emission_data = {}
        if self.emission_req and self.motion_on and kinematics_data:
            # A. 统一计算 VSP (基于瞬时车型，仅供显示)
            vsp_map = {} 
            vsp_calc = self.comps.get('vsp_calculator')
            if vsp_calc:
                for tid, k_data in kinematics_data.items():
                    mask = detections.tracker_id == tid
                    if not np.any(mask): continue
                    class_id = int(detections.class_id[mask][0]) 
                    vsp = vsp_calc.calculate(k_data['speed'], k_data['accel'], class_id)
                    vsp_map[tid] = vsp

            # B. 运行刹车模型 (UI用)
            # 这里调用的是 BrakeModel.process (批处理接口)，它内部会复用 calculate_single_point
            if self.comps.get('brake_model'):
                emission_data = self.comps['brake_model'].process(
                    kinematics_data, detections, self.plate_cache, self.classifier, vsp_map,
                    dt=1.0/self.cfg.FPS
                )

        # --- 6. 数据打包与渲染 ---
        label_data_list = self._prepare_labels(detections, kinematics_data, emission_data)
        return self.visualizer.render(frame, detections, label_data_list)

    def _handle_exits(self, frame_id):
        """
        处理车辆离场
        """
        for tid, record in self.registry.check_exits(frame_id):
            
            # 1. 确定最终车型 (使用加权投票结果)
            final_plate, final_type_str = self.classifier.resolve_type(
                record['class_id'], record.get('plate_history', [])
            )
            
            # 2. 回放轨迹，计算并保存微观历史数据
            if self.emission_req and 'trajectory' in record:
                self._calculate_and_save_history(tid, record, final_type_str)

            # 3. 写入宏观汇总 (Macro)
            # record 中的统计值已经在第2步中被更新完毕
            self.db.insert_macro(tid, record, final_type_str, final_plate)

            # 4. 打印调试报告
            if self.debug_mode and self.comps.get('reporter'):
                self.comps['reporter'].print_exit_report(
                    tid,
                    record,
                    self.comps.get('kinematics'),
                    self.classifier
                )

    def _calculate_and_save_history(self, tid, record, final_type_str):
        """
        [核心逻辑] 离场结算
        功能：基于最终确定的车型，回放轨迹，调用模型的 calculate_single_point 接口重新计算排放。
        """
        trajectory = record.get('trajectory', [])

        # 轨迹清洗 (Trajectory Cleaning)
        # 理由：车辆进出画面边缘时检测框不稳定，导致物理计算出现巨大尖峰。
        # 策略：丢弃前 5 帧 (入场不稳定期) 和后 5 帧 (离场截断期)
        TRIM_SIZE = 5 
        if len(trajectory) > (TRIM_SIZE * 2 + 5): # 确保剩余至少5帧有效数据
            trajectory = trajectory[TRIM_SIZE : -TRIM_SIZE]
        else:
            # 如果轨迹太短，不够切，则不做处理或直接放弃
            pass

        if not trajectory:
            return

        # 获取组件引用
        vsp_calc = self.comps.get('vsp_calculator')
        brake_model = self.comps.get('brake_model')
        tire_model = self.comps.get('tire_model')
        
        if not (vsp_calc and brake_model and tire_model):
            return

        # 准备参数
        final_class_id = record['class_id']
        is_electric = "electric" in final_type_str
        
        # 确定 MOVES 大类
        category = 'CAR'
        if final_class_id == self.cfg.YOLO_CLASS_BUS: category = 'BUS'
        elif final_class_id == self.cfg.YOLO_CLASS_TRUCK: category = 'TRUCK'

        # 计算 dt
        dt = 1.0 / self.cfg.FPS

        # 遍历轨迹进行回放计算
        for point in trajectory:
            v = point['speed']
            a = point['accel']
            fid = point['frame_id']
            
            # A. 重算 VSP
            vsp = vsp_calc.calculate(v, a, final_class_id)
            
            # B. & C. 重算 OpMode 和 刹车排放
            # [修改点] 调用新接口 calculate_single_point
            brake_res = brake_model.calculate_single_point(
                v_ms=v, 
                a_ms2=a, 
                vsp=vsp, 
                vehicle_class_id=final_class_id, 
                dt=dt,
                type_str=final_type_str
            )
            
            brake_emission = brake_res['emission_mass']
            op_mode = brake_res['op_mode']
            
            # D. 重算 轮胎排放
            tire_res = tire_model.process(
                vehicle_type=category.lower(),
                speed_ms=v, accel_ms2=a, dt=1.0/self.cfg.FPS,
                vsp_kW_t=vsp,
                is_electric=is_electric,
                mass_factor=self.cfg.MASS_FACTOR_EV
            )
            tire_emission = tire_res['pm10']

            # E. 更新 Registry 统计值 (用于 Macro 表)
            self.registry.update_emission_stats(tid, op_mode, brake_emission, v)
            self.registry.update_tire_stats(tid, tire_emission)

            # F. 构造微观数据包并入库
            data = {
                'type_str': final_type_str,
                'plate_color': "Resolved",
                'speed': v, 
                'accel': a, 
                'vsp': vsp,
                'op_mode': op_mode,
                'brake_emission': brake_emission,
                'tire_emission': tire_emission
            }
            # 显式传入历史帧号 fid
            self.db.insert_micro(fid, tid, data)
            
        # 强制刷写一次 DB 缓冲区
        self.db.flush_micro_buffer()

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

    def _prepare_labels(self, detections, kinematics_data, emission_data):
        """构建 LabelData 对象列表 (UI显示用)"""
        labels = []
        for tid, raw_class_id in zip(detections.tracker_id, detections.class_id):
            # 优先从 Registry 获取经过平滑投票的 class_id
            record = self.registry.get_record(tid)
            voted_class_id = int(raw_class_id)
            if record:
                voted_class_id = record['class_id']

            data = LabelData(track_id=tid, class_id=voted_class_id)
            
            if tid in kinematics_data:
                data.speed = kinematics_data[tid]['speed']
            
            if tid in emission_data:
                d = emission_data[tid]
                data.emission_info = d
                data.display_type = d['type_str']
                if not self.ocr_on and "(Def)" not in data.display_type:
                    data.display_type += "(Def)"
            else:
                hist = self.registry.get_history(tid)
                color = self.plate_cache.get(tid)
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
