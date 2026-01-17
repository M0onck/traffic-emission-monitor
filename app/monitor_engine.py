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
        self.cfg = config
        self.comps = components
        
        self.model = components['model']
        self.tracker = components['tracker']
        self.registry = components['registry']
        self.visualizer = components['visualizer']
        self.smoother = components.get('smoother')
        self.db = components['db']
        self.classifier = components['classifier']
        
        self.plate_cache = {}
        self.plate_retry = {}
        
        self.debug_mode = config.DEBUG_MODE
        self.motion_on = config.ENABLE_MOTION
        self.ocr_on = config.ENABLE_OCR
        self.emission_req = config.ENABLE_EMISSION

    def run(self):
        video_info = sv.VideoInfo.from_video_path(self.cfg.VIDEO_PATH)
        print(f">>> [Engine] 开始处理视频: {self.cfg.VIDEO_PATH}")
        
        with sv.VideoSink(self.cfg.TARGET_VIDEO_PATH, video_info=video_info) as sink:
            for frame_idx, frame in enumerate(sv.get_video_frames_generator(self.cfg.VIDEO_PATH)):
                frame_id = frame_idx + 1
                annotated_frame = self.process_frame(frame, frame_id)
                sink.write_frame(annotated_frame)
                
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
            
            tid_to_pixel = {tid: pt for tid, pt in zip(detections.tracker_id, points)}
            transformer = self.comps['transformer']

            for tid, k_data in kinematics_data.items():
                raw_point = tid_to_pixel.get(tid)
                if raw_point is not None and transformer.is_in_roi(raw_point):
                    self.registry.append_kinematics(
                        tid, 
                        frame_id, 
                        k_data['speed'], 
                        k_data['accel'],
                        raw_x=k_data['curr_x'],
                        raw_y=k_data['curr_y']
                    )

        # --- 5. 排放计算 (仅用于 UI 展示) ---
        emission_data = {}
        if self.emission_req and self.motion_on and kinematics_data:
            vsp_map = {} 
            vsp_calc = self.comps.get('vsp_calculator')
            if vsp_calc:
                for tid, k_data in kinematics_data.items():
                    mask = detections.tracker_id == tid
                    if not np.any(mask): continue
                    class_id = int(detections.class_id[mask][0]) 
                    vsp = vsp_calc.calculate(k_data['speed'], k_data['accel'], class_id)
                    vsp_map[tid] = vsp

            if self.comps.get('brake_model'):
                emission_data = self.comps['brake_model'].process(
                    kinematics_data, detections, self.plate_cache, self.classifier, vsp_map,
                    dt=1.0/self.cfg.FPS
                )

        # --- 6. 数据打包与渲染 ---
        label_data_list = self._prepare_labels(detections, kinematics_data, emission_data)
        return self.visualizer.render(frame, detections, label_data_list)

    def _handle_exits(self, frame_id):
        for tid, record in self.registry.check_exits(frame_id):
            final_plate, final_type_str = self.classifier.resolve_type(
                record['class_id'], record.get('plate_history', [])
            )
            
            if self.emission_req and 'trajectory' in record:
                self._calculate_and_save_history(tid, record, final_type_str)

            self.db.insert_macro(tid, record, final_type_str, final_plate)

            if self.debug_mode and self.comps.get('reporter'):
                self.comps['reporter'].print_exit_report(
                    tid, record, self.comps.get('kinematics'), self.classifier
                )

    def _calculate_and_save_history(self, tid, record, final_type_str):
        """
        [核心逻辑] 离场结算
        """
        trajectory = record.get('trajectory', [])

        # 1. 轨迹清洗
        TRIM_SIZE = 5 
        if len(trajectory) > (TRIM_SIZE * 2 + 5): 
            trajectory = trajectory[TRIM_SIZE : -TRIM_SIZE]
        else:
            pass # 轨迹太短不做处理

        if not trajectory:
            return

        # 2. [修改] 全局轨迹重构 (传入 class_id 以应用差异化物理约束)
        if len(trajectory) > 10 and 'raw_x' in trajectory[0]:
             trajectory = self._refine_trajectory_global(trajectory, record['class_id']) 

        # 3. 获取组件引用
        vsp_calc = self.comps.get('vsp_calculator')
        brake_model = self.comps.get('brake_model')
        tire_model = self.comps.get('tire_model')
        
        if not (vsp_calc and brake_model and tire_model):
            return

        # 准备参数
        final_class_id = record['class_id']
        is_electric = "electric" in final_type_str
        
        category = 'CAR'
        if final_class_id == self.cfg.YOLO_CLASS_BUS: category = 'BUS'
        elif final_class_id == self.cfg.YOLO_CLASS_TRUCK: category = 'TRUCK'

        dt = 1.0 / self.cfg.FPS

        # 4. 遍历轨迹进行回放计算
        for point in trajectory:
            v = point['speed']
            a = point['accel']
            fid = point['frame_id']
            
            # A. 重算 VSP
            vsp = vsp_calc.calculate(v, a, final_class_id)
            
            # B. & C. 重算 OpMode 和 刹车排放
            brake_res = brake_model.calculate_single_point(
                v_ms=v, a_ms2=a, vsp=vsp, vehicle_class_id=final_class_id, dt=dt, type_str=final_type_str
            )
            brake_emission = brake_res['emission_mass']
            op_mode = brake_res['op_mode']
            
            # D. 重算 轮胎排放
            tire_res = tire_model.process(
                vehicle_type=category.lower(),
                speed_ms=v, accel_ms2=a, dt=dt,
                vsp_kW_t=vsp,
                is_electric=is_electric,
                mass_factor=self.cfg.MASS_FACTOR_EV
            )
            tire_emission = tire_res['pm10']

            # E. 更新 Registry 统计值
            if hasattr(self.registry, 'accumulate_opmode'):
                self.registry.accumulate_opmode(record, op_mode)
                self.registry.accumulate_brake_emission(record, brake_emission)
                self.registry.accumulate_tire_emission(record, tire_emission)
            else:
                self.registry.update_emission_stats(record, op_mode, brake_emission)
                self.registry.update_tire_stats(record, tire_emission)

            # F. 入库
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
            self.db.insert_micro(fid, tid, data)
            
        self.db.flush_micro_buffer()

    def _handle_ocr(self, frame, frame_id, detections):
        worker = self.comps.get('ocr_worker')
        if not worker: return

        img_h, img_w = frame.shape[:2]
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

        for (tid, color, conf, area) in worker.get_results():
            self.registry.add_plate_history(tid, color, area, conf)
            if conf > self.cfg.OCR_CONF_THRESHOLD:
                self.plate_cache[tid] = color
                if tid in self.plate_retry: del self.plate_retry[tid]

    def _prepare_labels(self, detections, kinematics_data, emission_data):
        labels = []
        for tid, raw_class_id in zip(detections.tracker_id, detections.class_id):
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

    def _refine_trajectory_global(self, trajectory, class_id):
        """
        [离场重构] 全局轨迹优化器 v3.1
        改进点：在 Step 4 密集计算阶段引入基于车型的物理约束 (Physical Constraint)，
        将离谱值在插值前就进行截断。
        """
        if len(trajectory) < 5: return trajectory

        # --- [新增] 定义物理约束阈值 (m/s^2) ---
        # 轿车极限较高，大车受限于功率重量比，极限较低
        ACCEL_LIMITS = {
            self.cfg.YOLO_CLASS_CAR: 5.0,    # 轿车：急加速能力强
            self.cfg.YOLO_CLASS_BUS: 2.5,    # 大巴：载人求稳，且自重大
            self.cfg.YOLO_CLASS_TRUCK: 2.0   # 卡车：最笨重，极难超过 0.2g
        }
        # 默认为轿车标准 (5.0)
        phys_limit = ACCEL_LIMITS.get(class_id, 5.0)

        # 1. 提取原始数据
        raw_x = np.array([p['raw_x'] for p in trajectory])
        raw_y = np.array([p['raw_y'] for p in trajectory])
        dt = 1.0 / self.cfg.FPS
        n_points = len(raw_x)

        # 2. 定义带 Padding 的平滑函数
        def bidirectional_smooth(data, window=15):
            pad_width = window // 2
            padded = np.pad(data, (pad_width, pad_width), mode='edge')
            kernel = np.ones(window) / window
            fwd = np.convolve(padded, kernel, mode='valid')
            padded_rev = np.pad(data[::-1], (pad_width, pad_width), mode='edge')
            bwd = np.convolve(padded_rev, kernel, mode='valid')[::-1]
            return (fwd + bwd) / 2.0

        # 3. 计算平滑速度
        smooth_x = bidirectional_smooth(raw_x, window=15)
        smooth_y = bidirectional_smooth(raw_y, window=15)
        
        grads_x = np.gradient(smooth_x, dt)
        grads_y = np.gradient(smooth_y, dt)
        refined_speed = np.sqrt(grads_x**2 + grads_y**2)
        smooth_speed = bidirectional_smooth(refined_speed, window=15)

        # 4. 计算“密集”加速度 (Dense Acceleration) 并应用物理约束
        k = 7 # 半窗口
        dense_accel = np.zeros(n_points)
        for i in range(n_points):
            idx_start = max(0, i - k)
            idx_end = min(n_points - 1, i + k)
            dv = smooth_speed[idx_end] - smooth_speed[idx_start]
            dt_span = (idx_end - idx_start) * dt
            
            if dt_span > 1e-4:
                raw_val = dv / dt_span
                # [关键修改] 发现离谱值自动落回约束范围
                # np.clip 会将 < -limit 的值变为 -limit，> limit 的值变为 limit
                dense_accel[i] = np.clip(raw_val, -phys_limit, phys_limit)
            else:
                dense_accel[i] = 0.0

        # 5. 锚点重采样与插值
        anchor_step = 15
        anchor_indices = np.arange(0, n_points, anchor_step)
        if anchor_indices[-1] != n_points - 1:
            anchor_indices = np.append(anchor_indices, n_points - 1)
            
        anchor_values = dense_accel[anchor_indices]
        anchor_values[0] = 0.0
        anchor_values[-1] = 0.0
        
        final_accel = np.interp(np.arange(n_points), anchor_indices, anchor_values)

        # 6. 回写结果
        for i, point in enumerate(trajectory):
            point['rt_speed'] = point['speed'] 
            point['rt_accel'] = point['accel']
            point['speed'] = float(smooth_speed[i])
            point['accel'] = float(final_accel[i])

        return trajectory

    def cleanup(self, final_frame_id):
            print("\n[Engine] 正在清理资源...")
            if self.comps.get('ocr_worker'):
                self.comps['ocr_worker'].stop()
            cv2.destroyAllWindows()
            print("[Engine] 保存剩余车辆数据...")
            self._handle_exits(final_frame_id + 1000)
            self.db.close()
