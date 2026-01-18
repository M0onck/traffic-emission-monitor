import cv2
import numpy as np
import supervision as sv
from collections import defaultdict
from ui.renderer import resize_with_pad, LabelData
from perception.camera import CameraPreprocessor

class TrafficMonitorEngine:
    """
    [应用层] 交通监测引擎
    封装主循环逻辑，协调各模块工作。
    负责感知、业务更新、OCR、物理估算、排放计算以及数据可视化。
    """
    def __init__(self, config, components):
        self.cfg = config
        self.comps = components
        
        # 核心组件引用
        self.model = components['model']
        self.tracker = components['tracker']
        self.registry = components['registry']
        self.visualizer = components['visualizer']
        self.smoother = components.get('smoother')
        self.db = components['db']
        self.classifier = components['classifier']
        
        # 缓存与状态管理
        self.plate_cache = {}
        self.plate_retry = {}
        
        # 功能开关
        self.debug_mode = config.DEBUG_MODE
        self.motion_on = config.ENABLE_MOTION
        self.ocr_on = config.ENABLE_OCR
        self.emission_req = config.ENABLE_EMISSION

        # 初始化相机前处理器
        self.camera_preprocessor = CameraPreprocessor(config)

    def run(self):
        # 1. 获取原始视频信息
        video_info = sv.VideoInfo.from_video_path(self.cfg.VIDEO_PATH)
        
        # 2. 预读取逻辑：修正输出分辨率
        # 必须先读取一帧并通过 CameraPreprocessor 处理，以获取裁剪后的实际尺寸。
        if hasattr(self, 'camera_preprocessor'):
            print(">>> [Engine] 正在计算去畸变后的输出尺寸...")
            temp_cap = cv2.VideoCapture(self.cfg.VIDEO_PATH)
            ret, temp_frame = temp_cap.read()
            temp_cap.release()
            
            if ret:
                # 预处理一帧，获取新尺寸
                processed = self.camera_preprocessor.preprocess(temp_frame)
                h, w = processed.shape[:2]
                
                # 检查尺寸是否发生变化
                if w != video_info.width or h != video_info.height:
                    print(f">>> [Engine] 分辨率已自适应调整: {video_info.width}x{video_info.height} -> {w}x{h}")
                    # 更新 VideoInfo 以匹配新的帧尺寸
                    video_info.width = w
                    video_info.height = h
            else:
                print(">>> [Warn] 预读取失败，将使用原始分辨率（可能导致写入错误）。")

        print(f">>> [Engine] 开始处理视频: {self.cfg.VIDEO_PATH}")
        
        # 3. 使用修正后的 video_info 初始化 Sink
        with sv.VideoSink(self.cfg.TARGET_VIDEO_PATH, video_info=video_info) as sink:
            for frame_idx, frame in enumerate(sv.get_video_frames_generator(self.cfg.VIDEO_PATH)):
                frame_id = frame_idx + 1
                
                # 核心处理流水线 (内部会调用 camera_preprocessor 改变尺寸)
                annotated_frame = self.process_frame(frame, frame_id)
                
                # 此时 annotated_frame 的尺寸与 sink 的设定尺寸一致，写入成功
                sink.write_frame(annotated_frame)
                
                # 实时显示
                display = resize_with_pad(annotated_frame, (1280, 720))
                cv2.imshow("Traffic Monitor", display)
                if cv2.waitKey(1) == ord('q'):
                    break
        
        self.cleanup(frame_id)

    def process_frame(self, frame, frame_id):
        """
        单帧处理逻辑
        """
        # --- Step 0: 图像去畸变校准 ---
        # 必须在这一步做，因为所有的后续逻辑 (YOLO坐标、ROI标定、测速) 
        # 都必须基于"变直"后的几何空间。
        frame = self.camera_preprocessor.preprocess(frame)

        img_h, img_w = frame.shape[:2]
        
        # --- 1. 感知 (Detection & Tracking) ---
        res = self.model(frame, conf=0.3, iou=0.5, agnostic_nms=True, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(res)
        
        # 过滤非感兴趣类别
        detections = detections[np.isin(detections.class_id, self.cfg.YOLO_INTEREST_CLASSES)]
        detections = self.tracker.update_with_detections(detections)
        
        # (可选) 轨迹平滑器
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
            # 提取底部中心点并进行透视变换
            points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            transformed = self.comps['transformer'].transform_points(points)
            
            # 更新卡尔曼滤波与速度估算
            kinematics_data = self.comps['kinematics'].update(detections, transformed, frame.shape)
            
            tid_to_pixel = {tid: pt for tid, pt in zip(detections.tracker_id, points)}
            transformer = self.comps['transformer']

            for tid, k_data in kinematics_data.items():
                raw_point = tid_to_pixel.get(tid)
                # 仅记录 ROI 区域内的运动数据
                if raw_point is not None and transformer.is_in_roi(raw_point):
                    self.registry.append_kinematics(
                        tid, 
                        frame_id, 
                        k_data['speed'], 
                        k_data['accel'],
                        raw_x=k_data['curr_x'],
                        raw_y=k_data['curr_y'],
                        pixel_x=raw_point[0], # [新增] 传入像素坐标 X
                        pixel_y=raw_point[1]  # [新增] 传入像素坐标 Y
                    )

        # --- 5. 排放计算 (仅用于 UI 实时展示，非最终入库数据) ---
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
        """
        处理离场车辆：结算数据、生成报表、存入数据库。
        """
        for tid, record in self.registry.check_exits(frame_id):
            # 解析最终车型和车牌
            final_plate, final_type_str = self.classifier.resolve_type(
                record['class_id'], record.get('plate_history', [])
            )
            
            # [新增] 使用几何连线法重算总里程 (覆盖原有的速度积分距离)
            self._recalculate_distance_geometric(record)

            # 如果启用排放计算，执行离场微观结算
            if self.emission_req and 'trajectory' in record:
                self._calculate_and_save_history(tid, record, final_type_str)

            # 存入宏观汇总表
            self.db.insert_macro(tid, record, final_type_str, final_plate)

            # 控制台输出报告
            if self.debug_mode and self.comps.get('reporter'):
                self.comps['reporter'].print_exit_report(
                    tid, record, self.comps.get('kinematics'), self.classifier
                )
    
    def _recalculate_distance_geometric(self, record):
        """
        [算法优化] 几何距离重算
        使用原始记录的像素轨迹点 -> 透视变换 -> 累加欧氏距离。
        相比速度积分法 (Speed * dt)，该方法不受滤波滞后影响，更接近真实物理轨迹长度。
        """
        trajectory = record.get('trajectory', [])
        if len(trajectory) < 2: return

        # 1. 提取有效的像素坐标
        pixels = []
        for p in trajectory:
            if p.get('pixel_x') is not None and p.get('pixel_y') is not None:
                pixels.append([p['pixel_x'], p['pixel_y']])
        
        if len(pixels) < 2: return
        
        # 2. 批量透视变换 (Pixel -> Meter)
        # transform_points 需要 numpy 数组 (N, 2)
        pts_phys = self.comps['transformer'].transform_points(np.array(pixels))
        
        # 3. 累加线段长度
        # dist = sum( norm(p[i] - p[i-1]) )
        diffs = pts_phys[1:] - pts_phys[:-1]
        dists = np.linalg.norm(diffs, axis=1)
        total_dist = float(np.sum(dists))
        
        # 4. 更新记录 (DatabaseManager 会基于这个新距离重算 avg_speed)
        record['total_distance_m'] = total_dist

    def _calculate_and_save_history(self, tid, record, final_type_str):
        """
        [核心逻辑] 离场结算 (v3.2 完整版)
        流程：
        1. 轨迹头尾清洗 (Trimming)
        2. 全局轨迹重构 (Global Refinement): 平滑位置 -> 大跨度微分 -> 物理截断 -> 锚点插值
        3. [Pass 1] 物理参数预计算: 算出 v, a, vsp, raw_opmode
        4. [Pass 2] 序列清洗: 使用状态机去除 OpMode 毛刺和非法跳变
        5. [Pass 3] 最终结算: 基于清洗后的 OpMode 查表计算排放并入库
        """
        trajectory = record.get('trajectory', [])

        # 1. 轨迹清洗 (Trajectory Cleaning)
        # 去除进出画面边缘时检测框不稳定的头尾数据
        TRIM_SIZE = 5 
        if len(trajectory) > (TRIM_SIZE * 2 + 5):
            trajectory = trajectory[TRIM_SIZE : -TRIM_SIZE]
            # 将 Trim 后的列表回写给 record
            # 这样 Console Reporter 拿到的就是去掉头尾噪声的干净数据了
            record['trajectory'] = trajectory
        else:
            return # 轨迹太短，放弃计算

        # 2. 全局轨迹重构 (Global Trajectory Refinement)
        # 传入 class_id 以应用基于车型的加速度物理约束 (如卡车 2.0m/s²)
        if len(trajectory) > 10 and 'raw_x' in trajectory[0]:
             trajectory = self._refine_trajectory_global(trajectory, record['class_id']) 

        # 3. 获取组件引用
        vsp_calc = self.comps.get('vsp_calculator')
        brake_model = self.comps.get('brake_model')
        tire_model = self.comps.get('tire_model')
        
        # 从 brake_model 中借用 OpMode 计算器实例
        opmode_calc = getattr(brake_model, 'opmode_calculator', None)
        
        if not (vsp_calc and brake_model and tire_model and opmode_calc):
            return

        # 准备基础参数
        final_class_id = record['class_id']
        category = 'CAR'
        if final_class_id == self.cfg.YOLO_CLASS_BUS: category = 'BUS'
        elif final_class_id == self.cfg.YOLO_CLASS_TRUCK: category = 'TRUCK'

        is_electric = "electric" in final_type_str
        dt = 1.0 / self.cfg.FPS

        # --- [Pass 1] 物理参数预计算 (Pre-calculation) ---
        pre_calc_data = []
        raw_opmodes = []

        for point in trajectory:
            v = point['speed']
            a = point['accel'] # 此处的 a 已经是经过物理约束和锚点插值后的高质量数据
            fid = point['frame_id']
            
            # 计算 VSP
            vsp = vsp_calc.calculate(v, a, final_class_id)
            
            # 初步判定 OpMode (Raw)
            raw_op = opmode_calc.get_opmode(v, a, vsp)
            
            pre_calc_data.append({
                'fid': fid, 'v': v, 'a': a, 'vsp': vsp
            })
            raw_opmodes.append(raw_op)

        # --- [Pass 2] OpMode 序列清洗 (Sequence Cleaning) ---
        # 引入状态机逻辑，消除不合理的工况跳变 (如 Brake <-> Hard Accel)
        clean_opmodes = self._clean_opmode_sequence(raw_opmodes)

        # --- [Pass 3] 排放结算与入库 (Final Settlement) ---
        for i, data in enumerate(pre_calc_data):
            op_mode = clean_opmodes[i] # 使用清洗后的工况
            v, a, vsp = data['v'], data['a'], data['vsp']
            
            # A. 刹车排放结算
            # 1. 查基础排放率
            brake_base = brake_model._get_emission_factor(op_mode, category)
            
            # 2. 应用 EV 修正 (再生制动)
            brake_factor = 1.0
            if is_electric:
                # OpMode 0 (Braking) 时 Regen 贡献较小(0.4)，其他工况(0.1)
                regen_factor = 0.4 if op_mode == 0 else 0.1
                brake_factor = self.cfg.MASS_FACTOR_EV * regen_factor
            
            brake_emission = brake_base * brake_factor * dt

            # B. 轮胎排放结算
            # 1. 查基础排放率
            tire_base = tire_model._get_rate(category, op_mode)
            
            # 2. 应用 EV 修正 (车重惩罚)
            tire_factor = self.cfg.MASS_FACTOR_EV if is_electric else 1.0
            tire_emission = tire_base * tire_factor * dt

            # C. 更新 Registry 统计值 (用于 Macro 表)
            if hasattr(self.registry, 'accumulate_opmode'):
                self.registry.accumulate_opmode(record, op_mode)
                self.registry.accumulate_brake_emission(record, brake_emission)
                self.registry.accumulate_tire_emission(record, tire_emission)
            else:
                # 兼容旧接口名
                self.registry.update_emission_stats(record, op_mode, brake_emission)
                self.registry.update_tire_stats(record, tire_emission)

            # D. 构造微观数据包并入库
            db_payload = {
                'type_str': final_type_str,
                'plate_color': "Resolved",
                'speed': v, 
                'accel': a, 
                'vsp': vsp,
                'op_mode': op_mode, # 存入清洗后的工况
                'brake_emission': brake_emission,
                'tire_emission': tire_emission
            }
            # 显式传入历史帧号 fid，支持离场后的乱序写入
            self.db.insert_micro(data['fid'], tid, db_payload)
            
        # 强制刷写一次 DB 缓冲区，确保本车数据立即落盘
        self.db.flush_micro_buffer()

    def _handle_ocr(self, frame, frame_id, detections):
        worker = self.comps.get('ocr_worker')
        if not worker: return

        img_h, img_w = frame.shape[:2]
        if frame_id % self.cfg.OCR_INTERVAL == 0:
            for tid, box, cid in zip(detections.tracker_id, detections.xyxy, detections.class_id):
                # 冷却时间检查
                if frame_id - self.plate_retry.get(tid, -999) < self.cfg.OCR_RETRY_COOLDOWN:
                    continue
                
                # 坐标和尺寸检查
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1+x2)/2, (y1+y2)/2
                
                # 仅对位于屏幕中心区域的车辆进行 OCR
                if not (0.1*img_w < cx < 0.9*img_w and 0.4*img_h < cy < 0.98*img_h):
                    continue
                
                if (x2-x1)*(y2-y1) > self.cfg.MIN_PLATE_AREA:
                    crop = frame[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)].copy()
                    if worker.push_task(tid, crop, cid):
                        self.plate_retry[tid] = frame_id

        # 获取并处理 OCR 结果
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
        [算法补丁] 全局轨迹优化器 v5.0 (高保真版)
        修改点：
        1. 移除 'Anchor Interpolation' (锚点插值)：彻底解决加速度与速度相位不匹配的问题。
        2. 增强位置平滑：防止大车(Bus/Truck)因检测框抖动导致的速度虚高。
        3. 采用宽窗微分：物理上更合理的加速度计算方式。
        """
        if len(trajectory) < 5: return trajectory

        # 1. 准备物理参数
        ACCEL_LIMITS = {
            self.cfg.YOLO_CLASS_CAR: 5.0,
            self.cfg.YOLO_CLASS_BUS: 2.5,
            self.cfg.YOLO_CLASS_TRUCK: 2.0
        }
        phys_limit = ACCEL_LIMITS.get(class_id, 5.0)
        dt = 1.0 / self.cfg.FPS
        
        raw_x = np.array([p['raw_x'] for p in trajectory])
        raw_y = np.array([p['raw_y'] for p in trajectory])
        n_points = len(raw_x)

        # 内部平滑函数
        def bidirectional_smooth(data, window):
            if len(data) < window: window = len(data) if len(data) % 2 == 1 else len(data) - 1
            if window < 3: return data
            pad_width = window // 2
            padded = np.pad(data, (pad_width, pad_width), mode='edge')
            kernel = np.ones(window) / window
            fwd = np.convolve(padded, kernel, mode='valid')
            padded_rev = np.pad(data[::-1], (pad_width, pad_width), mode='edge')
            bwd = np.convolve(padded_rev, kernel, mode='valid')[::-1]
            return (fwd + bwd) / 2.0

        # 2. 位置平滑 (加强版)
        # 大车(Bus)的检测框更容易抖动，需要更强的平滑来防止"虚假位移"
        pos_window = 15 if class_id == self.cfg.YOLO_CLASS_CAR else 21 
        smooth_x = bidirectional_smooth(raw_x, window=pos_window)
        smooth_y = bidirectional_smooth(raw_y, window=pos_window)
        
        # 3. 速度计算 (基于平滑后的位置)
        # 使用 np.gradient 计算微分，但再次平滑结果
        grads_x = np.gradient(smooth_x, dt)
        grads_y = np.gradient(smooth_y, dt)
        inst_speed = np.sqrt(grads_x**2 + grads_y**2)
        
        # 速度平滑：消除微分带来的高频噪声
        smooth_speed = bidirectional_smooth(inst_speed, window=15)

        # 4. [透视补偿] (保留之前的逻辑，解决拱形误差)
        path_len = np.sum(np.sqrt(np.diff(smooth_x)**2 + np.diff(smooth_y)**2))
        duration = (n_points - 1) * dt
        avg_speed = path_len / duration if duration > 0 else 0
        
        if avg_speed > 1.5 and n_points > self.cfg.FPS * 1.5: 
            EDGE_RATIO = 0.25 
            BASE_TRUST = 0.3   
            weights = np.ones(n_points)
            fade_len = int(n_points * EDGE_RATIO)
            if fade_len > 0:
                ramp = np.linspace(0, 1, fade_len)
                fade_curve = BASE_TRUST + (1 - BASE_TRUST) * np.sin(ramp * np.pi / 2)
                weights[:fade_len] = fade_curve
                weights[-fade_len:] = fade_curve[::-1]
            corrected_speed = weights * smooth_speed + (1 - weights) * avg_speed
        else:
            corrected_speed = smooth_speed

        # 5. 加速度计算 (核心修改：宽窗微分 + 移除锚点)
        # 不再使用 anchor_interpolation，直接计算物理导数
        k = 7  # 宽窗半径 (约0.25s)
        dense_accel = np.zeros(n_points)
        
        for i in range(n_points):
            idx_start = max(0, i - k)
            idx_end = min(n_points - 1, i + k)
            dv = corrected_speed[idx_end] - corrected_speed[idx_start]
            dt_span = (idx_end - idx_start) * dt
            
            if dt_span > 1e-4:
                val = dv / dt_span
                dense_accel[i] = np.clip(val, -phys_limit, phys_limit)
            else:
                dense_accel[i] = 0.0

        # [二次平滑] 这一步至关重要，它替代了锚点插值的作用，但保留了局部特征
        final_accel = bidirectional_smooth(dense_accel, window=21)

        # 6. 回写
        for i, p in enumerate(trajectory):
            p['rt_speed'] = float(inst_speed[i]) 
            p['rt_accel'] = float(dense_accel[i])
            
            p['speed'] = float(corrected_speed[i])
            p['accel'] = float(final_accel[i])

        return trajectory

    def _clean_opmode_sequence(self, opmodes):
        """
        [数据清洗] OpMode 序列优化器 (简易状态机)
        功能：基于物理约束清洗工况序列，消除不合理的突变。
        """
        if not opmodes: return []
        
        cleaned = np.array(opmodes, dtype=int)
        n = len(cleaned)
        
        # --- 策略 1: 消除短时毛刺 (Min Duration Filter) ---
        # 如果中间某帧的状态与前后都不一样，且前后状态一致，则认为是噪声
        for i in range(1, n - 1):
            prev, curr, next_ = cleaned[i-1], cleaned[i], cleaned[i+1]
            if prev == next_ and curr != prev:
                cleaned[i] = prev

        # --- 策略 2: 强制过渡约束 (State Transition Logic) ---
        # 物理事实: 刹车(0) 不能瞬间变为 急加速(37)
        for i in range(1, n):
            prev, curr = cleaned[i-1], cleaned[i]
            
            # 规则: 刹车 -> 急加速 => 降级为 缓加速(35)
            if prev == 0 and curr == 37:
                cleaned[i] = 35 
            
        return cleaned.tolist()

    def cleanup(self, final_frame_id):
        print("\n[Engine] 正在清理资源...")
        if self.comps.get('ocr_worker'):
            self.comps['ocr_worker'].stop()
        cv2.destroyAllWindows()
        print("[Engine] 保存剩余车辆数据...")
        self._handle_exits(final_frame_id + 1000)
        self.db.close()
