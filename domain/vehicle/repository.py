import numpy as np
import math  # [新增] 用于计算面积权重的开根号
from collections import defaultdict

class VehicleRegistry:
    """
    [业务层] 车辆注册表
    职责：
    1. 维护车辆的全生命周期档案 (Records)。
    2. 实现基于多帧观测的车型投票机制，消除检测抖动。
    3. 存储车辆的运动轨迹 (Trajectory)，支持离场后的物理回放计算。
    """
    def __init__(self, fps: int = 30, min_survival_frames: int = 15, exit_threshold: int = 30,
                 min_valid_pts: int = 15, min_moving_dist: float = 2.0):
        """
        :param fps: 系统帧率，用于计算时间步长
        :param min_survival_frames: 最小存活帧数，低于此值的轨迹被视为噪点，离场时不报告
        :param exit_threshold: 离场判定阈值 (多少帧不见后判定为离开)
        """
        self.records = {}
        # 将配置参数保存为实例变量
        self.fps = fps
        self.min_survival_frames = min_survival_frames
        self.exit_threshold = exit_threshold
        self.min_valid_pts = min_valid_pts
        self.min_moving_dist = min_moving_dist

    def update(self, detections, frame_id, model):
        """
        根据当前帧检测结果更新车辆档案
        [修改] 修复了zip解包错误，并引入了基于面积权重的车型投票逻辑
        """
        # [关键修复] zip 中补全了 detections.xyxy，否则会报 ValueError
        for tid, cid, conf, box in zip(
                detections.tracker_id, 
                detections.class_id, 
                detections.confidence,
                detections.xyxy
            ):
            cid = int(cid)
            conf = float(conf)
            
            # --- 加权投票逻辑 ---
            # 计算权重: W = Conf * sqrt(Area)
            # 理由：避免极近距离的大框误检（Area平方级增长）主导结果，同时保留距离近权重大的物理特性。
            width = box[2] - box[0]
            height = box[3] - box[1]
            area = width * height
            weight = conf * math.sqrt(area) 

            if tid not in self.records:
                # 新车注册
                self.records[tid] = {
                    'class_id': cid,                   # 当前帧瞬时类别
                    'class_name': model.names[cid],
                    'class_votes': defaultdict(float), # 车型投票箱 {class_id: accumulated_weight}
                    'trajectory': [],                  # 运动轨迹容器，存储 (frame_id, speed, accel)
                    'valid_samples_count': 0,          # 有效轨迹样本计数器
                    'first_frame': frame_id,
                    'max_conf': float(conf),
                    'last_seen_frame': frame_id,
                    'reported': False,
                    'plate_history': [],
                    
                    # 统计数据 (将在离场回放时或实时更新)
                    'op_mode_stats': defaultdict(int),
                    'brake_emission_mg': 0.0,
                    'tire_emission_mg': 0.0,
                    'max_speed': 0.0,
                    'speed_sum': 0.0,
                    'speed_count': 0,
                    'total_distance_m': 0.0
                }
            
            # 老车更新
            rec = self.records[tid]
            rec['last_seen_frame'] = frame_id
            
            # [修改] 累加车型权重
            rec['class_votes'][cid] += weight
            
            # [修改] 实时更新最优车型 (不再仅依赖 max_conf)
            # 只有当累积权重超过当前记录的类别时才切换，这比单帧判定更稳
            best_class = max(rec['class_votes'], key=rec['class_votes'].get)
            rec['class_id'] = best_class
            rec['class_name'] = model.names[best_class]

            # 仍然保留 max_conf 记录用于参考
            if conf > rec['max_conf']:
                rec['max_conf'] = conf

    def append_kinematics(self, tid, frame_id, speed, accel, raw_x=None, raw_y=None):
        """
        记录单帧运动学数据 (不进行排放计算)
        该方法由 Engine 在每帧调用，仅做数据采集，为离场时的 VSP 回放计算做准备。
        """
        if tid in self.records:
            rec = self.records[tid]
            # 存入轨迹列表
            rec['trajectory'].append({
                'frame_id': frame_id,
                'speed': speed,
                'accel': accel,
                'raw_x': raw_x,
                'raw_y': raw_y
            })
            # 累加有效样本计数
            rec['valid_samples_count'] = rec.get('valid_samples_count', 0) + 1

            # 累加行驶里程
            # Distance = Speed * dt (dt = 1/FPS)
            dt = 1.0 / self.fps
            rec['total_distance_m'] = rec.get('total_distance_m', 0.0) + (speed * dt)
            
            # 实时更新速度统计 (用于UI显示或简略统计)
            if speed > rec['max_speed']:
                rec['max_speed'] = speed
            rec['speed_sum'] += speed
            rec['speed_count'] += 1

    def accumulate_opmode(self, record, op_mode: int):
        """
        累加工况统计 (OpMode Histogram)
        """
        record['op_mode_stats'][op_mode] += 1

    def accumulate_brake_emission(self, record, mass_mg: float):
        """
        累加刹车排放量
        """
        record['brake_emission_mg'] += mass_mg

    def accumulate_tire_emission(self, record, mass_mg: float):
        """
        累加轮胎排放量
        """
        record['tire_emission_mg'] += mass_mg

    def add_plate_history(self, tid, color, area, conf):
        """记录一次有效的车牌识别结果"""
        if tid in self.records:
            # 原始数据保留，权重计算留给 classifier 在离场时处理
            self.records[tid]['plate_history'].append({
                'color': color,
                'area': area,
                'conf': conf
            })

    def check_exits(self, frame_id):
        """
        检查并移除已离开画面的车辆，返回符合质量标准的有效车辆记录。
        
        筛选策略 (Data Quality Gate):
        为了确保入库数据的物理有效性，采用以下“三级门控”筛选机制：
        1. 时间门控 (Survival): 生命周期 > min_survival_frames (过滤瞬间闪烁的噪点)
        2. 质量门控 (Quality): 有效运动学样本数 > min_valid_trajectory_points (过滤仅在ROI边缘短暂路过的车辆)
        3. 空间门控 (Movement): 累计位移 > min_moving_distance_m (过滤路边静止车辆或背景误检)
        
        重要说明：
        凡是判定为离场（超时未更新）的车辆，无论是否通过筛选，均会从内存(self.records)中彻底清除，
        以防止长时间运行下的内存泄漏。但只有通过筛选的车辆会被返回给上层进行数据库写入。
        
        :param frame_id: 当前帧号
        :return: List[Tuple(tid, record)] 有效的离场车辆列表
        """
        # 1. 识别所有超时未更新的车辆 (判定为已离场)
        # 注意：这里不能直接在遍历时删除字典元素，需先收集 ID
        timed_out_ids = []
        for tid, record in self.records.items():
            if frame_id - record['last_seen_frame'] > self.exit_threshold:
                timed_out_ids.append(tid)

        valid_exits = []
        
        # 2. 逐一进行质量筛选并清理内存
        for tid in timed_out_ids:
            record = self.records[tid]
            
            # --- 门控 1: 存活时间 (原有逻辑) ---
            # 过滤掉存在时间极短的误检
            life_span = record['last_seen_frame'] - record['first_frame']
            has_survival = life_span >= self.min_survival_frames
            
            # --- 门控 2: 数据质量 (新增) ---
            # 需确保车辆在 ROI 内且滤波器已收敛的帧数足够多 (建议值: 15帧)
            # 防止只在标定框边缘蹭了一下的车辆入库，这类车辆的平均速度计算极不准确
            min_valid_pts = getattr(self, 'min_valid_trajectory_points', 15)
            valid_samples = record.get('valid_samples_count', 0)
            has_quality = valid_samples >= min_valid_pts
            
            # --- 门控 3: 移动距离 (新增) ---
            # 过滤掉虽然一直被追踪但实际上没动过的目标 (如树影、停止的车辆)
            # 建议阈值: 2.0米
            min_dist = getattr(self, 'min_moving_distance_m', 2.0)
            total_dist = record.get('total_distance_m', 0.0)
            has_movement = total_dist >= min_dist
            
            # 综合判定：必须同时通过三个门控
            if has_survival and has_quality and has_movement:
                valid_exits.append((tid, record))
            # 调试日志：如果被丢弃，且在调试模式下，打印原因 (可选)
            # else:
            #     print(f"[Discard] ID:{tid} Life:{life_span} ValidPts:{valid_samples} Dist:{total_dist:.1f}")

            # 3. 内存清理 (Critical Fix)
            # 无论车辆是否有效，一旦离场，必须从内存中删除，否则会造成内存泄漏
            del self.records[tid]
            
        return valid_exits

    def get_history(self, tid):
        """获取车牌识别历史"""
        return self.records.get(tid, {}).get('plate_history', [])
    
    def get_record(self, tid):
        """获取车辆完整档案"""
        return self.records.get(tid)
