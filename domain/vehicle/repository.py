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
    def __init__(self, fps: int = 30, min_survival_frames: int = 15, exit_threshold: int = 30):
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
                    'class_id': cid,            # 当前帧瞬时类别
                    'class_name': model.names[cid],
                    'class_votes': defaultdict(float), # [新增] 车型投票箱 {class_id: accumulated_weight}
                    'trajectory': [],           # [新增] 运动轨迹容器，存储 (frame_id, speed, accel)
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
                    'speed_count': 0
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

    def append_kinematics(self, tid, frame_id, speed, accel):
        """
        [新增] 记录单帧运动学数据 (不进行排放计算)
        该方法由 Engine 在每帧调用，仅做数据采集，为离场时的 VSP 回放计算做准备。
        """
        if tid in self.records:
            rec = self.records[tid]
            # 1. 存入轨迹列表
            rec['trajectory'].append({
                'frame_id': frame_id,
                'speed': speed,
                'accel': accel
            })
            
            # 2. 实时更新速度统计 (用于UI显示或简略统计)
            if speed > rec['max_speed']:
                rec['max_speed'] = speed
            rec['speed_sum'] += speed
            rec['speed_count'] += 1

    def update_emission_stats(self, tid, op_mode, emission_mass_mg, current_speed):
        """
        [保留] 更新排放统计
        注意：在新的架构下，此方法应在【车辆离场时】的 Replay 循环中被调用，
        而不是在每帧实时调用。
        """
        if tid in self.records:
            rec = self.records[tid]
            # 1. 累积时间 (帧数)
            rec['op_mode_stats'][op_mode] += 1
            
            # 2. 累积排放量 (mg)
            rec['brake_emission_mg'] += emission_mass_mg

            # 3. 更新速度统计
            # (注意：此逻辑其实在 append_kinematics 里也有一份，
            # 但为了 Macro 表的完整性，这里保留无害，或者可以删去速度部分仅保留排放)
            if current_speed > rec['max_speed']:
                rec['max_speed'] = current_speed
            rec['speed_sum'] += current_speed
            rec['speed_count'] += 1

    def update_tire_stats(self, tid, pm10_mg):
        """
        [保留] 更新轮胎排放统计
        同上，应在离场 Replay 时调用。
        """
        if tid in self.records:
            self.records[tid]['tire_emission_mg'] += pm10_mg

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
        检查并返回刚离场的车辆列表
        在此处使用 min_survival_frames 过滤噪点
        :return: list of (tid, record)
        """
        exited_vehicles = []
        for tid, record in self.records.items():
            # 检查是否满足“消失太久”的离场条件
            if not record['reported'] and (frame_id - record['last_seen_frame'] > self.exit_threshold):
                # 无论是否是噪点，都标记为 reported，避免下次重复检查
                record['reported'] = True
                
                # 计算车辆生命周期
                life_span = record['last_seen_frame'] - record['first_frame']
                
                # 只有存活够久的车辆才会被返回给 Engine 进行数据库记录
                if life_span >= self.min_survival_frames:
                    exited_vehicles.append((tid, record))
                # else:
                #     噪点车辆被静默忽略
                    
        return exited_vehicles

    def get_history(self, tid):
        """获取车牌识别历史"""
        return self.records.get(tid, {}).get('plate_history', [])
    
    def get_record(self, tid):
        """获取车辆完整档案"""
        return self.records.get(tid)
