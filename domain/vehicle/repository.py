import numpy as np
from collections import defaultdict

class VehicleRegistry:
    """
    [业务层] 车辆注册表
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
        """
        for tid, cid, conf in zip(detections.tracker_id, detections.class_id, detections.confidence):
            if tid not in self.records:
                # 新车注册
                self.records[tid] = {
                    'class_id': cid, 
                    'class_name': model.names[cid],
                    'first_frame': frame_id,
                    'max_conf': float(conf),
                    'last_seen_frame': frame_id,
                    'reported': False,
                    'plate_history': [],
                    'op_mode_stats': defaultdict(int),
                    'total_emission_mg': 0.0,
                    'max_speed': 0.0,
                    'speed_sum': 0.0,
                    'speed_count': 0
                }
            else:
                # 老车更新
                self.records[tid]['last_seen_frame'] = frame_id
                if conf > self.records[tid]['max_conf']:
                    self.records[tid]['max_conf'] = float(conf)
                    self.records[tid]['class_id'] = cid
                    self.records[tid]['class_name'] = model.names[cid]

    def update_emission_stats(self, tid, op_mode, emission_rate_mg_s, current_speed):
        if tid in self.records:
            rec = self.records[tid]
            # 1. 累积时间 (帧数)
            rec['op_mode_stats'][op_mode] += 1
            
            # 2. 累积排放量 (mg)
            # 使用 self.fps 计算单帧时间
            # emission_rate 单位是 mg/s，当前是 1 帧，时间为 1/FPS 秒
            emission_per_frame = emission_rate_mg_s * (1.0 / self.fps)
            rec['total_emission_mg'] += emission_per_frame

            # 3. 更新速度统计
            if current_speed > rec['max_speed']:
                rec['max_speed'] = current_speed
            rec['speed_sum'] += current_speed
            rec['speed_count'] += 1

    def add_plate_history(self, tid, color, area, conf):
        """记录一次有效的车牌识别结果"""
        if tid in self.records:
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
        return self.records.get(tid, {}).get('plate_history', [])
    
    def get_record(self, tid):
        return self.records.get(tid)
