import config.settings as cfg
from collections import defaultdict

class VehicleRegistry:
    """
    [核心组件] 车辆注册表
    功能：管理所有车辆的生命周期（注册、更新、历史记录、离场判定）。
    """
    def __init__(self):
        # 核心数据库: {tid: {info...}}
        self.records = {}
        # 离场阈值
        self.exit_threshold = 30 

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
                    'op_mode_stats': defaultdict(int), # {OpMode: frames}
                    'total_emission_mg': 0.0           # 累积排放总量
                }
            else:
                # 老车更新
                self.records[tid]['last_seen_frame'] = frame_id
                # 始终保留最高的置信度对应的分类结果
                if conf > self.records[tid]['max_conf']:
                    self.records[tid]['max_conf'] = float(conf)
                    self.records[tid]['class_id'] = cid
                    self.records[tid]['class_name'] = model.names[cid]

    def update_emission_stats(self, tid, op_mode, emission_rate_mg_s):
        if tid in self.records:
            # 1. 累积时间 (帧数)
            self.records[tid]['op_mode_stats'][op_mode] += 1
            
            # 2. 累积排放量 (mg)
            # emission_rate 单位是 mg/s，当前是 1 帧，时间为 1/FPS 秒
            emission_per_frame = emission_rate_mg_s * (1.0 / cfg.FPS)
            self.records[tid]['total_emission_mg'] += emission_per_frame

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
        :return: list of (tid, record)
        """
        exited_vehicles = []
        for tid, record in self.records.items():
            # 如果尚未报告过，且超时未出现
            if not record['reported'] and (frame_id - record['last_seen_frame'] > self.exit_threshold):
                record['reported'] = True
                exited_vehicles.append((tid, record))
        return exited_vehicles

    def get_history(self, tid):
        """获取指定车辆的车牌历史"""
        return self.records.get(tid, {}).get('plate_history', [])
    
    def get_record(self, tid):
        return self.records.get(tid)
