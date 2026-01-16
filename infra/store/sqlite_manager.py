import sqlite3
import json
import os
from typing import List, Dict, Any

class DatabaseManager:
    """
    [基础层] SQLite 数据库管理器
    功能：负责微观数据的批量写入和宏观数据的汇总存储。
    
    [适配说明 - On-Exit Replay Architecture]
    为配合“离场结算”模式，本管理器支持：
    1. insert_micro: 接收历史 frame_id (非当前流式帧号)，支持乱序写入 (按车辆聚类)。
    2. flush_micro_buffer: 支持外部显式调用，确保单车轨迹计算完成后立即落盘。
    """
    def __init__(self, db_path: str = "data/traffic_data.db", fps: float = 30.0):
        # 确保目录存在
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.fps = fps
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        
        # 开启 WAL 模式 (Write-Ahead Logging) 以提高并发写入性能
        self.cursor.execute("PRAGMA journal_mode=WAL;")
        self.cursor.execute("PRAGMA synchronous=NORMAL;")
        
        # 初始化表结构
        self._init_tables()
        
        # 微观数据缓冲区 (用于批量写入)
        self.micro_buffer: List[tuple] = []
        self.BATCH_SIZE = 100 

    def _init_tables(self):
        """初始化数据库表结构"""
        
        # 1. 微观表 (Microscopic): 记录每帧的瞬时状态
        # 注意: 这里的 timestamp 默认为插入时间。在离场结算模式下，
        # 它代表"结算时间"而非"物理发生时间"。物理时间请参考 frame_id。
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS micro_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id INTEGER,
                track_id INTEGER,
                vehicle_type TEXT,
                plate_color TEXT,
                speed REAL,
                accel REAL,
                vsp REAL,
                op_mode INTEGER,
                brake_emission REAL,
                tire_emission REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # 创建索引加速查询 (按帧号或车辆ID查询轨迹)
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_frame ON micro_logs (frame_id);")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_track ON micro_logs (track_id);")

        # 2. 宏观表 (Macroscopic): 记录车辆离场后的汇总信息
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS macro_summary (
                track_id INTEGER PRIMARY KEY,
                vehicle_type TEXT,
                plate_text TEXT,
                first_frame INTEGER,
                last_frame INTEGER,
                duration_sec REAL,
                total_distance_m REAL,
                avg_speed REAL,
                max_speed REAL,
                total_brake_mg REAL,
                total_tire_mg REAL,
                brake_mg_per_km REAL,
                tire_mg_per_km REAL,
                op_mode_stats JSON,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def insert_micro(self, frame_id: int, tid: int, data: Dict[str, Any]):
        """
        添加一条微观记录到缓冲区
        :param frame_id: 必须显式传入。在回放模式下，这是历史轨迹中的帧号。
        :param tid: 车辆追踪ID
        :param data: 包含物理计算结果的字典
        """
        # [修改] 增加 get() 和 默认值处理，防止计算层偶发的 None 导致崩溃
        # 强制转换为原生类型，避免 numpy 类型导致的数据库错误
        try:
            row = (
                int(frame_id),
                int(tid),
                str(data.get('type_str', 'Unknown')),
                str(data.get('plate_color', 'Unknown')),
                float(round(data.get('speed', 0.0), 2)),
                float(round(data.get('accel', 0.0), 2)),
                float(round(data.get('vsp', 0.0), 2)),
                int(data.get('op_mode', -1)),
                float(round(data.get('brake_emission', 0.0), 4)),
                float(round(data.get('tire_emission', 0.0), 4))
            )
            self.micro_buffer.append(row)
            
            if len(self.micro_buffer) >= self.BATCH_SIZE:
                self.flush_micro_buffer()
                
        except Exception as e:
            print(f"[Database Warning] Failed to prepare micro log row: {e}")

    def flush_micro_buffer(self):
        """
        强制写入微观数据缓冲区
        [使用场景] 在 Engine 中完成一辆车的全轨迹计算后调用，确保数据即时落盘。
        """
        if not self.micro_buffer:
            return
            
        try:
            # SQL 插入语句
            self.cursor.executemany("""
                INSERT INTO micro_logs (
                    frame_id, track_id, vehicle_type, plate_color, 
                    speed, accel, vsp, op_mode, brake_emission, tire_emission
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, self.micro_buffer)
            self.conn.commit()
            self.micro_buffer.clear()
        except Exception as e:
            print(f"[Database Error] Micro batch insert failed: {e}")

    def insert_macro(self, tid: int, record: Dict[str, Any], final_type: str, final_plate: str):
        """
        车辆离场时，写入宏观统计数据
        """
        try:
            life_span_frames = record['last_seen_frame'] - record['first_frame']
            duration_sec = life_span_frames / self.fps
            
            # [新增] 获取里程与计算平均速度
            dist_m = record.get('total_distance_m', 0.0)
            
            # 平均速度计算 (优先使用 distance / time，避免 sum/count 的精度累积误差)
            # 防止除以零
            avg_speed = (dist_m / duration_sec) if duration_sec > 0 else 0.0
            max_speed = record.get('max_speed', 0.0)

            # [新增] 计算单位排放 (mg/km)
            # distance (m) -> km: dist_m / 1000
            dist_km = dist_m / 1000.0
            
            total_brake = record.get('brake_emission_mg', 0)
            total_tire = record.get('tire_emission_mg', 0)
            
            # 避免极短距离导致的除以零 (例如 < 10米 算作无效样本)
            brake_per_km = (total_brake / dist_km) if dist_km > 0.01 else 0.0
            tire_per_km = (total_tire / dist_km) if dist_km > 0.01 else 0.0

            stats_dict = {int(k): int(v) for k, v in record.get('op_mode_stats', {}).items()}
            op_stats_json = json.dumps(stats_dict)
            
            # SQL 插入语句
            self.cursor.execute("""
                INSERT OR REPLACE INTO macro_summary (
                    track_id, vehicle_type, plate_text, 
                    first_frame, last_frame, duration_sec, 
                    total_distance_m, avg_speed, max_speed, 
                    total_brake_mg, total_tire_mg, 
                    brake_mg_per_km, tire_mg_per_km,
                    op_mode_stats
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                int(tid),
                str(final_type),
                str(final_plate),
                int(record['first_frame']),
                int(record['last_seen_frame']),
                float(round(duration_sec, 2)),
                float(round(dist_m, 2)),         # [新增]
                float(round(avg_speed, 2)), 
                float(round(max_speed, 2)), 
                float(round(total_brake, 2)), 
                float(round(total_tire, 2)),
                float(round(brake_per_km, 2)),   # [新增]
                float(round(tire_per_km, 2)),    # [新增]
                op_stats_json
            ))
            self.conn.commit()
        except Exception as e:
            print(f"[Database Error] Macro insert failed for ID {tid}: {e}")

    def close(self):
        """关闭连接前确保缓冲区已写入"""
        self.flush_micro_buffer()
        self.conn.close()
        print("[Database] Connection closed.")
