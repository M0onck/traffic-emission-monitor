import sqlite3
import json
import os
from typing import List, Dict, Any

class DatabaseManager:
    """
    [基础层] SQLite 数据库管理器
    功能：负责微观数据的批量写入和宏观数据的汇总存储。
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
                emission_rate REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # 创建索引加速查询
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
                avg_speed REAL,
                max_speed REAL,
                total_emission_mg REAL,
                op_mode_stats JSON,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def insert_micro(self, frame_id: int, tid: int, data: Dict[str, Any]):
        """
        添加一条微观记录到缓冲区
        """
        # 强制转换为原生类型，避免 numpy 类型导致的数据库错误
        row = (
            int(frame_id),
            int(tid),
            str(data['type_str']),
            str(data['plate_color']),
            float(round(data['speed'], 2)),
            float(round(data['accel'], 2)),
            float(round(data['vsp'], 2)),
            int(data['op_mode']),
            float(round(data['emission_rate'], 4))
        )
        self.micro_buffer.append(row)
        
        if len(self.micro_buffer) >= self.BATCH_SIZE:
            self.flush_micro_buffer()

    def flush_micro_buffer(self):
        """强制写入微观数据缓冲区"""
        if not self.micro_buffer:
            return
            
        try:
            self.cursor.executemany("""
                INSERT INTO micro_logs (
                    frame_id, track_id, vehicle_type, plate_color, 
                    speed, accel, vsp, op_mode, emission_rate
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            # 计算统计量
            life_span_frames = record['last_seen_frame'] - record['first_frame']
            duration_sec = life_span_frames / self.fps
            
            # 计算速度统计
            max_speed = record.get('max_speed', 0.0)
            avg_speed = 0.0
            count = record.get('speed_count', 0)
            if count > 0:
                avg_speed = record.get('speed_sum', 0.0) / count

            # OpMode 统计转 JSON 字符串
            # 确保 keys/values 都是原生类型
            stats_dict = {int(k): int(v) for k, v in record.get('op_mode_stats', {}).items()}
            op_stats_json = json.dumps(stats_dict)
            
            self.cursor.execute("""
                INSERT OR REPLACE INTO macro_summary (
                    track_id, vehicle_type, plate_text, 
                    first_frame, last_frame, duration_sec, 
                    avg_speed, max_speed, total_emission_mg, op_mode_stats
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                int(tid),
                str(final_type),
                str(final_plate),
                int(record['first_frame']),
                int(record['last_seen_frame']),
                float(round(duration_sec, 2)),
                float(round(avg_speed, 2)), 
                float(round(max_speed, 2)), 
                float(round(record.get('total_emission_mg', 0), 2)),
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
