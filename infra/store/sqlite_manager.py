import sqlite3
import json
import os
import re # 正则表达式
from typing import List, Dict, Any

class DatabaseManager:
    """
    [基础层] SQLite 数据库管理器 (Final)
    功能：负责微观数据的批量写入和宏观数据的汇总存储。
    改进：完全剥离 SQL 语句（DDL 在 schema.sql, DML 在 queries.sql）。
    """
    def __init__(self, db_path: str = "data/traffic_data.db", fps: float = 30.0):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.fps = fps
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        
        self.cursor.execute("PRAGMA journal_mode=WAL;")
        self.cursor.execute("PRAGMA synchronous=NORMAL;")
        
        # 1. 初始化表结构 (DDL)
        self._init_schema()
        
        # 2. 加载查询模板 (DML)
        self.queries = self._load_queries()
        
        self.micro_buffer: List[tuple] = []
        self.BATCH_SIZE = 100 

    def _init_schema(self):
        """加载并执行 schema.sql"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        schema_path = os.path.join(current_dir, 'schema.sql')

        if os.path.exists(schema_path):
            try:
                with open(schema_path, 'r', encoding='utf-8') as f:
                    self.cursor.executescript(f.read())
                self.conn.commit()
            except sqlite3.Error as e:
                print(f"[Database Error] Schema 初始化失败: {e}")
        else:
            print(f"[Database Warning] Schema 文件未找到: {schema_path}")

    def _load_queries(self) -> Dict[str, str]:
        """
        [核心逻辑] 解析 queries.sql 文件
        返回格式: {'insert_micro_log': 'INSERT INTO...', ...}
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        query_path = os.path.join(current_dir, 'queries.sql')
        queries = {}

        if not os.path.exists(query_path):
            print(f"[Database Error] 查询文件未找到: {query_path}")
            return queries

        try:
            with open(query_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 使用正则匹配 "-- name: key" 块
            # 逻辑：匹配 name 行，捕获名称，然后非贪婪匹配后续内容直到下一个 name 行或文件结束
            pattern = re.compile(r'--\s*name:\s*(\w+)\s+(.+?)(?=--\s*name:|\Z)', re.DOTALL)
            
            for match in pattern.finditer(content):
                name = match.group(1).strip()
                sql = match.group(2).strip()
                # 移除行尾分号（sqlite3 execute 不需要分号，有时甚至会报错）
                if sql.endswith(';'):
                    sql = sql[:-1]
                queries[name] = sql
                
            print(f">>> [Database] 已加载 {len(queries)} 条 SQL 模板")
            
        except Exception as e:
            print(f"[Database Error] 解析 queries.sql 失败: {e}")
            
        return queries

    def insert_micro(self, frame_id: int, tid: int, data: Dict[str, Any]):
        """添加一条微观记录到缓冲区"""
        try:
            # 准备数据 Tuple (逻辑不变)
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
        """强制写入微观数据缓冲区"""
        if not self.micro_buffer: return
        
        # 使用 self.queries 字典获取 SQL
        sql = self.queries.get('insert_micro_log')
        if not sql:
            print("[Database Error] SQL模板 'insert_micro_log' 未定义")
            return
            
        try:
            self.cursor.executemany(sql, self.micro_buffer)
            self.conn.commit()
            self.micro_buffer.clear()
        except Exception as e:
            print(f"[Database Error] Batch insert failed: {e}")

    def insert_macro(self, tid: int, record: Dict[str, Any], final_type: str, final_plate: str):
        """车辆离场时，写入宏观统计数据"""
        # 使用 self.queries 字典获取 SQL
        sql = self.queries.get('insert_macro_summary')
        if not sql:
            print("[Database Error] SQL模板 'insert_macro_summary' 未定义")
            return

        try:
            # 准备数据
            life_span_frames = record['last_seen_frame'] - record['first_frame']
            duration_sec = life_span_frames / self.fps
            dist_m = record.get('total_distance_m', 0.0)
            avg_speed = (dist_m / duration_sec) if duration_sec > 0 else 0.0
            max_speed = record.get('max_speed', 0.0)
            
            dist_km = dist_m / 1000.0
            total_brake = record.get('brake_emission_mg', 0)
            total_tire = record.get('tire_emission_mg', 0)
            brake_per_km = (total_brake / dist_km) if dist_km > 0.01 else 0.0
            tire_per_km = (total_tire / dist_km) if dist_km > 0.01 else 0.0

            stats_dict = {int(k): int(v) for k, v in record.get('op_mode_stats', {}).items()}
            op_stats_json = json.dumps(stats_dict)
            
            # 执行插入
            self.cursor.execute(sql, (
                int(tid), str(final_type), str(final_plate),
                int(record['first_frame']), int(record['last_seen_frame']),
                float(round(duration_sec, 2)), float(round(dist_m, 2)),
                float(round(avg_speed, 2)), float(round(max_speed, 2)), 
                float(round(total_brake, 2)), float(round(total_tire, 2)),
                float(round(brake_per_km, 2)), float(round(tire_per_km, 2)),
                op_stats_json
            ))
            self.conn.commit()
        except Exception as e:
            print(f"[Database Error] Macro insert failed for ID {tid}: {e}")

    def close(self):
        self.flush_micro_buffer()
        self.conn.close()
        print("[Database] Connection closed.")
