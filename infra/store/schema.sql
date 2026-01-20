-- infra/store/schema.sql

-- ==========================================
-- 1. 微观数据表 (Microscopic Data)
-- 记录每一帧的瞬时物理状态和排放计算结果
-- ==========================================
CREATE TABLE IF NOT EXISTS micro_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id INTEGER NOT NULL,
    track_id INTEGER NOT NULL,
    vehicle_type TEXT,
    plate_color TEXT,
    speed REAL,           -- m/s
    accel REAL,           -- m/s^2
    vsp REAL,             -- kW/tonne
    op_mode INTEGER,      -- EPA MOVES OpMode ID
    brake_emission REAL,  -- mg
    tire_emission REAL,   -- mg
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 索引优化查询性能
CREATE INDEX IF NOT EXISTS idx_frame ON micro_logs (frame_id);
CREATE INDEX IF NOT EXISTS idx_track ON micro_logs (track_id);

-- ==========================================
-- 2. 宏观汇总表 (Macroscopic Summary)
-- 记录车辆离场后的全生命周期统计数据
-- ==========================================
CREATE TABLE IF NOT EXISTS macro_summary (
    track_id INTEGER PRIMARY KEY,
    vehicle_type TEXT,
    plate_text TEXT,
    first_frame INTEGER,
    last_frame INTEGER,
    duration_sec REAL,
    total_distance_m REAL,
    avg_speed REAL,       -- m/s
    max_speed REAL,       -- m/s
    total_brake_mg REAL,
    total_tire_mg REAL,
    brake_mg_per_km REAL, -- mg/km
    tire_mg_per_km REAL,  -- mg/km
    op_mode_stats JSON,   -- JSON string: {op_mode: count}
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
