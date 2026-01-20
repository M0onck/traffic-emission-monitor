-- infra/store/queries.sql

-- ==========================================
-- 具名查询定义文件
-- 格式说明:
-- -- name: 查询名称
-- SQL 语句...
-- ==========================================

-- name: insert_micro_log
INSERT INTO micro_logs (
    frame_id, track_id, vehicle_type, plate_color, 
    speed, accel, vsp, op_mode, brake_emission, tire_emission
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);

-- name: insert_macro_summary
INSERT OR REPLACE INTO macro_summary (
    track_id, vehicle_type, plate_text, 
    first_frame, last_frame, duration_sec, 
    total_distance_m, avg_speed, max_speed, 
    total_brake_mg, total_tire_mg, 
    brake_mg_per_km, tire_mg_per_km,
    op_mode_stats
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
