import sys
import cv2
import supervision as sv
from ultralytics import YOLO

# 导入配置和组件
from app.monitor_engine import TrafficMonitorEngine
from domain.vehicle.repository import VehicleRegistry
from domain.physics.vsp_calculator import VSPCalculator
from domain.physics.opmode_calculator import MovesOpModeCalculator # [新增导入]
from domain.physics.brake_emission_model import BrakeEmissionModel
from domain.physics.tire_emission_model import TireEmissionModel
from domain.vehicle.classifier import VehicleClassifier
import infra.config.loader as cfg
from infra.concurrency.ocr_worker import AsyncOCRManager
from infra.store.sqlite_manager import DatabaseManager
from infra.sys.process_optimizer import SystemOptimizer
from perception.math.geometry import ViewTransformer
from perception.kinematics_estimator import KinematicsEstimator
from ui.renderer import Visualizer
from ui.calibration_window import CalibrationUI
from ui.console_reporter import Reporter

def main():
    # 1. 系统初始化
    SystemOptimizer.set_cpu_affinity("main")
    print(f"\n>>> [System] Initializing Traffic Monitor...", flush=True)

    try:
        # 2. 交互式标定
        calibrator = CalibrationUI(cfg.VIDEO_PATH)
        source_points, target_points = calibrator.run()
        print(f">>> [System] 标定完成", flush=True)

        # 3. 组装组件 (Dependency Injection)
        print(f">>> [System] 正在组装组件...", flush=True)
        
        # 构建报告模块配置
        reporter_config = {
            "debug_mode": cfg.DEBUG_MODE,
            "fps": cfg.FPS,
            "min_survival_frames": cfg.MIN_SURVIVAL_FRAMES
        }

        # 构建车辆分类器配置
        classifier_config = {
            "car": cfg.YOLO_CLASS_CAR,
            "bus": cfg.YOLO_CLASS_BUS,
            "truck": cfg.YOLO_CLASS_TRUCK
        }

        # 构建 VSP 计算器配置
        vsp_config = {
            "vsp_coefficients": cfg.VSP_COEFFS,
            "road_grade_percent": cfg.ROAD_GRADE_PERCENT
        }

        # [新增] 实例化共享的 OpMode 计算器 (策略对象)
        # 这确保了轮胎模型和可能的其他模型使用完全一致的工况判定逻辑
        opmode_calculator = MovesOpModeCalculator(config=cfg._e)

        # 构建刹车磨损排放模型配置
        brake_emission_config = {
            "braking_decel_threshold": cfg.BRAKING_DECEL_THRESHOLD,
            "idling_speed_threshold": cfg.IDLING_SPEED_THRESHOLD,
            "low_speed_threshold": cfg.LOW_SPEED_THRESHOLD,
            "mass_factor_ev": cfg.MASS_FACTOR_EV,
            "moves_brake_wear_rates": cfg.MOVES_BRAKE_WEAR_RATES
        }

        # [修改] 构建轮胎磨损排放模型配置
        tire_emission_config = {
            "tire_wear_rates": cfg.TIRE_WEAR_RATES, # 使用新加载的费率表
            "emission_params": cfg._e
        }
        
        # 构建运动学配置
        kinematics_config = {
            "fps": cfg.FPS,
            "kinematics": {
                "speed_window": cfg.SPEED_WINDOW,  # 分离的窗口参数
                "accel_window": cfg.ACCEL_WINDOW,
                "border_margin": cfg.BORDER_MARGIN,
                "min_tracking_frames": cfg.MIN_TRACKING_FRAMES,
                "max_physical_accel": cfg.MAX_PHYSICAL_ACCEL,
                "poly_order": cfg.KINEMATICS_POLY_ORDER
            }
        }

        # 加载所有默认组件
        components = {
            'model': YOLO("model/yolov8n.pt"),
            'tracker': sv.ByteTrack(frame_rate=cfg.FPS),
            'smoother': sv.DetectionsSmoother(length=3),
            'transformer': ViewTransformer(source_points, target_points),
            'visualizer': Visualizer(calibration_points=source_points, trace_length=cfg.FPS),
            'registry': VehicleRegistry(
                fps=cfg.FPS,
                min_survival_frames=cfg.MIN_SURVIVAL_FRAMES,
                exit_threshold=cfg.EXIT_THRESHOLD
            ), 
            'db': DatabaseManager(db_path=cfg.DB_PATH, fps=cfg.FPS),
            'classifier': VehicleClassifier(type_map=cfg.TYPE_MAP, yolo_classes=classifier_config)
        }

        # 按需加载可选组件
        if cfg.DEBUG_MODE:
            components['reporter'] = Reporter(config=reporter_config)

        if cfg.ENABLE_MOTION:
            components['kinematics'] = KinematicsEstimator(config=kinematics_config)
        
        if cfg.ENABLE_EMISSION and cfg.ENABLE_MOTION:
            components['vsp_calculator'] = VSPCalculator(config=vsp_config)
            components['brake_model'] = BrakeEmissionModel(config=brake_emission_config)
            
            # [关键修改] 注入 tire_emission_config 和 opmode_calculator
            components['tire_model'] = TireEmissionModel(
                config=tire_emission_config,
                opmode_calculator=opmode_calculator
            )
            
        if cfg.ENABLE_OCR:
            print(f">>> [System] 启动 OCR 进程...", flush=True)
            ocr_worker = AsyncOCRManager()
            ocr_worker.start()
            components['ocr_worker'] = ocr_worker

        # 4. 启动引擎
        print(f">>> [System] 初始化引擎 (TrafficMonitorEngine)...", flush=True)
        engine = TrafficMonitorEngine(config=cfg, components=components)
        
        print(f">>> [System] 进入主循环 (Engine.run)...", flush=True)
        engine.run()

    except KeyboardInterrupt:
        print("\n[System] 用户中断，停止运行。", flush=True)
        if 'engine' in locals(): engine.cleanup(0)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[System] 发生严重错误: {e}", flush=True)
        
        if 'ocr_worker' in locals() and hasattr(ocr_worker, 'stop'):
            print(">>> [System] 正在强制关闭 OCR 进程...", flush=True)
            ocr_worker.stop()
        
        if 'engine' in locals(): 
            engine.cleanup(0)

if __name__ == "__main__":
    main()
