import sys
import os
import cv2
from ultralytics import YOLO
import supervision as sv

from app.monitor_engine import TrafficMonitorEngine
from domain.vehicle.repository import VehicleRegistry
from domain.physics.vsp_calculator import VSPCalculator
from domain.physics.opmode_calculator import MovesOpModeCalculator
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
    SystemOptimizer.set_cpu_affinity("main")
    print(f"\n>>> [System] Initializing Traffic Monitor...", flush=True)

    try:
        calibrator = CalibrationUI(cfg.VIDEO_PATH)
        source_points, target_points = calibrator.run()
        print(f">>> [System] 标定完成", flush=True)

        print(f">>> [System] 正在组装组件...", flush=True)
        
        reporter_config = {
            "debug_mode": cfg.DEBUG_MODE,
            "fps": cfg.FPS,
            "min_survival_frames": cfg.MIN_SURVIVAL_FRAMES
        }

        classifier_config = {
            "car": cfg.YOLO_CLASS_CAR,
            "bus": cfg.YOLO_CLASS_BUS,
            "truck": cfg.YOLO_CLASS_TRUCK
        }

        vsp_config = {
            "vsp_coefficients": cfg.VSP_COEFFS,
            "road_grade_percent": cfg.ROAD_GRADE_PERCENT
        }
        
        # 共享 OpMode 计算器
        opmode_calculator = MovesOpModeCalculator(config=cfg._e)

        # 刹车模型配置
        brake_emission_config = {
            "braking_decel_threshold": cfg.BRAKING_DECEL_THRESHOLD,
            "idling_speed_threshold": cfg.IDLING_SPEED_THRESHOLD,
            "low_speed_threshold": cfg.LOW_SPEED_THRESHOLD,
            "mass_factor_ev": cfg.MASS_FACTOR_EV,
            "brake_wear_coefficients": cfg.BRAKE_WEAR_COEFFICIENTS 
        }

        # 轮胎模型配置
        tire_emission_config = {
            "tire_wear_coefficients": cfg.TIRE_WEAR_COEFFICIENTS,
            "emission_params": cfg._e
        }
        
        kinematics_config = {
            "fps": cfg.FPS,
            "kinematics": {
                "speed_window": cfg.SPEED_WINDOW,
                "accel_window": cfg.ACCEL_WINDOW,
                "border_margin": cfg.BORDER_MARGIN,
                "min_tracking_frames": cfg.MIN_TRACKING_FRAMES,
                "max_physical_accel": cfg.MAX_PHYSICAL_ACCEL,
                "poly_order": cfg.KINEMATICS_POLY_ORDER
            }
        }

        components = {
            'model': YOLO("model/yolov8n.pt"),
            'tracker': sv.ByteTrack(frame_rate=cfg.FPS),
            'smoother': sv.DetectionsSmoother(length=3),
            'transformer': ViewTransformer(source_points, target_points),
            'visualizer': Visualizer(
                calibration_points=source_points,
                trace_length=cfg.FPS,
                opmode_calculator=opmode_calculator
            ),
            'registry': VehicleRegistry(
                fps=cfg.FPS,
                min_survival_frames=cfg.MIN_SURVIVAL_FRAMES,
                exit_threshold=cfg.EXIT_THRESHOLD,
                min_valid_pts=cfg.MIN_VALID_POINTS,
                min_moving_dist=cfg.MIN_MOVING_DIST
            ),
            'db': DatabaseManager(cfg.DB_PATH, cfg.FPS),
            'classifier': VehicleClassifier(cfg.TYPE_MAP, classifier_config)
        }

        if cfg.DEBUG_MODE:
            components['reporter'] = Reporter(reporter_config, opmode_calculator)

        if cfg.ENABLE_MOTION:
            components['kinematics'] = KinematicsEstimator(kinematics_config)
        
        if cfg.ENABLE_EMISSION and cfg.ENABLE_MOTION:
            components['vsp_calculator'] = VSPCalculator(vsp_config)
            components['brake_model'] = BrakeEmissionModel(brake_emission_config)
            components['tire_model'] = TireEmissionModel(tire_emission_config, opmode_calculator)
            
        if cfg.ENABLE_OCR:
            ocr_worker = AsyncOCRManager()
            ocr_worker.start()
            components['ocr_worker'] = ocr_worker

        engine = TrafficMonitorEngine(cfg, components)
        engine.run()

    except KeyboardInterrupt:
        if 'engine' in locals(): engine.cleanup(0)
    except Exception as e:
        import traceback
        traceback.print_exc()
        if 'ocr_worker' in locals() and hasattr(ocr_worker, 'stop'):
            ocr_worker.stop()
        if 'engine' in locals(): engine.cleanup(0)

if __name__ == "__main__":
    # import multiprocessing
    # multiprocessing.freeze_support()

    main()
