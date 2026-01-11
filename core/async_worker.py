# core/async_worker.py
import multiprocessing
import queue
import time
import numpy as np
# 注意：这里只导入类定义，不要在全局实例化
from perception.plate_reader import LicensePlateRecognizer
from utils.optimization import SystemOptimizer

class AsyncOCRManager:
    def __init__(self):
        # 使用 multiprocessing 的 Queue
        # maxsize=3: 稍微减小缓冲，保证只处理最新的，宁缺毋滥
        self.task_queue = multiprocessing.Queue(maxsize=3)
        self.result_queue = multiprocessing.Queue()
        
        self.worker_process = None

    def start(self):
        """启动后台子进程"""
        # target 指向 _worker_entry_point 函数
        self.worker_process = multiprocessing.Process(
            target=self._worker_entry_point,
            args=(self.task_queue, self.result_queue),
            daemon=True
        )
        self.worker_process.start()
        print(f">>> [System] OCR 独立进程已启动 (PID: {self.worker_process.pid})")

    def stop(self):
        """停止进程"""
        if self.worker_process and self.worker_process.is_alive():
            self.worker_process.terminate() # 强制结束
            self.worker_process.join()
        print(">>> [System] OCR 进程已停止")

    def push_task(self, tid, crop, class_id):
        """
        生产者：主进程调用
        """
        if self.task_queue.full():
            return False
        
        try:
            # 这里的 crop 是 numpy array，跨进程传递会有序列化开销
            # 但对于车牌小图(几KB)，这个开销远小于 OCR 计算耗时，是可接受的
            self.task_queue.put_nowait((tid, crop, class_id))
            return True
        except queue.Full:
            return False

    def get_results(self):
        """
        主进程调用：获取结果
        """
        results = []
        while not self.result_queue.empty():
            try:
                res = self.result_queue.get_nowait()
                results.append(res)
            except queue.Empty:
                break
        return results

    @staticmethod
    def _worker_entry_point(task_q, result_q):
        """
        【子进程入口】
        注意：这个函数是在完全独立的进程空间运行的。
        必须在这里初始化 HyperLPR，不能从外部传进来。
        """
        # 1. 设置子进程 CPU 亲和性
        SystemOptimizer.set_cpu_affinity("worker")

        # 2. 在子进程内初始化模型
        print(">>> [OCR Process] 正在加载 HyperLPR 模型...")
        recognizer = LicensePlateRecognizer()
        print(">>> [OCR Process] 模型加载完毕，开始待命")
        
        while True:
            try:
                # 阻塞等待任务
                tid, crop, class_id = task_q.get()
                
                # === 执行识别 ===
                # 这里不需要 try-catch queue.Empty，因为 get() 默认阻塞
                code, color, conf = recognizer.predict(crop)
                
                # 放入结果
                if color != "Unknown":
                    area = crop.shape[0] * crop.shape[1]
                    result_q.put((tid, color, conf, area))
                    
            except Exception as e:
                # 捕获异常防止子进程直接崩掉
                print(f"[OCR Process Error] {e}")
