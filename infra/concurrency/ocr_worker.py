import multiprocessing
import queue
import time
import numpy as np
from perception.lpr_adaptor import LicensePlateRecognizer
from infra.sys.process_optimizer import SystemOptimizer

"""
[基础层] 异步 OCR 组件
功能：封装多进程 OCR 识别逻辑，将 CPU 密集型的车牌识别任务从主线程剥离。
设计模式：生产者-消费者模式 (Producer-Consumer)
职责：
1. 维护独立的 OCR 进程，避免阻塞视频流处理。
2. 提供线程安全的任务队列 (Task Queue) 和结果队列 (Result Queue)。
"""

class AsyncOCRManager:
    """
    异步车牌识别管理器
    """
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
        """安全停止进程"""
        if self.worker_process and self.worker_process.is_alive():
            self.worker_process.terminate() # 强制结束
            self.worker_process.join()
        print(">>> [System] OCR 进程已停止")

    def push_task(self, tid, crop, class_id):
        """
        提交识别任务 (非阻塞)
        :return: True if pushed, False if queue full
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
        批量获取已完成的识别结果
        :return: list of (tracker_id, text, conf, area)
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
        [独立进程] 实际运行的消费者循环
        注意：HyperLPR3 必须在此进程内导入和初始化，不能跨进程共享。
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
