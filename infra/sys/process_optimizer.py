import psutil
import os
import platform
import math

"""
[基础层] 系统进程优化器
功能：负责操作系统级别的进程优先级调整与 CPU 核心亲和性绑定。
职责：
1. 提升主进程的 CPU 优先级，减少视频丢帧。
2. 将主进程绑定到特定 CPU 核心（通常是性能核），避免频繁上下文切换。
3. 屏蔽 Windows/Linux 的底层 API 差异。
"""

class SystemOptimizer:
    @staticmethod
    def set_cpu_affinity(role: str):
        """
        设置当前进程的 CPU 亲和性与优先级
        根据 CPU 核心数进行 50/50 资源划分。
        策略：
        - 单核环境：不进行绑定，完全由 OS 调度。
        - 多核环境：
            - Main (主进程): 占用前 50% (向上取整) 的核心。
            - Worker (OCR进程): 占用后 50% (向下取整) 的核心。
        
        例如：
        - 2核: Main=[0], Worker=[1]
        - 3核: Main=[0, 1], Worker=[2] (奇数核优先给主进程)
        - 4核: Main=[0, 1], Worker=[2, 3]
        - 8核: Main=[0, 1, 2, 3], Worker=[4, 5, 6, 7]
        """
        # macOS (Darwin) 不支持标准的 CPU 亲和性设置，直接跳过
        if platform.system() == 'Darwin':
            return

        try:
            p = psutil.Process(os.getpid())
            # 获取逻辑核心数 (包含超线程)
            count = psutil.cpu_count(logical=True)
            
            # 1. 【边界情况处理】检测不到核心数 或 单核CPU
            if count is None or count < 2:
                print(f">>> [System] 检测到单核或未知CPU({count})，跳过亲和性绑定 (由OS自动调度)")
                # 在单核情况下，显式绑定到 [0] 和不绑定效果一样，
                # 但不绑定更安全，防止 range 报错。
                return

            # 2. 【核心分割算法】
            # 使用 ceil (向上取整) 确保奇数时主进程多拿一个
            # 例如 3 / 2 = 1.5 -> split_idx = 2
            split_idx = math.ceil(count / 2)

            all_cores = list(range(count))
            target_cores = []

            if role == "main":
                # 主进程拿前半段: [0, ..., split_idx-1]
                target_cores = all_cores[:split_idx]
                p.cpu_affinity(target_cores)
                print(f">>> [System] 主进程(UI) 已绑定至核心: {target_cores} (共{len(target_cores)}核)")

            elif role == "worker":
                # OCR进程拿后半段: [split_idx, ..., count-1]
                target_cores = all_cores[split_idx:]
                p.cpu_affinity(target_cores)
                print(f">>> [System] OCR进程(计算) 已绑定至核心: {target_cores} (共{len(target_cores)}核)")
                
        except Exception as e:
            # 容错：防止权限不足或其他底层错误导致程序崩溃
            print(f"[Warning] CPU亲和性设置失败: {e}")
