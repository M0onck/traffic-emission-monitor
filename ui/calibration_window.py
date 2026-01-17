import cv2
import numpy as np
from perception.camera import CameraPreprocessor

class CalibrationUI:
    """
    [表现层] 交互式标定界面 (增强版)
    新增功能：
    1. 实时鸟瞰图 (BEV) 预览：辅助判断平行度。
    2. 虚拟网格投影：辅助判断纵向距离拉伸情况。
    """
    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(video_path)
        ret, raw_frame = self.cap.read()
        if not ret:
            raise ValueError(f"无法读取视频: {video_path}")
        self.cap.release()

        # --- 图像去畸变 ---
        # 确保 UI 显示的画面与 Engine 处理的画面完全一致
        # 如果您的 CameraPreprocessor 需要 config，这里可以传入 {} 或 None (取决于您之前的实现是否支持默认参数)
        preprocessor = CameraPreprocessor(config={}) 
        self.frame = preprocessor.preprocess(raw_frame)
        
        self.img_h, self.img_w = self.frame.shape[:2]
        self.window_name = "Calibration: Left=ROI, Right=BEV (Enter to confirm)"
        self.drag_idx = -1
        
        # 初始化标定点 (默认位于画面中心)
        cx, cy = self.img_w // 2, self.img_h // 2
        dx, dy = int(self.img_w * 0.20), int(self.img_h * 0.20)
        
        # 顺序：左下，右下，右上，左上 (符合逆时针或顺时针均可，这里用 0:BL, 1:BR, 2:TR, 3:TL)
        self.points = np.array([
            [cx - dx, cy + dy], [cx + dx, cy + dy], 
            [cx + dx, cy - dy], [cx - dx, cy - dy]
        ], dtype=np.float32)

        # 默认物理尺寸 (用户可调整)
        self.phys_w = 3.5 * 3 # 假设3车道宽
        self.phys_h = 30.0    # 假设30米长

        # 交互参数
        self.scale = 1.0
        self.pad_x = 0
        self.pad_y = 0

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1600, 720) # 加宽窗口以容纳双视图

    def _get_homography(self):
        """计算当前的单应性矩阵"""
        # 目标点：左下(0,h), 右下(w,h), 右上(w,0), 左上(0,0) -> 对应图像坐标系 (x,y)
        # 注意：在BEV图像中，我们希望上方是远处(y=0)，下方是近处(y=h)
        # 为了显示方便，我们映射到一个固定的像素尺寸，比如 300x600
        bev_w, bev_h = 300, 600
        
        src = self.points.astype(np.float32)
        dst = np.array([
            [0, bev_h], [bev_w, bev_h], 
            [bev_w, 0], [0, 0]
        ], dtype=np.float32)
        
        return cv2.findHomography(src, dst)[0], bev_w, bev_h

    def _update_transform_params(self, win_w, win_h):
        # 左侧留 70% 给原图，右侧 30% 给 BEV
        self.split_x = int(win_w * 0.7)
        view_w = self.split_x
        
        self.scale = min(view_w / self.img_w, win_h / self.img_h)
        self.pad_x = (view_w - int(self.img_w * self.scale)) // 2
        self.pad_y = (win_h - int(self.img_h * self.scale)) // 2

    def _mouse_to_img_coords(self, mx, my):
        # 限制鼠标只能在左侧区域操作
        if mx > self.split_x: return -1, -1
        img_x = (mx - self.pad_x) / self.scale
        img_y = (my - self.pad_y) / self.scale
        return float(img_x), float(img_y)

    def _img_to_display_coords(self, ix, iy):
        dx = int(ix * self.scale + self.pad_x)
        dy = int(iy * self.scale + self.pad_y)
        return dx, dy

    def _mouse_callback(self, event, x, y, flags, param):
        real_x, real_y = self._mouse_to_img_coords(x, y)
        hit_radius = 20 / self.scale

        if event == cv2.EVENT_LBUTTONDOWN:
            if real_x < 0: return
            for i, (px, py) in enumerate(self.points):
                if np.linalg.norm([real_x - px, real_y - py]) < hit_radius:
                    self.drag_idx = i
                    break
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drag_idx != -1 and real_x > 0:
                self.points[self.drag_idx] = [real_x, real_y]
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_idx = -1

    def run(self):
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        print(">>> [图形标定] 拖动角点。右侧预览用于检查平行度。")
        print(">>> 键盘操作: [W/S]调整物理长度, [A/D]调整物理宽度, [Enter]确认")
        
        while True:
            # 1. 窗口布局
            try:
                _, _, win_w, win_h = cv2.getWindowImageRect(self.window_name)
                if win_w <= 0: win_w, win_h = 1600, 720
            except:
                win_w, win_h = 1600, 720
            
            self._update_transform_params(win_w, win_h)
            
            # 创建画布
            canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
            
            # --- 左侧：原图与控制点 ---
            new_w, new_h = int(self.img_w * self.scale), int(self.img_h * self.scale)
            resized_frame = cv2.resize(self.frame, (new_w, new_h))
            y_off, x_off = self.pad_y, self.pad_x
            canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized_frame
            
            # 计算变换矩阵
            M, bev_w, bev_h = self._get_homography()
            
            # 绘制虚拟网格 (Projected Grid)
            # 在原图上画出物理世界的等分线，比如每5米画一条横线
            if M is not None:
                try:
                    M_inv = np.linalg.inv(M)
                    # 在 BEV 空间生成网格点
                    grid_lines = []
                    steps_y = 5 # 纵向每 1/5 画一条线
                    for i in range(1, steps_y):
                        # BEV 坐标系下的横线
                        y_bev = i * (bev_h / steps_y)
                        pt1_bev = np.array([[[0, y_bev]]], dtype=np.float32)
                        pt2_bev = np.array([[[bev_w, y_bev]]], dtype=np.float32)
                        
                        # 反变换回原图坐标
                        pt1_src = cv2.perspectiveTransform(pt1_bev, M_inv)[0][0]
                        pt2_src = cv2.perspectiveTransform(pt2_bev, M_inv)[0][0]
                        
                        p1 = self._img_to_display_coords(pt1_src[0], pt1_src[1])
                        p2 = self._img_to_display_coords(pt2_src[0], pt2_src[1])
                        cv2.line(canvas, p1, p2, (0, 255, 255), 1)
                        
                        # 标注距离
                        dist_label = f"{self.phys_h * (1 - i/steps_y):.1f}m"
                        cv2.putText(canvas, dist_label, (p1[0]-60, p1[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                except Exception as e:
                    pass

            # 绘制控制点多边形
            display_points = [self._img_to_display_coords(p[0], p[1]) for p in self.points]
            cv2.polylines(canvas, [np.array(display_points, np.int32)], True, (0, 255, 0), 2)
            
            for i, (dx, dy) in enumerate(display_points):
                col = (0, 0, 255) if i == self.drag_idx else (0, 255, 0)
                cv2.circle(canvas, (dx, dy), 8, col, -1)
                label = ["BL", "BR", "TR", "TL"][i]
                cv2.putText(canvas, label, (dx+10, dy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

            # --- 右侧：BEV 预览 ---
            if M is not None:
                # 截取原图的 ROI 并变换
                bev_img = cv2.warpPerspective(self.frame, M, (bev_w, bev_h))
                
                # 将 BEV 贴到右侧居中
                bev_x_start = self.split_x + (win_w - self.split_x - bev_w) // 2
                bev_y_start = (win_h - bev_h) // 2
                if bev_x_start > 0 and bev_y_start > 0:
                    canvas[bev_y_start:bev_y_start+bev_h, bev_x_start:bev_x_start+bev_w] = bev_img
                    
                    # 画外框
                    cv2.rectangle(canvas, (bev_x_start, bev_y_start), 
                                  (bev_x_start+bev_w, bev_y_start+bev_h), (255, 255, 255), 2)
                    
                    # 标注信息
                    info_txt = f"Physical Size: {self.phys_w:.1f}m x {self.phys_h:.1f}m"
                    cv2.putText(canvas, info_txt, (self.split_x + 20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(canvas, "Check if lanes are PARALLEL here ->", (self.split_x + 20, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            cv2.imshow(self.window_name, canvas)
            
            # 键盘交互
            key = cv2.waitKey(10)
            if key == 13: # Enter
                break
            elif key == ord('w'): self.phys_h += 1.0
            elif key == ord('s'): self.phys_h = max(1.0, self.phys_h - 1.0)
            elif key == ord('a'): self.phys_w = max(1.0, self.phys_w - 0.5)
            elif key == ord('d'): self.phys_w += 0.5
            elif key == ord('q'): exit()
        
        cv2.destroyWindow(self.window_name)
        
        # 返回最终标定结果
        # 目标点构造：[0,h], [w,h], [w,0], [0,0] (对应 BL, BR, TR, TL)
        # 注意物理坐标系通常以左下角为原点，或者以车辆进入方向为Y轴正向
        # 这里为了计算方便，我们定义物理坐标系:
        # 0: (0, 0)          -> BL
        # 1: (phys_w, 0)     -> BR
        # 2: (phys_w, phys_h)-> TR
        # 3: (0, phys_h)     -> TL
        # 这样 Y 轴正方向是 "远离摄像机" 的方向
        target_points = np.array([
            [0, 0], 
            [self.phys_w, 0], 
            [self.phys_w, self.phys_h], 
            [0, self.phys_h]
        ], dtype=np.float32)
        
        return self.points, target_points
