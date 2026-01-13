import cv2
import numpy as np

class CalibrationUI:
    """
    [表现层] 交互式标定界面
    功能：允许用户在视频首帧拖动4个角点，定义感兴趣区域 (ROI) 的物理尺寸。
    """
    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(video_path)
        ret, self.frame = self.cap.read()
        if not ret:
            raise ValueError(f"无法读取视频: {video_path}")
        self.cap.release()
        
        self.img_h, self.img_w = self.frame.shape[:2]
        self.window_name = "Calibration (Drag corners, Enter to finish)"
        self.drag_idx = -1
        
        # 初始化标定点 (默认位于画面中心)
        cx, cy = self.img_w // 2, self.img_h // 2
        dx, dy = int(self.img_w * 0.25), int(self.img_h * 0.25)
        
        self.points = np.array([
            [cx - dx, cy + dy], [cx + dx, cy + dy], 
            [cx + dx, cy - dy], [cx - dx, cy - dy]
        ], dtype=np.int32)

        # 变换参数 (用于鼠标坐标映射)
        self.scale = 1.0
        self.pad_x = 0
        self.pad_y = 0

        # 初始化为可缩放窗口
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)

    def _update_transform_params(self, win_w, win_h):
        """计算 [原始图像] -> [显示窗口] 的缩放和平移参数"""
        if win_w <= 0 or win_h <= 0: return
        self.scale = min(win_w / self.img_w, win_h / self.img_h)
        new_w = int(self.img_w * self.scale)
        new_h = int(self.img_h * self.scale)
        self.pad_x = (win_w - new_w) // 2
        self.pad_y = (win_h - new_h) // 2

    def _mouse_to_img_coords(self, mx, my):
        """逆变换: 窗口坐标 -> 原始图像坐标"""
        img_x = (mx - self.pad_x) / self.scale
        img_y = (my - self.pad_y) / self.scale
        return int(np.clip(img_x, 0, self.img_w - 1)), int(np.clip(img_y, 0, self.img_h - 1))

    def _img_to_display_coords(self, ix, iy):
        """正变换: 原始图像坐标 -> 窗口坐标"""
        dx = int(ix * self.scale + self.pad_x)
        dy = int(iy * self.scale + self.pad_y)
        return dx, dy

    def _mouse_callback(self, event, x, y, flags, param):
        """处理鼠标拖拽事件"""
        real_x, real_y = self._mouse_to_img_coords(x, y)
        hit_radius = 15 / self.scale # 动态调整点击判定半径

        if event == cv2.EVENT_LBUTTONDOWN:
            for i, (px, py) in enumerate(self.points):
                if np.linalg.norm([real_x - px, real_y - py]) < hit_radius:
                    self.drag_idx = i
                    break
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drag_idx != -1:
                self.points[self.drag_idx] = [real_x, real_y]
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_idx = -1

    def run(self):
        """运行标定循环"""
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        print(">>> [图形标定] 请拖动角点匹配车道，按 'Enter' 确认。")
        
        while True:
            # 1. 窗口自适应
            try:
                _, _, win_w, win_h = cv2.getWindowImageRect(self.window_name)
                if win_w <= 0: win_w, win_h = 1280, 720
            except:
                win_w, win_h = 1280, 720
            
            self._update_transform_params(win_w, win_h)

            # 2. 绘图
            canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
            new_w, new_h = int(self.img_w * self.scale), int(self.img_h * self.scale)
            resized_frame = cv2.resize(self.frame, (new_w, new_h))
            
            y_off, x_off = self.pad_y, self.pad_x
            canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized_frame

            # 3. 绘制控制点 (转换到显示坐标)
            display_points = [self._img_to_display_coords(p[0], p[1]) for p in self.points]
            display_points_np = np.array(display_points, dtype=np.int32)
            
            cv2.polylines(canvas, [display_points_np], True, (255, 255, 0), 2)
            for i, (dx, dy) in enumerate(display_points):
                col = (0, 0, 255) if i == self.drag_idx else (0, 255, 0)
                cv2.circle(canvas, (dx, dy), 10, col, -1)
                cv2.putText(canvas, str(i), (dx+15, dy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

            cv2.imshow(self.window_name, canvas)
            key = cv2.waitKey(10)
            if key in [13, 32]: break # Enter/Space to confirm
            elif key == ord('q'): exit()
        
        cv2.destroyWindow(self.window_name)
        
        # 控制台输入逻辑 (保持精简)
        print("\n" + "="*40)
        try:
            w = float(input(">>> 请输入区域宽度 (m, 默认21): ").strip() or 21.0)
            h = float(input(">>> 请输入区域长度 (m, 默认25): ").strip() or 25.0)
        except:
            w, h = 21.0, 25.0
            print(">>> 输入无效，使用默认值。")

        target_points = np.array([[0,0], [w,0], [w,h], [0,h]], dtype=np.float32)
        return self.points, target_points
