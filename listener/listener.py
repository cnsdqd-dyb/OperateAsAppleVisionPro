import cv2
from screeninfo import get_monitors
import logging
import numpy as np

# logging.basicConfig(level=logging.INFO)

# 定义一个logger GUI
class LoggerGUI:
    def __init__(self, window_name="Logger", size=(200, 200)):
        self.window_name = window_name
        self.text = ""
        self.running = True
        self.image = None
        self.size = size

    # 开始显示日志
    def start(self):

        # 创建窗口 左上角显示日志
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.size[0], self.size[1])
        # 透明度
        # cv2.setWindowProperty(self.window_name, cv2.WND_PROP_OPACITY, 0.5)

    def update(self):
        # 显示日志
        if self.image is not None:
            # 查看图像尺寸
            if self.image.shape[0] != self.size[0] or self.image.shape[1] != self.size[1]:
                # 重新设置image尺寸
                self.image = cv2.resize(self.image, self.size)

            # 显示图像
            cv2.putText(self.image, self.text, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(self.window_name, self.image)
        else:
            cv2.putText(np.zeros((100, 100, 3), dtype=np.uint8), self.text, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(self.window_name, np.zeros((100, 100, 3), dtype=np.uint8))
        if cv2.waitKey(5) & 0xFF == 27:
            self.running = False

    def resize_img(self):
        if self.image is not None:
            # 查看图像尺寸
            if self.image.shape[0] != self.size[0] or self.image.shape[1] != self.size[1]:
                # 重新设置image尺寸
                self.image = cv2.resize(self.image, self.size)

            # 显示图像
            cv2.putText(self.image, self.text, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            self.image = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.putText(self.image, self.text, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        return self.image
    
    # 停止显示日志
    def stop(self):
        self.running = False
        # 关闭窗口
        cv2.destroyAllWindows()

    def log(self, text):
        self.text = text

# 定义一个TrackerListener类
class TrackerListener:
    def __init__(self, tracker_functions, control_function=None):
        self.tracker_functions = tracker_functions
        self.control_function = control_function
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.loggers = {}
        for tracker_function in self.tracker_functions:
            self.loggers[tracker_function] = LoggerGUI(window_name=tracker_function.__name__)

        for key, logger in self.loggers.items():
            logger.start()

        screen_width = get_monitors()[0].width
        screen_height = get_monitors()[0].height
        self.canvas = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 255
        self.canvas_foreground = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 255
        # 全屏窗口
        cv2.namedWindow("Canvas", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Canvas", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self.screen_width = screen_width
        self.screen_height = screen_height

        self.eye_position = (0, 0)
        self.hand_position = (0, 0)

    # 开始监听
    def start(self):
        
        while self.cap.isOpened() and self.running:
            success, image = self.cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue   


            control_events = []
            for tracker_function in self.tracker_functions:
                control_, meta = tracker_function(image, self.loggers[tracker_function])
                if meta:
                    if "eye_position" in meta:
                        self.eye_position = meta["eye_position"]
                    if "hand_position" in meta:
                        self.hand_position = meta["hand_position"]
                if control_:
                    control_events.append(control_)

            if self.control_function and control_events:
                self.canvas, self.canvas_foreground = self.control_function(control_events, eye_position=self.eye_position, canvas_background=self.canvas)

            if cv2.waitKey(5) & 0xFF == 27:
                self.running = False
                for key, logger in self.loggers.items():
                    logger.stop()

            # LOGGER GUI
            for idx, (key, logger) in enumerate(self.loggers.items()):
                # 更新日志 写到 Canvas
                logger_image = logger.resize_img()
                
                # 图片大小不一致，拼接到左上角第idx个位置
                self.canvas[:logger_image.shape[0], idx*logger_image.shape[1]:(idx+1)*logger_image.shape[1]] = logger_image

            cv2.imshow("Canvas", self.canvas_foreground)


        self.cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        self.running = False
