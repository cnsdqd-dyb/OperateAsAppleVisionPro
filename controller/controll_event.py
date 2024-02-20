import threading
import pyautogui
import time
import joblib
import numpy as np
import cv2

class ControlEvent:
    MOVE_TO = 0
    MOVE_REL = 1
    LEFT_CLICK = 2
    RESIZE = 3
    FIST = 4

    def __init__(self, event_type, **kwargs):
        self.event_type = event_type
        self.x = kwargs.get('x')
        self.y = kwargs.get('y')
        self.additional_args = kwargs

    def __str__(self):
        return f"ControlEvent(type={self.event_type}, x={self.x}, y={self.y}, additional_args={self.additional_args})"

class StoppableThread(threading.Thread):
    def __init__(self, target, args=(), kwargs=None, max_runtime=1):
        super().__init__(target=target, args=args, kwargs=kwargs if kwargs else {})
        self._stop_event = threading.Event()
        self._start_time = time.time()
        self._max_runtime = max_runtime

    def run(self):
        while not self._stop_event.is_set():
            if time.time() - self._start_time > self._max_runtime:
                print(f"Stopping thread after {self._max_runtime} seconds.")
                break
            super().run()
            break  # If the target function is finished, exit the loop

    def stop(self):
        self._stop_event.set()

last_move_time = time.time()
def handle_events_virtual_mouse(events: ControlEvent = [], eye_position=None, canvas_background=None):
    global last_move_time
    if canvas_background is None:
        canvas_background = np.zeros((500, 500, 3), dtype=np.uint8) * 255
    canvas_background = canvas_background.copy()
    canvas_foreground = canvas_background.copy()
    for event in events:
        if event.event_type == ControlEvent.MOVE_TO and last_move_time + 0.001 < time.time():
            # draw a circle on the canvas
            last_move_time = time.time()
            canvas_foreground = cv2.circle(canvas_foreground, (int(event.x), int(event.y)), 10, (0, 255, 0), -1)
        
        if event.event_type == ControlEvent.LEFT_CLICK:
            # draw a circle on the canvas
            canvas_background = cv2.circle(canvas_background, (int(eye_position[0]), int(eye_position[1])), 10, (0, 0, 0), -1)
            
        if event.event_type == ControlEvent.MOVE_REL:
            # draw a line on the canvas
            canvas_foreground = cv2.line(canvas_foreground, (int(eye_position[0]), int(eye_position[1])), 
                                         (int(eye_position[0] + event.x), int(eye_position[1] + event.y)), (255, 0, 0), 5)
            
        if event.event_type == ControlEvent.RESIZE:
            try:
                resize_v = max(event.x, event.y)
                print(event.x, event.y, resize_v)
                # 计算新的尺寸
                new_width = int(canvas_foreground.shape[1] * resize_v)
                new_height = int(canvas_foreground.shape[0] * resize_v)
                # 缩放图像
                resized_foreground = cv2.resize(canvas_foreground, (new_width, new_height))
                
                if resize_v < 1:
                    # 缩小图像，需要在周围填充白色
                    border_top = (canvas_foreground.shape[0] - new_height) // 2
                    border_bottom = canvas_foreground.shape[0] - new_height - border_top
                    border_left = (canvas_foreground.shape[1] - new_width) // 2
                    border_right = canvas_foreground.shape[1] - new_width - border_left
                    canvas_foreground = cv2.copyMakeBorder(resized_foreground, border_top, border_bottom, border_left, border_right, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                else:
                    # 放大图像，需要裁剪到原始尺寸
                    center_x, center_y = eye_position
                    start_x = max(0, center_x - canvas_foreground.shape[1] // 2)
                    end_x = start_x + canvas_foreground.shape[1]
                    start_y = max(0, center_y - canvas_foreground.shape[0] // 2)
                    end_y = start_y + canvas_foreground.shape[0]
                    # 确保不超出边界
                    end_x = min(end_x, resized_foreground.shape[1])
                    end_y = min(end_y, resized_foreground.shape[0])
                    start_x = end_x - canvas_foreground.shape[1]
                    start_y = end_y - canvas_foreground.shape[0]
                    canvas_foreground = resized_foreground[start_y:end_y, start_x:end_x]
            except Exception as e:
                print(f"An error occurred: {e}")


        if event.event_type == ControlEvent.FIST:
            # clear the canvas
            canvas_background = np.ones(canvas_background.shape, dtype=np.uint8) * 255
            canvas_foreground = np.ones(canvas_foreground.shape, dtype=np.uint8) * 255

    return canvas_background, canvas_foreground

last_move_time = time.time()
def handle_events(events: ControlEvent = [], eye_position=None, canvas_background=None):
    global last_move_time
    if canvas_background is None:
        canvas_background = np.zeros((500, 500, 3), dtype=np.uint8)
    canvas_background = canvas_background.copy()
    canvas_foreground = canvas_background.copy()
    for event in events:
        if event.event_type == ControlEvent.MOVE_TO and last_move_time + 0.001 < time.time():
            last_move_time = time.time()
            # pyautogui.moveTo(event.x, event.y)
            t = StoppableThread(target=pyautogui.moveTo, args=(event.x, event.y))
            t.run()
        
        if event.event_type == ControlEvent.LEFT_CLICK:
            t = StoppableThread(target=pyautogui.leftClick)
            t.run()

        if event.event_type == ControlEvent.MOVE_REL:
            t = StoppableThread(target=pyautogui.moveRel, args=(event.x, event.y))
            t.run()

        if event.event_type == ControlEvent.RESIZE:
            ...
            # TODO: Implement resizing of the window
            # image = np.ones((100, 100, 3), dtype=np.uint8)
            # cv2.imshow('image', cv2.resize(image, (int(event.x), int(event.y))))

    return canvas_background, canvas_foreground
