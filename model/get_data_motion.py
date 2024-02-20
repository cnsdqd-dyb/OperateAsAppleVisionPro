import tkinter as tk
import time
import numpy as np
import cv2
import mediapipe as mp
from screeninfo import get_monitors
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tkinter import ttk  # 导入ttk模块，用于进度条

def cal_eye_ray(iris, eye_center, eye_size):

    # 计算眼睛的方向向量
    eye_ray = np.array([iris.x - eye_center[0], iris.y - eye_center[0], eye_size])
    eye_ray = eye_ray / np.linalg.norm(eye_ray)
    # print(iris,eye_center)
    return eye_ray

def cal_face_ray(left, right, nose_top):
    # 计算面部的方向向量
    face_center = [(left.x + right.x) / 2, (left.y + right.y) / 2, (left.z + right.z) / 2]
    face_ray = np.array([face_center[0] - nose_top.x, face_center[1] - nose_top.y, face_center[2] - nose_top.z])
    face_ray = face_ray / np.linalg.norm(face_ray)
    # print(face_ray)
    return face_ray

def preprocess_data(detection_result, screen_width, screen_height):
    # 计算
    left_eye_center = [(detection_result.face_landmarks[0][33].x + detection_result.face_landmarks[0][133].x) / 2, (detection_result.face_landmarks[0][168].y + detection_result.face_landmarks[0][6].y) / 2, (detection_result.face_landmarks[0][33].z + detection_result.face_landmarks[0][133].z) / 2]
    left_eye_size = np.linalg.norm([detection_result.face_landmarks[0][33].x - detection_result.face_landmarks[0][133].x, detection_result.face_landmarks[0][33].y - detection_result.face_landmarks[0][133].y, detection_result.face_landmarks[0][33].z - detection_result.face_landmarks[0][133].z])
    left_ray = cal_eye_ray(detection_result.face_landmarks[0][468], left_eye_center, left_eye_size) # 单位向量

    right_eye_center = [(detection_result.face_landmarks[0][362].x + detection_result.face_landmarks[0][263].x) / 2, (detection_result.face_landmarks[0][168].y + detection_result.face_landmarks[0][6].y) / 2, (detection_result.face_landmarks[0][362].z + detection_result.face_landmarks[0][263].z) / 2]
    right_eye_size = np.linalg.norm([detection_result.face_landmarks[0][362].x - detection_result.face_landmarks[0][263].x, detection_result.face_landmarks[0][362].y - detection_result.face_landmarks[0][263].y, detection_result.face_landmarks[0][362].z - detection_result.face_landmarks[0][263].z])
    right_ray = cal_eye_ray(detection_result.face_landmarks[0][473], right_eye_center, right_eye_size) # 单位向量


    face_ray = cal_face_ray(detection_result.face_landmarks[0][93], detection_result.face_landmarks[0][323], detection_result.face_landmarks[0][19])

    # Calculate the closest point between the two eye rays
    focal_point_x = (left_ray[0] + right_ray[0]) / 2


    # 根据脸的法向量和中心点进行焦点位置变换
    
    # 先计算face_ray在屏幕的偏移投影
    x_offset = face_ray[0] * detection_result.face_landmarks[0][19].x
    y_offset = face_ray[1] * detection_result.face_landmarks[0][19].y

    # 计算焦点在屏幕的位置
    screen_x_offset = (x_offset - focal_point_x * 1.5) * 3 + .5
    screen_y_offset = -(y_offset + .15) * 5 + .5

    # tm.add_trace([screen_x_offset, screen_y_offset, 0])
    # screen_x, screen_y, _ = tm.get_current()
    screen_x, screen_y = screen_x_offset * screen_width, screen_y_offset * screen_height

    train_data = []
    train_data.append(left_ray[0])
    train_data.append(right_ray[0])
    train_data.append(x_offset)
    train_data.append(y_offset)
    train_data.append(screen_x)
    train_data.append(screen_y)

    for idx, lm in enumerate(detection_result.face_landmarks[0]):
        if idx in [93, 323, 10, 152, 19, 473, 474, 475, 476, 477, 362, 263, 468, 469, 470, 471, 472, 386, 374]:
            train_data.append(lm.x)
            train_data.append(lm.y)
            train_data.append(lm.z)

    for v in face_ray:
        train_data.append(v)

    return train_data


# 获取屏幕分辨率
screen_width_ = get_monitors()[0].width
screen_height_ = get_monitors()[0].height

# 初始化MediaPipe检测器
base_options = python.BaseOptions(model_asset_path='tracker/face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=False,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# 初始化Tkinter窗口
root = tk.Tk()
root.title("Eye Tracking Calibration")
root.attributes('-fullscreen', True)  # 全屏

# 获取屏幕宽度和高度
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# 创建进度条
progress_var = tk.DoubleVar()  # 进度条变量
progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100, length=screen_width)
progress_bar.pack(side="bottom")  # 将进度条放置在窗口底部

# 定义路径点
path_points = [
    (screen_width // 10, screen_height // 10),  # 左上角
    (screen_width - screen_width // 10, screen_height // 10),  # 右上角
    (screen_width - screen_width // 10, screen_height - screen_height // 10),  # 右下角
    (screen_width // 10, screen_height - screen_height // 10),  # 左下角
    (screen_width // 10, screen_height // 10),  # 回到左上角
]

# 添加中间的路径点
num_intermediate_points = 4  # 可以根据需要调整中间点的数量
for i in range(1, num_intermediate_points + 1):
    fraction = i / (num_intermediate_points + 1)
    path_points.append((int(screen_width * fraction), screen_height // 10))  # 上边缘
    path_points.append((int(screen_width * fraction), screen_height - screen_height // 10))  # 下边缘

# 再次遍历屏幕的高度，添加垂直路径点
for i in range(1, num_intermediate_points + 1):
    fraction = i / (num_intermediate_points + 1)
    path_points.append((screen_width // 10, int(screen_height * fraction)))  # 左边缘
    path_points.append((screen_width - screen_width // 10, int(screen_height * fraction)))  # 右边缘


# 定义运动速度（像素/秒）
speed = 1000  # 假设每秒移动speed像素

# 定义采样时间间隔（秒）
sampling_interval = 0.1

# 用户引导时间
user_guide_time = 1

# 创建画布
canvas = tk.Canvas(root, width=screen_width, height=screen_height)
canvas.pack()

# 添加标题
title_text = "Eye Tracking Calibration - Multi-segment Movement"
canvas.create_text(screen_width // 2, 50, text=title_text, font=('Helvetica', 24), fill='black')

# 用于保存数据的列表
data_list = []

# 打开摄像头
cap = cv2.VideoCapture(0)

# 运行路径点
total_points = len(path_points) - 1
for i in range(len(path_points) - 1):
    start_pos = path_points[i]
    end_pos = path_points[i + 1]

    # 更新进度条
    progress_var.set((i + 1) / total_points * 100)
    root.update_idletasks()

    # 计算总运动时间
    distance = np.sqrt((end_pos[0] - start_pos[0]) ** 2 + (end_pos[1] - start_pos[1]) ** 2)
    total_time = distance / speed

    print(f"Moving from {start_pos} to {end_pos} in {total_time} seconds")


    wait_time = user_guide_time
    while wait_time > 0:
        canvas.delete("target")
        canvas.create_oval(start_pos[0] - 30, start_pos[1] - 30, start_pos[0] + 30, start_pos[1] + 30, fill='green', tags="target")
        canvas.create_oval(end_pos[0] - 30, end_pos[1] - 30, end_pos[0] + 30, end_pos[1] + 30, fill='red', tags="target")
        canvas.create_line(start_pos[0], start_pos[1], end_pos[0], end_pos[1], fill='black', tags="target")
        time.sleep(sampling_interval)
        wait_time -= sampling_interval

        # 显示倒计时
        canvas.delete("countdown")
        canvas.create_text(screen_width // 2, screen_height // 2, text=f"start in {int(wait_time)} sec", font=('Helvetica', 48), fill='black', tags="countdown")
        root.update()

    # 开始直线运动
    start_time = time.time()
    current_time = 0
    current_pos_x = start_pos[0]
    current_pos_y = start_pos[1]

    while abs(current_pos_x - end_pos[0]) > 1 or abs(current_pos_y - end_pos[1]) > 1:
        # 计算当前位置
        t = current_time / total_time
        print(f"rate: {t}")
        if t > 1:
            break

        current_pos_x = start_pos[0] + (end_pos[0] - start_pos[0]) * t
        current_pos_y = start_pos[1] + (end_pos[1] - start_pos[1]) * t
        current_pos = (int(current_pos_x), int(current_pos_y))

        # 画圆圈表示当前位置
        canvas.delete("target")
        canvas.create_oval(current_pos[0] - 30, current_pos[1] - 30, current_pos[0] + 30, current_pos[1] + 30, fill='green', tags="target")
        # 显示倒计时
        canvas.delete("countdown")
        canvas.create_text(screen_width // 2, screen_height // 2, text=f"Sampling", font=('Helvetica', 48), fill='black', tags="countdown")
        root.update()

        # 按照采样时间间隔采集数据
        if cap.isOpened():
            success, image = cap.read()
            if success:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                numpy_image = np.array(image)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)
                detection_result = detector.detect(image)
                if detection_result.face_landmarks:
                    train_data = preprocess_data(detection_result, screen_width, screen_height)
                    # print(train_data)
                    # input("Press Enter to continue...")
                    landmarks = []
                    for lm in detection_result.face_landmarks[0]:
                        landmarks.append(lm.x)
                        landmarks.append(lm.y)
                        landmarks.append(lm.z)

                    data_list.append({
                        'train_data': train_data,
                        'landmarks': landmarks,
                        'target': current_pos,
                    })
            else:
                print("Ignoring empty camera frame.")

        # 更新当前时间
        current_time = time.time() - start_time
        time.sleep(0.01)  # 短暂休眠以减少CPU使用率

    # 短暂停留在路径点
    time.sleep(1)

# 显示ESC退出
canvas.delete("exit")
canvas.create_text(screen_width // 2, screen_height - 50, text=f"Press ESC to exit", font=('Helvetica', 24), fill='black', tags="exit")
root.update()

# 保存数据
import json, os
user_config = json.load(open('config.json'))

if not os.path.exists(f'model/{user_config["name"]}'):
    os.makedirs(f'model/{user_config["name"]}')
with open(f'model/{user_config["name"]}/eye_tracking_data_multi_segment_movement.npz', 'wb') as f:
    np.savez(f, data=data_list)

# 退出程序的函数
def exit_program(event=None):
    root.destroy()
    cap.release()
    cv2.destroyAllWindows()

# 绑定Esc键到退出函数
root.bind('<Escape>', exit_program)

# 运行Tkinter事件循环
root.mainloop()