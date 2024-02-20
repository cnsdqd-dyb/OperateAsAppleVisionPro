import cv2
import mediapipe as mp
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from screeninfo import get_monitors
import threading
import pyautogui
import os
import joblib
from controller.controll_event import ControlEvent
import json
pyautogui.FAILSAFE=False

# 构建模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1500, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)

        return x

## 绘制类 #########################################
# 绘制人脸关键点
def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    # solutions.drawing_utils.draw_landmarks(
    #     image=annotated_image,
    #     landmark_list=face_landmarks_proto,
    #     connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
    #     landmark_drawing_spec=None,
    #     connection_drawing_spec=mp.solutions.drawing_styles
    #     .get_default_face_mesh_contours_style())
    # solutions.drawing_utils.draw_landmarks(
    #     image=annotated_image,
    #     landmark_list=face_landmarks_proto,
    #     connections=mp.solutions.face_mesh.FACEMESH_IRISES,
    #       landmark_drawing_spec=None,
    #       connection_drawing_spec=mp.solutions.drawing_styles
    #       .get_default_face_mesh_iris_connections_style())

  return annotated_image

# 裁剪人脸
def centerCropSquare(img, center, side=None, scaleWRTHeight=None):
    a = side is None
    b = scaleWRTHeight is None
    assert (not a and b) or (a and not b)  # Python没有实现异或操作
    half = 0
    if side is None:
        half = int(img.shape[0] * scaleWRTHeight / 2)
    else:
        half = int(side / 2)

    return img[(center[0] - half):(center[0] + half), (center[1] - half):(center[1] + half), :]

# 绘制脸
def plot_face_blendshapes_bar_graph(face_blendshapes):
  # Extract the face blendshapes category names and scores.
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # The blendshapes are ordered in decreasing score value.
  face_blendshapes_ranks = range(len(face_blendshapes_names))

  fig, ax = plt.subplots(figsize=(12, 12))
  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
  ax.invert_yaxis()

  # Label each bar with values
  for score, patch in zip(face_blendshapes_scores, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

  ax.set_xlabel('Score')
  ax.set_title("Face Blendshapes")
  plt.tight_layout()
  plt.show()

# 绘制手部关键点
def draw_hands(image, hands, mp_hands, mp_drawing, mp_drawing_styles):
    # 将BGR图像转换为RGB图像
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)

    # 将RGB图像转换回BGR图像以进行绘制
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        
    return image, results

# 计算眼睛的方向向量
def cal_eye_ray(iris, eye_center, eye_size):

    # 计算眼睛的方向向量
    eye_ray = np.array([iris.x - eye_center[0], iris.y - eye_center[0], eye_size])
    eye_ray = eye_ray / np.linalg.norm(eye_ray)
    # print(iris,eye_center)
    return eye_ray

# 计算面部的方向向量
def cal_face_ray(left, right, nose_top):
    # 计算面部的方向向量
    face_center = [(left.x + right.x) / 2, (left.y + right.y) / 2, (left.z + right.z) / 2]
    face_ray = np.array([face_center[0] - nose_top.x, face_center[1] - nose_top.y, face_center[2] - nose_top.z])
    face_ray = face_ray / np.linalg.norm(face_ray)
    # print(face_ray)
    return face_ray

# 预处理数据
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

# 绘制网格线
def draw_grid(img, grid_size):
    # 绘制网格线
    h, w = img.shape[:2]
    for i in range(0, w, grid_size):
        cv2.line(img, (i, 0), (i, h), (255, 255, 255), 1, 1)
    for j in range(0, h, grid_size):
        cv2.line(img, (0, j), (w, j), (255, 255, 255), 1, 1)
    return img

# 眼动追踪器 #########################################
## 定义一个KalmanFilter类
class KalmanFilter:
    def __init__(self, process_variance, measurement_variance):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimated_position = 0
        self.estimated_variance = 1

    def update(self, measurement):
        # 预测步骤
        self.estimated_variance += self.process_variance

        # 更新步骤
        kalman_gain = self.estimated_variance / (self.estimated_variance + self.measurement_variance)
        self.estimated_position += kalman_gain * (measurement - self.estimated_position)
        self.estimated_variance *= (1 - kalman_gain)

        return self.estimated_position
## 定义一个TraceManager类
class TraceManager:
    def __init__(self):
        self.trace = []

        self.trace_length = 10
        self.kalman_filter_x = KalmanFilter(process_variance=1e-4, measurement_variance=1e-2)
        self.kalman_filter_y = KalmanFilter(process_variance=1e-4, measurement_variance=1e-2)

        self.kalman_filter_vx = KalmanFilter(process_variance=1e-4, measurement_variance=1e-2)
        self.kalman_filter_vy = KalmanFilter(process_variance=1e-4, measurement_variance=1e-2)

    def add_trace(self, trace):
        if len(self.trace) >= self.trace_length:
            self.trace.pop(0)
        self.trace.append(trace)

    def get_current(self):
        if not self.trace:
            return np.zeros(2)

        current_measurement = self.trace[-1][:2]  # 只取x和y
        estimated_x = self.kalman_filter_x.update(current_measurement[0])
        estimated_y = self.kalman_filter_y.update(current_measurement[1])
        return np.array([estimated_x, estimated_y])
    
    def get_speed(self):
        self.speed_trace = []
        if len(self.trace) < 2:
            return np.zeros(2)
        
        for i in range(len(self.trace) - 1):
            self.speed_trace.append([self.trace[i + 1][0] - self.trace[i][0], self.trace[i + 1][1] - self.trace[i][1]])
        
        current_measurement = self.speed_trace[-1]
        estimated_vx = self.kalman_filter_vx.update(current_measurement[0])
        estimated_vy = self.kalman_filter_vy.update(current_measurement[1])
        return np.array([estimated_vx, estimated_vy])
    
# 全局变量 #######################################
user_config = json.load(open('config.json', 'r'))
# 读取配置文件

## dl模型
dl_model = Net()
if os.path.exists(f'model/{user_config["name"]}/best_model.pth'):
    # load dl model
    # 加载最佳模型
    dl_model.load_state_dict(torch.load(f'model/{user_config["name"]}/best_model.pth'))
else:
    # 引导用户进行模型训练
    import model.get_data_motion
    import model.train
    dl_model.load_state_dict(torch.load(f'model/{user_config["name"]}/best_model.pth'))

if os.path.exists(f'model/{user_config["name"]}/random_forest_eye_tracking_model.joblib'):
    rf_model = joblib.load(f'model/{user_config["name"]}/random_forest_eye_tracking_model.joblib')

if os.path.exists(f'model/{user_config["name"]}/eye_tracking_calibration.npz'):
    args = np.load(f'model/{user_config["name"]}/eye_tracking_calibration.npz')['coefficients']

## trace manager
tm = TraceManager()

# 
base_options = python.BaseOptions(model_asset_path='tracker/face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                    output_face_blendshapes=False,
                                    output_facial_transformation_matrixes=True,
                                    num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# tracker functions ################################
# track function 类
class EyeTrackerFunction:
    DL_EYE_TRACKING = 0
    RF_EYE_TRACKING = 1
    ARGS_EYE_TRACKING = 2

    @staticmethod
    def get_tracker_function(tracker_type):
        if tracker_type == EyeTrackerFunction.DL_EYE_TRACKING:
            return EyeTrackerFunction.tracking_dl
        elif tracker_type == EyeTrackerFunction.RF_EYE_TRACKING:
            return EyeTrackerFunction.tracking_rf
        elif tracker_type == EyeTrackerFunction.ARGS_EYE_TRACKING:
            return EyeTrackerFunction.tracking_args
        else:
            assert False, "Invalid tracker type"

    def tracking_dl(image, logger):
        # 评估模型
        dl_model.eval()
        # gpu
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dl_model.to(device)
        
        original_image = image.copy()
        image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        numpy_image = np.array(image)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)
        detection_result = detector.detect(image)

        try:
            w, h = get_monitors()[0].width, get_monitors()[0].height
            data = preprocess_data(detection_result, w, h)
            landmarks = []
            for lm in detection_result.face_landmarks[0]:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
                landmarks.append(lm.z)
            with torch.no_grad():
                input_data = torch.tensor(np.array([data + landmarks])).float()
                if torch.cuda.is_available():
                    input_data = input_data.to(device)
                with torch.no_grad():
                    screen_x, screen_y = dl_model(input_data)[0]

            tm.add_trace([screen_x, screen_y])
            screen_x, screen_y = tm.get_current()
            # vx, vy = tm.get_speed()
            logger.text = f"eye pos: {int(screen_x)}, {int(screen_y)}"

            # event = ControlEvent(ControlEvent.MOVE_REL, x = vx, y = vy) 
            event = ControlEvent(ControlEvent.MOVE_TO, x = screen_x, y = screen_y)

            # 转换回BGR图像以进行绘制
            original_image.flags.writeable = True
            # 绘制左右眼
            img_h, img_w, _ = original_image.shape
            left_iris_center = (int(detection_result.face_landmarks[0][468].x * img_w), int(detection_result.face_landmarks[0][468].y * img_h))
            right_iris_center = (int(detection_result.face_landmarks[0][473].x * img_w), int(detection_result.face_landmarks[0][473].y * img_h))
            left_radius = np.linalg.norm([detection_result.face_landmarks[0][469].x * img_w - detection_result.face_landmarks[0][468].x * img_w, detection_result.face_landmarks[0][469].y * img_h - detection_result.face_landmarks[0][468].y * img_h])
            right_radius = np.linalg.norm([detection_result.face_landmarks[0][474].x * img_w - detection_result.face_landmarks[0][473].x * img_w, detection_result.face_landmarks[0][474].y * img_h - detection_result.face_landmarks[0][473].y * img_h])
            original_image = cv2.circle(original_image, left_iris_center, int(left_radius-1), (255, 0, 0), 1)
            original_image = cv2.circle(original_image, left_iris_center, int(2), (0, 255, 0), 1)
            original_image = cv2.circle(original_image, right_iris_center, int(right_radius-1), (255, 0, 0), 1)
            original_image = cv2.circle(original_image, right_iris_center, int(2), (0, 255, 0), 1)

            # crop eye
            radius = int(max(left_radius, right_radius)) * 2
            left_eye = original_image[int(left_iris_center[1] - radius):int(left_iris_center[1] + radius), int(left_iris_center[0] - radius):int(left_iris_center[0] + radius)]
            right_eye = original_image[int(right_iris_center[1] - radius):int(right_iris_center[1] + radius), int(right_iris_center[0] - radius):int(right_iris_center[0] + radius)]
            
            # eyes
            eyes = np.concatenate((left_eye, right_eye), axis=1)
            # resize
            face = cv2.resize(original_image, (radius*4, radius*2))
            eyes_face = np.concatenate((eyes, face), axis=0)
            logger.image = eyes_face

            return event, {"eye_position": (int(screen_x), int(screen_y))}
        except Exception as e:
            print(e)
            return None, {}
        
    def tracking_rf(image, logger):
        original_image = image.copy()

        image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        numpy_image = np.array(image)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)
        detection_result = detector.detect(image)

        try:
            left_eye_center = [(detection_result.face_landmarks[0][33].x + detection_result.face_landmarks[0][133].x) / 2, (detection_result.face_landmarks[0][168].y + detection_result.face_landmarks[0][6].y) / 2, (detection_result.face_landmarks[0][33].z + detection_result.face_landmarks[0][133].z) / 2]
            left_eye_size = np.linalg.norm([detection_result.face_landmarks[0][33].x - detection_result.face_landmarks[0][133].x, detection_result.face_landmarks[0][33].y - detection_result.face_landmarks[0][133].y, detection_result.face_landmarks[0][33].z - detection_result.face_landmarks[0][133].z])

            right_eye_center = [(detection_result.face_landmarks[0][362].x + detection_result.face_landmarks[0][263].x) / 2, (detection_result.face_landmarks[0][168].y + detection_result.face_landmarks[0][6].y) / 2, (detection_result.face_landmarks[0][362].z + detection_result.face_landmarks[0][263].z) / 2]
            right_eye_size = np.linalg.norm([detection_result.face_landmarks[0][362].x - detection_result.face_landmarks[0][263].x, detection_result.face_landmarks[0][362].y - detection_result.face_landmarks[0][263].y, detection_result.face_landmarks[0][362].z - detection_result.face_landmarks[0][263].z])
            
            w, h = get_monitors()[0].width, get_monitors()[0].height
            data = preprocess_data(detection_result, w, h)
            landmarks = []
            for lm in detection_result.face_landmarks[0]:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
                landmarks.append(lm.z)
            screen_x, screen_y = rf_model.predict([data + landmarks])[0]
            tm.add_trace([screen_x, screen_y])
            screen_x, screen_y = tm.get_current()

            logger.text = f"eye pos: {int(screen_x)}, {int(screen_y)}"
            
            event = ControlEvent(ControlEvent.MOVE_TO, x = screen_x, y = screen_y)
            # 转换回BGR图像以进行绘制
            original_image.flags.writeable = True
            # 绘制左右眼
            original_image = cv2.circle(original_image, (int(left_eye_center[0]), int(left_eye_center[1])), int(left_eye_size), (0, 0, 255), -1)
            original_image = cv2.circle(original_image, (int(right_eye_center[0]), int(right_eye_center[1])), int(right_eye_size), (0, 0, 255), -1)
            
            logger.image = original_image
            return event, {"eye_position": (int(screen_x), int(screen_y))}
        
        except Exception as e:
            print(e)
            return None, {}


    def tracking_args(image, logger):
        original_image = image.copy()
        image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        numpy_image = np.array(image)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)
        detection_result = detector.detect(image)

        try:
            w, h = get_monitors()[0].width, get_monitors()[0].height
            # 计算
            left_eye_center = [(detection_result.face_landmarks[0][33].x + detection_result.face_landmarks[0][133].x) / 2, (detection_result.face_landmarks[0][168].y + detection_result.face_landmarks[0][6].y) / 2, (detection_result.face_landmarks[0][33].z + detection_result.face_landmarks[0][133].z) / 2]
            left_eye_size = np.linalg.norm([detection_result.face_landmarks[0][33].x - detection_result.face_landmarks[0][133].x, detection_result.face_landmarks[0][33].y - detection_result.face_landmarks[0][133].y, detection_result.face_landmarks[0][33].z - detection_result.face_landmarks[0][133].z])
            left_ray = cal_eye_ray(detection_result.face_landmarks[0][468], left_eye_center, left_eye_size) # 单位向量

            right_eye_center = [(detection_result.face_landmarks[0][362].x + detection_result.face_landmarks[0][263].x) / 2, (detection_result.face_landmarks[0][168].y + detection_result.face_landmarks[0][6].y) / 2, (detection_result.face_landmarks[0][362].z + detection_result.face_landmarks[0][263].z) / 2]
            right_eye_size = np.linalg.norm([detection_result.face_landmarks[0][362].x - detection_result.face_landmarks[0][263].x, detection_result.face_landmarks[0][362].y - detection_result.face_landmarks[0][263].y, detection_result.face_landmarks[0][362].z - detection_result.face_landmarks[0][263].z])
            right_ray = cal_eye_ray(detection_result.face_landmarks[0][473], right_eye_center, right_eye_size) # 单位向量


            face_ray = cal_face_ray(detection_result.face_landmarks[0][93], detection_result.face_landmarks[0][323], detection_result.face_landmarks[0][19])

            # Calculate the closest point between the two eye rays
            focal_point_x = (left_ray[0] * 1 + right_ray[0] * 1) / 2 + 0

            # 根据脸的法向量和中心点进行焦点位置变换
            
            # 先计算face_ray在屏幕的偏移投影
            x_offset = face_ray[0] * detection_result.face_landmarks[0][19].x
            y_offset = face_ray[1] * detection_result.face_landmarks[0][19].y

            # 计算焦点在屏幕的位置
            screen_x_offset = (x_offset - focal_point_x * args[4] * 2) * args[5] + .5
            screen_y_offset = -(y_offset + .15) * 3 + .5

            tm.add_trace([screen_x_offset, screen_y_offset])
            screen_x, screen_y = tm.get_current()

            logger.text = f"eye pos: {int(screen_x)}, {int(screen_y)}"
            
            event = ControlEvent(ControlEvent.MOVE_TO, x = screen_x, y = screen_y)

            # 转换回BGR图像以进行绘制
            original_image.flags.writeable = True
            # 绘制左右眼
            original_image = cv2.circle(original_image, (int(left_eye_center[0]), int(left_eye_center[1])), int(left_eye_size), (0, 0, 255), -1)
            original_image = cv2.circle(original_image, (int(right_eye_center[0]), int(right_eye_center[1])), int(right_eye_size), (0, 0, 255), -1)

            logger.image = original_image
            return event, {"eye_position": (int(screen_x), int(screen_y))}

        except Exception as e:
            print(e)
            return None, {}

        