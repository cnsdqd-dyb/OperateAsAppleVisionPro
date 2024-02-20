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
import time
import joblib
from controller.controll_event import ControlEvent
import math

pyautogui.FAILSAFE=False

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)


def vector_2d_angle(v1,v2):
    '''
        求解二维向量的角度
    '''
    v1_x=v1[0]
    v1_y=v1[1]
    v2_x=v2[0]
    v2_y=v2[1]
    try:
        angle_= math.degrees(math.acos((v1_x*v2_x+v1_y*v2_y)/(((v1_x**2+v1_y**2)**0.5)*((v2_x**2+v2_y**2)**0.5))))
    except:
        angle_ = 65535.
    if angle_ > 180.:
        angle_ = 65535.
    return angle_
def hand_angle(hand_):
    '''
        获取对应手相关向量的二维角度,根据角度确定手势
    '''
    angle_list = []
    #---------------------------- thumb 大拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[2][0])),(int(hand_[0][1])-int(hand_[2][1]))),
        ((int(hand_[3][0])- int(hand_[4][0])),(int(hand_[3][1])- int(hand_[4][1])))
        )
    angle_list.append(angle_)
    #---------------------------- index 食指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])-int(hand_[6][0])),(int(hand_[0][1])- int(hand_[6][1]))),
        ((int(hand_[7][0])- int(hand_[8][0])),(int(hand_[7][1])- int(hand_[8][1])))
        )
    angle_list.append(angle_)
    #---------------------------- middle 中指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[10][0])),(int(hand_[0][1])- int(hand_[10][1]))),
        ((int(hand_[11][0])- int(hand_[12][0])),(int(hand_[11][1])- int(hand_[12][1])))
        )
    angle_list.append(angle_)
    #---------------------------- ring 无名指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[14][0])),(int(hand_[0][1])- int(hand_[14][1]))),
        ((int(hand_[15][0])- int(hand_[16][0])),(int(hand_[15][1])- int(hand_[16][1])))
        )
    angle_list.append(angle_)
    #---------------------------- pink 小拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[18][0])),(int(hand_[0][1])- int(hand_[18][1]))),
        ((int(hand_[19][0])- int(hand_[20][0])),(int(hand_[19][1])- int(hand_[20][1])))
        )
    angle_list.append(angle_)
    return angle_list
def h_gesture(angle_list):
    '''
        # 二维约束的方法定义手势
        # fist five gun love one six three thumbup yeah
    '''
    thr_angle = 65.
    thr_angle_thumb = 53.
    thr_angle_s = 49.
    gesture_str = None
    if 65535. not in angle_list:
        if (angle_list[0]>thr_angle_thumb) and (angle_list[1]>thr_angle) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "fist"
        elif (angle_list[0]<thr_angle_s) and (angle_list[1]<thr_angle_s) and (angle_list[2]<thr_angle_s) and (angle_list[3]<thr_angle_s) and (angle_list[4]<thr_angle_s):
            gesture_str = "five"
        elif (angle_list[0]<thr_angle_s)  and (angle_list[1]<thr_angle_s) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "gun"
        elif (angle_list[0]<thr_angle_s)  and (angle_list[1]<thr_angle_s) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]<thr_angle_s):
            gesture_str = "love"
        elif (angle_list[0]>5)  and (angle_list[1]<thr_angle_s) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "one"
        elif (angle_list[0]<thr_angle_s)  and (angle_list[1]>thr_angle) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]<thr_angle_s):
            gesture_str = "six"
        elif (angle_list[0]>thr_angle_thumb)  and (angle_list[1]<thr_angle_s) and (angle_list[2]<thr_angle_s) and (angle_list[3]<thr_angle_s) and (angle_list[4]>thr_angle):
            gesture_str = "three"
        elif (angle_list[0]<thr_angle_s)  and (angle_list[1]>thr_angle) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "thumbUp"
        elif (angle_list[0]>thr_angle_thumb)  and (angle_list[1]<thr_angle_s) and (angle_list[2]<thr_angle_s) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "two"
        else:
            gesture_str = "other"
    return gesture_str

def finger_gesture(hand_) -> str:
    # 4 8 靠近
    # print(hand_[4][0]-hand_[8][0],hand_[4][1]-hand_[8][1])
    if np.linalg.norm((hand_[4][0]-hand_[8][0],hand_[4][1]-hand_[8][1]))<30:
        return "touch"
    else:
        return "other"

def hands_tracker(image, logger):
    # 将BGR图像转换为RGB图像
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)

    finger_tips = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
          
            hand_local = []
            for i in range(21):
                x = hand_landmarks.landmark[i].x*image.shape[1]
                y = hand_landmarks.landmark[i].y*image.shape[0]
                hand_local.append((x,y))
            if hand_local:
                gesture_str = finger_gesture(hand_local)
                if gesture_str == "other":
                    angle_list = hand_angle(hand_local)
                    gesture_str = h_gesture(angle_list)
            
                logger.text = f"gesture: {gesture_str}"

                if gesture_str == "touch":
                    if len(results.multi_hand_landmarks) == 1:
                        return ControlEvent(ControlEvent.LEFT_CLICK), {"hand_position": hand_local[8]}
                    finger_tips.append(hand_local[8])
                elif gesture_str == "fist":
                    return ControlEvent(ControlEvent.FIST), {"hand_position": hand_local[8]}
                # elif gesture_str == "one":
                #     return ControlEvent(ControlEvent.MOVE_TO, x=hand_local[8][0], y=hand_local[8][1])
                # elif gesture_str == "six":
                #     return ControlEvent(ControlEvent.MOVE_TO, x=hand_local[8][0], y=hand_local[8][1])
                # elif gesture_str == "three":
                #     return ControlEvent(ControlEvent.MOVE_TO, x=hand_local[8][0], y=hand_local[8][1])
                # elif gesture_str == "two":
                #     return ControlEvent(ControlEvent.MOVE_TO, x=hand_local[8][0], y=hand_local[8][1])
        if len(finger_tips) == 2:
            x_ = abs(finger_tips[0][0]-finger_tips[1][0])
            y_ = abs(finger_tips[0][1]-finger_tips[1][1])
            print("resize:", x_,y_)
            # 创建一个 500x500 的全白图像
            image = np.ones((500, 500, 3), dtype=np.uint8) * 255

            # 计算矩形的左上角和右下角坐标
            top_left_x = int(250 - x_ // 2)
            top_left_y = int(250 - y_ // 2)
            bottom_right_x = int(top_left_x + x_)
            bottom_right_y = int(top_left_y + y_)

            # 在图像上绘制矩形
            image = cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 3)

            # 假设 logger 是一个可以记录图像的对象
            logger.image = image

            # 返回一个控制事件和一个参数字典
            x_ = x_/250 # 大于1表示放大，小于1表示缩小
            y_ = y_/250 # 大于1表示放大，小于1表示缩小
            return ControlEvent(ControlEvent.RESIZE, x=x_, y=y_), {"hand_position": finger_tips}
    return None, {}

class HandTrackerFunction():
    BASE_METHOD = 'mediapipe'

    @staticmethod
    def get_tracker_function(tracker_name):
        if tracker_name == HandTrackerFunction.BASE_METHOD:
            return hands_tracker
        else:
            raise ValueError(f"Tracker {tracker_name} not found")