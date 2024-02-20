import numpy as np
from scipy.optimize import minimize
import numpy as np
from screeninfo import get_monitors
import json

user_config = json.load(open('config.json', 'r'))
# 假设您已经加载了数据并将其存储在变量data中
data = np.load(f'model/{user_config["name"]}/eye_tracking_data_multi_segment_movement.npz', allow_pickle=True)['data']
# 解析数据
observed_x_offsets = []  # 观测到的x偏移
observed_y_offsets = []  # 观测到的y偏移
left_ray = []      # 左眼焦点x坐标
right_ray = []     # 右眼焦点x坐标
guess_screen_positions_x = []  # 猜测的屏幕x位置
guess_screen_positions_y = []  # 猜测的屏幕y位置

true_screen_positions_x = [] # 真实的屏幕x位置
true_screen_positions_y = [] # 真实的屏幕y位置

for item in data:
    left_ray.append(item['train_data'][0])
    right_ray.append(item['train_data'][1])
    observed_x_offsets.append(item['train_data'][2])
    observed_y_offsets.append(item['train_data'][3])
    guess_screen_positions_x.append(item['train_data'][4])
    guess_screen_positions_y.append(item['train_data'][5])
    
    true_screen_positions_x.append(item['target'][0])
    true_screen_positions_y.append(item['target'][1])

left_ray = np.array(left_ray)
right_ray = np.array(right_ray)
observed_x_offsets = np.array(observed_x_offsets)
observed_y_offsets = np.array(observed_y_offsets)
true_screen_positions_x = np.array(true_screen_positions_x)
true_screen_positions_y = np.array(true_screen_positions_y)
guess_screen_positions_x = np.array(guess_screen_positions_x)
guess_screen_positions_y = np.array(guess_screen_positions_y)

# 获取屏幕分辨率
screen_width = get_monitors()[0].width
screen_height = get_monitors()[0].height

# 定义损失函数
def loss_function(coefficients, observed_x_offsets, observed_y_offsets, left_ray, right_ray, true_screen_positions_x, true_screen_positions_y):
    focal_points_x = (left_ray * 1 + right_ray * 1) / 2 + 0
    predicted_screen_positions_x_offset = (observed_x_offsets - focal_points_x * coefficients[4]) * coefficients[5] + .5
    predicted_screen_positions_y_offset = -(observed_y_offsets + .15) * 3 + .5
    predicted_screen_positions_x = predicted_screen_positions_x_offset * screen_width
    predicted_screen_positions_y = predicted_screen_positions_y_offset * screen_height
    loss_x = np.sum((predicted_screen_positions_x - true_screen_positions_x) ** 2)
    loss_y = np.sum((predicted_screen_positions_y - true_screen_positions_y) ** 2)
    return loss_x + loss_y

# 初始系数猜测
initial_guess = [1, 1, 2, 0, 1.5, 3, 0.5, 0.15, 5, 0.5]

# 运行优化
result = minimize(
    loss_function,
    initial_guess,
    args=(observed_x_offsets, observed_y_offsets, left_ray, right_ray, true_screen_positions_x, true_screen_positions_y),
    method='L-BFGS-B',  # 选择一个适合你问题的优化算法
    options={'maxiter': 10000000, 'disp': True}
)

# 输出最佳系数
if result.success:
    fitted_coefficients = result.x
    print("Found coefficients:", fitted_coefficients)
else:
    raise ValueError(result.message)

# 尝试预测
for i in range(10):

    focal_points_x = (left_ray[i] * fitted_coefficients[0] + right_ray[i] * fitted_coefficients[1]) / fitted_coefficients[2] + fitted_coefficients[3]
    predicted_x_offset = (observed_x_offsets[i] - focal_points_x * fitted_coefficients[4]) * fitted_coefficients[5] + fitted_coefficients[6]
    predicted_y_offset = -(observed_y_offsets[i] + fitted_coefficients[7]) * fitted_coefficients[8] + fitted_coefficients[9]
    predicted_x = predicted_x_offset * screen_width
    predicted_y = predicted_y_offset * screen_height
    
    print(f'Predicted: ({predicted_x}, {predicted_y}), Actual: ({true_screen_positions_x[i]}, {true_screen_positions_y[i]}), guess: ({guess_screen_positions_x[i]}, {guess_screen_positions_y[i]})')

# 保存模型
np.savez(f'model/{user_config["name"]}/eye_tracking_calibration.npz', coefficients=fitted_coefficients)