import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib  # 用于模型保存
import json

user_config = json.load(open('config.json', 'r'))
# 假设您已经加载了数据并将其存储在变量data中
data = np.load(f'model/{user_config["name"]}/eye_tracking_data_multi_segment_movement.npz', allow_pickle=True)['data']
# 解析数据
positions = []
face_landmarks = []
for item in data:
    positions.append(item['target'])
    face_landmarks.append(item['train_data'] + item['landmarks'])
positions = np.array(positions)
face_landmarks = np.array(face_landmarks)
print(positions.shape)
print(face_landmarks.shape)
# 预处理数据
# 这里可以添加归一化或标准化的步骤
# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(face_landmarks, positions, test_size=0.1, random_state=42)

# 创建随机森林回归模型
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# 训练模型
rf.fit(X_train, y_train)

# 保存模型
joblib.dump(rf, f'model/{user_config["name"]}/random_forest_eye_tracking_model.joblib')

# 预测测试集
y_pred = rf.predict(X_test)

# 计算测试集上的均方误差
test_mse = mean_squared_error(y_test, y_pred)
print(f'Test MSE: {test_mse}')

# 打印预测值和真实值
for i in range(len(y_pred)):
    print(f'Predicted: {y_pred[i]}, Actual: {y_test[i]}')
