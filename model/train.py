import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import tkinter as tk
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
print(positions.shape, face_landmarks.shape)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(face_landmarks, positions, test_size=0.1, random_state=42)

# 转换为PyTorch张量
X_train_tensor = torch.tensor(face_landmarks).float()
y_train_tensor = torch.tensor(positions).float()

X_train_tensor = torch.tensor(X_train).float()
y_train_tensor = torch.tensor(y_train).float()
X_test_tensor = torch.tensor(X_test).float()
y_test_tensor = torch.tensor(y_test).float()

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 构建模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(X_train_tensor.shape[1], 128)
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

model = Net()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.01)

print(model)
# 训练模型
model.train()
min_loss = 1e10
for epoch in range(100):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    scheduler.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    if loss.item() < min_loss:
        min_loss = loss.item()
        torch.save(model.state_dict(), f'model/{user_config["name"]}/best_model.pth')

# 加载最佳模型
model.load_state_dict(torch.load(f'model/{user_config["name"]}/best_model.pth'))

# 评估模型
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    for i in range(len(predictions)):
        print(predictions[i], y_test_tensor[i])
    test_loss = criterion(predictions, y_test_tensor)
print(f'Test loss: {test_loss.item()}')
