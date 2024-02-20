import numpy as np
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1434, 128)
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



def load_model(model_path):
    # Load the model from the file
    model = Net()
    model.load_state_dict(torch.load(model_path))

    # Set the model to evaluation mode
    model.eval()

    return model



def inference(model, face_landmarks):
    # Convert the face landmarks to a PyTorch tensor
    x = []
    for lm in face_landmarks:
        x.append(lm['x'])
        x.append(lm['y'])
        x.append(lm['z'])
    face_landmarks_tensor = torch.tensor([x]).float()

    # Perform the inference
    with torch.no_grad():
        predicted_position = model(face_landmarks_tensor)

    return predicted_position
