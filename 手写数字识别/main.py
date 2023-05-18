import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from flask import Flask, jsonify, render_template, request


# 加载模型
class BP(nn.Module):
    def __init__(self):
        super(BP, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 将输入展平为一维向量
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# 加载训练好的模型
BP_model = BP()
BP_model.load_state_dict(torch.load("bp.ckpt"))
BP_model.eval()

CNN_model = CNN()
CNN_model.load_state_dict(torch.load("cnn.ckpt"))
CNN_model.eval()
# webapp
app = Flask(__name__)


def BP_predict(input):
    with torch.no_grad():
        input = torch.tensor(input, dtype=torch.float32)
        output = BP_model(input)
        probabilities = torch.softmax(output, dim=1)
        return probabilities[0].tolist()


def CNN_predict(input):
    with torch.no_grad():
        input = torch.tensor(input, dtype=torch.float32)
        output = CNN_model(input)
        probabilities = torch.softmax(output, dim=1)
        return probabilities[0].tolist()


@app.route("/api/mnist", methods=["POST"])
def mnist():
    input2 = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(
        1, 1, 28, 28
    )
    input1 = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784)
    output1 = BP_predict(input1)
    output2 = CNN_predict(input2)
    return jsonify(results=[output1, output2])


@app.route("/")
def main():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0")
