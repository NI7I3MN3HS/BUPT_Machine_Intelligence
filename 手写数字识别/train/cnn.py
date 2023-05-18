import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义神经网络模型


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


# 设置训练参数
batch_size = 64
learning_rate = 0.003
num_epochs = 10

# 加载MNIST数据集
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
train_dataset = datasets.MNIST(
    root="./data", train=True, transform=transform, download=True
)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False
)

# 初始化模型并将其移动到GPU上
model = CNN().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 训练模型
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 将数据移动到GPU上
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item()}"
            )

# 在测试集上评估模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    tp = 0
    fp = 0
    fn = 0
    class_correct = [0] * 10  # 用于存储每个数字的正确预测数
    class_total = [0] * 10  # 用于存储每个数字的总样本数
    for images, labels in test_loader:
        # 将数据移动到GPU上
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        tp += ((predicted == labels) & (labels == predicted)).sum().item()
        fp += ((predicted != labels) & (labels != predicted)).sum().item()
        fn += ((predicted != labels) & (labels == predicted)).sum().item()

        # 统计每个数字的正确预测数和总样本数
        for i in range(len(labels)):
            label = labels[i]
            prediction = predicted[i]
            class_correct[label] += (prediction == label).item()
            class_total[label] += 1

    # 计算每个数字的准确度
    for i in range(10):
        accuracy = 100 * class_correct[i] / class_total[i]
        print(f"Digit {i}:")
        print(f"Accuracy: {accuracy}%")

    # 计算整体准确度
    accuracy = 100 * correct / total
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    print("Overall:")
    print(f"Accuracy: {accuracy}%")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")


torch.save(model.state_dict(), "cnn.ckpt")
