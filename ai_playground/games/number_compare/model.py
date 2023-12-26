import torch
import torch.nn as nn
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# 定义模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # 增加更多的层和/或神经元
        self.fc1 = nn.Linear(1, 32)  # 增加第一层的神经元数量
        self.fc2 = nn.Linear(32, 64)  # 添加一个额外的隐藏层
        self.fc3 = nn.Linear(64, 3)  # 输出层保持不变
        self.relu = nn.ReLU()

    def forward(self, x):
        # 计算两个数的差值
        diff = (x[:, 0] - x[:, 1]).view(-1, 1)
        # 经过全连接层和ReLU激活函数
        x = self.relu(self.fc1(diff))
        x = self.relu(self.fc2(x))  # 经过第二个全连接层
        x = self.fc3(x)  # 经过第三个全连接层得到输出
        return x

#class SimpleNN(nn.Module):
#    def __init__(self):
#        super(SimpleNN, self).__init__()
#        self.fc1 = nn.Linear(1, 128)
#        self.bn1 = nn.BatchNorm1d(128)
#        self.dropout1 = nn.Dropout(0.5)
#        self.fc2 = nn.Linear(128, 256)
#        self.bn2 = nn.BatchNorm1d(256)
#        self.dropout2 = nn.Dropout(0.5)
#        self.fc3 = nn.Linear(256, 128)
#        self.bn3 = nn.BatchNorm1d(128)
#        self.dropout3 = nn.Dropout(0.5)
#        self.fc4 = nn.Linear(128, 3)
#        self.relu = nn.ReLU()
#
#    def forward(self, x):
#        diff = (x[:, 0] - x[:, 1]).view(-1, 1)
#        x = self.relu(self.bn1(self.fc1(diff)))
#        x = self.dropout1(x)
#        x = self.relu(self.bn2(self.fc2(x)))
#        x = self.dropout2(x)
#        x = self.relu(self.bn3(self.fc3(x)))
#        x = self.dropout3(x)
#        x = self.fc4(x)
#        return x

# 生成数据集的函数
def generate_dataset(num_samples, max_exponent):
    # 生成随机数作为输入
    inputs = torch.rand(num_samples, 2) * 10**max_exponent - 8**max_exponent
    targets = []

    # 添加剩余的"小于"和"大于"类别样本
    for i in range(num_samples):
        if inputs[i, 0] < inputs[i, 1]:
            targets.append(0)  # 类别0：小于
        elif inputs[i, 0] > inputs[i, 1]:
            targets.append(2)  # 类别2：大于

    small_inputs = (torch.randint(
        -10**max_exponent, 10**max_exponent,
        (num_samples, 2)).float() / 10**(max(1, max_exponent-3)))
    for i in range(num_samples):
       if small_inputs[i, 0] < inputs[i, 1]:
           targets.append(0)  # 类别0：小于
       elif small_inputs[i, 0] > inputs[i, 1]:
           targets.append(2)  # 类别2：大于
       elif small_inputs[i, 0] == inputs[i, 1]:
           targets.append(1)  # 类别1：等于
    inputs = torch.cat((inputs, small_inputs), dim=0)

    # 添加"等于"类别样本
    num_equal_samples = int(num_samples / 3)
    equal_nums = torch.rand(num_equal_samples) * 10**max_exponent - 10**max_exponent
    equal_inputs = torch.stack((equal_nums, equal_nums), dim=1)
    inputs = torch.cat((inputs, equal_inputs), dim=0)
    targets.extend([1] * num_equal_samples)  # 类别1：等于

    targets = torch.tensor(targets, dtype=torch.long)
    inputs, targets = inputs.to(device), targets.to(device)
    return inputs, targets


def train_batch(model, criterion, optimizer, inputs, targets, epochs, batch_size):
    model.train()
    num_samples = inputs.size(0)
    num_batches = (num_samples + batch_size - 1) // batch_size  # 计算总的批次数

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx in range(num_batches):
            # 计算批次的起始和结束索引
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size

            # 获取当前批次的数据和标签
            inputs_batch = inputs[start_idx:end_idx].to(device)
            targets_batch = targets[start_idx:end_idx].to(device)

            # 清空梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs_batch)
            loss = criterion(outputs, targets_batch)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 累加损失
            epoch_loss += loss.item()

        # 计算平均损失
        epoch_loss /= num_batches
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')



# 训练模型的函数
def train(model, criterion, optimizer, inputs, targets, epochs):
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')

# 保存模型的函数
def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)
    print(f'Model saved to {file_path}')

# 加载模型的函数
def load_model(file_path):
    model = SimpleNN()
    model.load_state_dict(torch.load(file_path))
    model.eval()
    return model

# 使用模型进行预测的函数
def predict(model, num1, num2):
    with torch.no_grad():
        inputs = torch.tensor([[num1, num2]])
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
    categories = ['小于', '等于', '大于']
    return categories[predicted.item()]

