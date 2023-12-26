import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from model import SimpleNN, generate_dataset, train, train_batch, save_model, device


# 实例化模型、损失函数和优化器
model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # 例如，从1e-3降到1e-4

# 生成训练数据
train_inputs, train_targets = generate_dataset(1000000, 4)
test_inputs, test_targets = generate_dataset(100000, 4)

# 使用最小-最大缩放进行标准化
min_val, _ = torch.min(train_inputs, dim=0)
max_val, _ = torch.max(train_inputs, dim=0)
train_inputs = (train_inputs - min_val) / (max_val - min_val)
test_inputs = (test_inputs - min_val) / (max_val - min_val)
#train_mean = train_inputs.mean(dim=0, keepdim=True)
#train_std = train_inputs.std(dim=0, keepdim=True)
#train_inputs = (train_inputs - train_mean) / train_std
#test_inputs = (test_inputs - train_mean) / train_std


# 训练模型
train_batch(model, criterion, optimizer, train_inputs, train_targets, epochs=2000, batch_size=10000)

# 保存模型
save_model(model, 'model.pth')

# 测试模型
def test(model, inputs, targets):
    model.eval()
    inputs, targets = inputs.to(device), targets.to(device)
    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == targets).sum().item()  # 直接比较预测和真实标签
        total = targets.size(0)
        accuracy = 100 * correct / total
        print(f'Accuracy of the network on the test inputs: {accuracy:.2f}%')

# 测试模型
test(model, test_inputs, test_targets)

