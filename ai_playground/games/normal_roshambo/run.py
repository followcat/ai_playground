import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from model import RPSNet, generate_dataset, train_batch, save_model, device


# 实例化模型、损失函数和优化器
model = RPSNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 生成训练数据
num_samples = 1000  # 我们想生成的样本数量
sequence_length = 1  # 我们想要的序列长度

train_inputs, train_targets = generate_dataset(10000, sequence_length)
test_inputs, test_targets = generate_dataset(1000, sequence_length)
                  
# 训练模型
train_batch(model, criterion, optimizer, train_inputs, train_targets, epochs=1000, batch_size=1000)

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
 
