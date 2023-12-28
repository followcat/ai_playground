import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# 定义模型
class RPSNet(nn.Module):
    def __init__(self):
        super(RPSNet, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

 
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


# 使用模型进行预测的函数
def predict(model, vs, our):
    with torch.no_grad():
        inputs = torch.tensor([[vs, our]], dtype=torch.float32)
        outputs = model(inputs)
        probabilities = torch.softmax(outputs, dim=1)
        distribution = Categorical(probabilities)
        sampled_move = distribution.sample()
    categories = ['石头', '剪刀', '布']
    print("预测的概率分布：", probabilities)
    return categories[sampled_move]
  

def generate_dataset(num_samples, sequence_length):
    """
    生成猜拳游戏的数据集。

    :param num_samples: 生成的样本数量
    :param sequence_length: 每个样本的序列长度
    :return: 返回一个元组，包含对手的出拳、玩家的出拳和每一轮的结果
    """
    def generate_rps_result(opponent_move, player_move):
       # 猜拳游戏的规则：石头胜剪刀，剪刀胜布，布胜石头
       if opponent_move == player_move:
           return 1  # 平局
       elif (opponent_move - player_move) % 3 == 1:
           return 0  # 玩家输
       else:
           return 2  # 玩家赢
 
    # 初始化列表来存储对手和玩家的出拳，以及结果
    opponent_history = []
    player_history = []
    result_history = []

    for _ in range(num_samples + sequence_length):
        # 随机生成对手和玩家的出拳
        opponent_move = random.randint(0, 2)  # 0: 石头, 1: 剪刀, 2: 布
        player_move = random.randint(0, 2)

        # 计算结果
        result = generate_rps_result(opponent_move, player_move)

        # 添加到历史记录中
        opponent_history.append(opponent_move)
        player_history.append(player_move)
        result_history.append(result)

    # 将历史记录转换为训练数据
    X = []
    y = []
    for i in range(num_samples):
        # 提取序列数据
        opponent_sequence = opponent_history[i:i + sequence_length]
        player_sequence = player_history[i:i + sequence_length]

        # 将对手和玩家的出拳合并为一个序列
        sequence = opponent_sequence + player_sequence
        X.append(sequence)

        # 结果是序列之后的下一次对手的出拳
        y.append(opponent_history[i + sequence_length])

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

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
 
