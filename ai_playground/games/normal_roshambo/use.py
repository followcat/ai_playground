import sys

import torch
import numpy as np

from model import RPSNet, predict


if __name__ == '__main__':
    # 加载模型
    model = RPSNet()
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    # 读取命令行参数
    if len(sys.argv) != 3:
        print("Usage: python use.py <number1> <number2>")
        sys.exit(1)

    try:
        num1 = float(sys.argv[1])
        num2 = float(sys.argv[2])
    except ValueError:
        print("Both arguments should be numbers.")
        sys.exit(1)

    categories = predict(model, num1, num2)
    print(f'预测结果: {categories}')
 
