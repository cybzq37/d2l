import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成训练数据
x = torch.linspace(0, 2*np.pi, 1000).unsqueeze(1)  # 输入x
y = torch.sin(x)  # 目标标签sin(x)

# 2. 定义单隐藏层ReLU网络
class SinNet(nn.Module):
    def __init__(self, hidden_size=50):
        super().__init__()
        self.layer1 = nn.Linear(1, hidden_size)  # 隐藏层，50个ReLU神经元
        self.layer2 = nn.Linear(hidden_size, 1)  # 输出层，加权求和

    def forward(self, x):
        hidden = torch.relu(self.layer1(x))  # ReLU激活
        output = self.layer2(hidden)
        return output

# 3. 训练
model = SinNet(hidden_size=50)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(5000):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# 4. 绘制拟合结果
y_pred = model(x).detach().numpy()
plt.plot(x, y, label="真实sin(x)")
plt.plot(x, y_pred, label="ReLU网络拟合", linestyle="--")
plt.legend()
plt.show()