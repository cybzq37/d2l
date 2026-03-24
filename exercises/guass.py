import numpy as np
import matplotlib.pyplot as plt

# 参数
mu = 0       # 均值
sigma = 1    # 标准差

# x 取值范围
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)

# 概率密度函数
y = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(- (x - mu)**2 / (2 * sigma**2))

# 画图
plt.figure()
plt.plot(x, y)
plt.title("Gaussian Distribution")
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.grid()

plt.show()
