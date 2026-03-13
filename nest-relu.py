import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------
# 1. 准备数据：生成输入 x（范围 -5 到 5，足够密集）
# ------------------------------------------------------
x = np.linspace(-5, 5, 1000)  # 1000个点，保证图形平滑

# ------------------------------------------------------
# 2. 定义不同层数的 ReLU 函数
# ------------------------------------------------------
def relu_single(x, w, b):
    """单层 ReLU: max(0, w*x + b)"""
    linear = w * x + b
    return np.maximum(0, linear)

def relu_double(x, w1, b1, w2, b2):
    """2层嵌套 ReLU: max(0, w2 * max(0, w1*x + b1) + b2)"""
    layer1 = np.maximum(0, w1 * x + b1)
    layer2 = w2 * layer1 + b2
    return np.maximum(0, layer2)

def relu_triple(x, w1, b1, w2, b2, w3, b3):
    """3层嵌套 ReLU: max(0, w3 * max(0, w2 * max(0, w1*x + b1) + b2) + b3)"""
    layer1 = np.maximum(0, w1 * x + b1)
    layer2 = np.maximum(0, w2 * layer1 + b2)
    layer3 = w3 * layer2 + b3
    return np.maximum(0, layer3)

# ------------------------------------------------------
# 3. 设置参数（特意选了让拐点明显的参数，方便观察）
# ------------------------------------------------------
# 单层参数：拐点在 x=0.5 (w1*x + b1 = 0 → x=0.5)
params_single = {'w': 2, 'b': -1}

# 两层参数：先做标准ReLU，再翻转+上移，形成一个三角形
params_double = {'w1': 1, 'b1': 0, 'w2': -1, 'b2': 2}

# 三层参数：在两层三角形的基础上，再做一次截断+平移
params_triple = {'w1': 1, 'b1': 0, 'w2': -1, 'b2': 2, 'w3': 1, 'b3': -0.5}

# ------------------------------------------------------
# 4. 计算各层的输出 y
# ------------------------------------------------------
y_single = relu_single(x, **params_single)
y_double = relu_double(x, **params_double)
y_triple = relu_triple(x, **params_triple)

# ------------------------------------------------------
# 5. 画图（1行3列）
# ------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# --- 子图1：单层 ReLU ---
ax1.plot(x, y_single, label=r'$y=\max(0, 2x-1)$', color='#1f77b4', linewidth=2.5)
ax1.axhline(0, color='k', linestyle='--', alpha=0.3)  # x轴
ax1.axvline(0, color='k', linestyle='--', alpha=0.3)  # y轴
ax1.axvline(0.5, color='r', linestyle=':', label='拐点 x=0.5')  # 标注拐点
ax1.set_title('单层 ReLU', fontsize=14, fontweight='bold')
ax1.set_xlabel('输入 x', fontsize=12)
ax1.set_ylabel('输出 y', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.5, 4)

# --- 子图2：2层嵌套 ReLU ---
ax2.plot(x, y_double, label=r'$y=\max(0, -1 \cdot \max(0, x) + 2)$', color='#ff7f0e', linewidth=2.5)
ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
ax2.axvline(0, color='k', linestyle='--', alpha=0.3)
ax2.axvline(0, color='r', linestyle=':', label='拐点1 x=0')
ax2.axvline(2, color='g', linestyle=':', label='拐点2 x=2')
ax2.set_title('2层嵌套 ReLU', fontsize=14, fontweight='bold')
ax2.set_xlabel('输入 x', fontsize=12)
ax2.set_ylabel('输出 y', fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-0.5, 4)

# --- 子图3：3层嵌套 ReLU ---
ax3.plot(x, y_triple, label=r'$y=\max(0, 1 \cdot [2层输出] - 0.5)$', color='#2ca02c', linewidth=2.5)
ax3.axhline(0, color='k', linestyle='--', alpha=0.3)
ax3.axvline(0, color='k', linestyle='--', alpha=0.3)
ax3.axvline(0, color='r', linestyle=':', label='拐点1 x=0')
ax3.axvline(1.5, color='g', linestyle=':', label='拐点2 x=1.5')
ax3.set_title('3层嵌套 ReLU', fontsize=14, fontweight='bold')
ax3.set_xlabel('输入 x', fontsize=12)
ax3.set_ylabel('输出 y', fontsize=12)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.set_ylim(-0.5, 4)

# 调整布局，防止重叠
plt.tight_layout()
plt.show()