import numpy as np
from math import erf,sqrt

def sigmoid(x):
    return 1/(1+np.exp(-x));

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x >= 0, x, alpha * x)

def elu(x, alpha=1.0):
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

def swish(x, beta=1):
    return x * sigmoid(beta * x)


def softmax(x):
    """Softmax 激活函数"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def softplus(x):
    """SoftPlus 激活函数"""
    return np.log(1 + np.exp(x))


def maxout(x, feature_maps=2):
    """Maxout 激活函数 (简化版用于可视化)

    Parameters:
    -----------
    x : array-like
        输入张量
    feature_maps : int, default=2
        特征图的数量，默认为2

    Note: 这是一个为了可视化而简化的实现，仅展示maxout的基本思想
    """
    # 为了可视化，我们可以通过对x应用不同的线性变换来模拟特征图
    # 创建feature_maps个不同的线性变换
    transformed = []
    for i in range(feature_maps):
        # 使用不同的斜率和偏移来创建不同的线性变换
        slope = 0.5 + i  # 每个特征图有不同的斜率
        offset = -i  # 每个特征图有不同的偏移
        transformed.append(slope * x + offset)

    # 将所有变换堆叠起来，形状变为 (feature_maps, len(x))
    stacked = np.stack(transformed, axis=0)

    # 在第一个维度上取最大值（即在所有特征图中选择最大的值）
    return np.max(stacked, axis=0)

def mish(x):
    """Mish 激活函数"""
    return x * np.tanh(np.log(1 + np.exp(x)))

def gelu(x):
    """GELU 激活函数的近似版本"""
    # This approximation doesn't use erf and should work better with arrays
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * np.power(x, 3))))

def prelu(x, alpha=0.25):
    """PReLU 激活函数 (Parametric ReLU)"""
    return np.where(x >= 0, x, alpha * x)
