import numpy as np
import math


def gaussKernel(sig, m=None):
    """
    高斯滤波核函数

    参数:
    - sig: 高斯函数的标准差
    - m: 高斯滤波核的大小 (可选)

    返回:
    - w: 高斯滤波核，一个二维 NumPy 数组
    """
    if m is None:
        # 根据 sigma 计算 m
        m = 2 * math.ceil(3 * sig) + 1
        print("警告：未提供 m 的值。根据 sigma 的计算结果为 m 分配了默认值:", m)
    elif m % 2 == 0:
        print("警告：提供的 m 值为偶数，请使用奇数值以获得正确的中心位置。")
        m += 1
        print("已自动调整 m 的值为:", m)

    # 创建空白的滤波核
    w = np.zeros((m, m))

    # 计算滤波核的中心坐标
    center = (m - 1) / 2

    # 计算高斯滤波核的每个元素的值
    for i in range(m):
        for j in range(m):
            x = i - center
            y = j - center
            w[i, j] = math.exp(-(x**2 + y**2) / (2 * sig**2))

    # 归一化滤波核
    w /= np.sum(w)

    return w

sigma = 1
m = 3

print("输入sigma：1，m=3") 
w = gaussKernel(sigma, m)
print("高斯滤波核：")
print(w)