import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math


def rgb1gray(f, method='NTSC'):
    """
    将彩色 RGB 图像转换为灰度图像。

    参数：
        f: numpy 数组，表示彩色图像，形状为 (H, W, 3)。
        method: 字符串，指定转换方法。可选值为 'average' 和 'NTSC'，默认为 'NTSC'。

    返回：
        gray: numpy 数组，表示灰度图像，形状为 (H, W)。

    """
    try:

        if method == 'average':
            gray = np.mean(f, axis=2, keepdims=True) # 取平均值方法
        elif method == 'NTSC':
            weights = np.array([0.2989, 0.5870, 0.1140]) # NTSC 标准方法
            gray = np.dot(f, weights) # 矩阵乘法
        else:
            raise ValueError("ERROR02:错误参数。支持的参数为 'average'和 'NTSC'")

    except ValueError as e:
        print("ERROR02: 参数值错误")
        raise e

    return gray.astype(np.uint8)

def twodConv(f, w, padding='zero'):
    """
    二维卷积函数

    参数:
    - f: 输入图像，一个二维 NumPy 数组
    - w: 卷积核，一个二维 NumPy 数组
    - padding: 填补选项，可选值为 'zero'（默认）和 'replicate'

    返回:
    - g: 卷积结果，一个二维 NumPy 数组
    """
    pad_width = (w.shape[0] - 1) // 2  # 填补宽度

    f_padded = pad_image(f, pad_width, padding)

    g = np.zeros_like(f)

    for i in range(1, f.shape[0] + 1):
        for j in range(1, f.shape[1] + 1):
            g[i - 1, j - 1] = np.sum(f_padded[i - 1:i + w.shape[0] - 1, j - 1:j + w.shape[1] - 1] * w)

    return g

def pad_image(image, pad_width, padding='zero'):
    """
    对图像进行填补

    参数:
    - image: 输入图像，一个二维 NumPy 数组
    - pad_width: 填补宽度
    - padding: 填补选项，可选值为 'zero'（默认）和 'replicate'

    返回:
    - padded_image: 填补后的图像，一个二维 NumPy 数组
    """
    rows, cols = image.shape
    padded_image = np.zeros((rows + 2 * pad_width, cols + 2 * pad_width), dtype=image.dtype)
    padded_image[pad_width:pad_width + rows, pad_width:pad_width + cols] = image

    if padding == 'replicate':
        # 复制边界像素
        # 复制行
        for i in range(pad_width):
            padded_image[i, pad_width:pad_width + cols] = image[0, :]
            padded_image[-i - 1, pad_width:pad_width + cols] = image[-1, :]

        # 复制列
        for j in range(pad_width):
            padded_image[pad_width:pad_width + rows, j] = image[:, 0]
            padded_image[pad_width:pad_width + rows, -j - 1] = image[:, -1]

    elif padding != 'zero':
        raise ValueError("Invalid padding option. Supported options are 'replicate' and 'zero'.")

    return padded_image

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

# 加载图像
cameraman_g = np.array(Image.open('cameraman.tif'))
einstein_g = np.array(Image.open('einstein.tif'))
mandril_color = np.array(Image.open('mandril_color.tif'))
lena512color = np.array(Image.open('lena512color.tiff'))


# NTSC灰度化
mandril_color_g = rgb1gray(mandril_color, method='NTSC')
lena512color_g = rgb1gray(lena512color, method='NTSC')

# 高斯滤波参数
sigmas = [1, 2, 3, 5]

# 像素填补选项
padding = 'replicate'

# 创建一个子图网格
num_rows = len(sigmas)  # 网格的行数
num_cols = 4  # 网格的列数（用于显示4张图片）

# 创建子图网格
fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 12))

# 遍历不同的 sigma 值并绘制图片
for i, sigma in enumerate(sigmas):
    # 生成高斯滤波核
    w = gaussKernel(sigma)

    # 对图像应用高斯滤波
    cameraman_filtered = twodConv(cameraman_g, w, padding)
    einstein_filtered = twodConv(einstein_g, w, padding)
    lena512color_filtered = twodConv(lena512color_g, w, padding)
    mandril_color_filtered = twodConv(mandril_color_g, w, padding)

    # 绘制滤波后的图片
    axs[i, 0].imshow(cameraman_filtered, cmap='gray')
    axs[i, 0].set_title('Cameraman (σ={})'.format(sigma))

    axs[i, 1].imshow(einstein_filtered, cmap='gray')
    axs[i, 1].set_title('Einstein (σ={})'.format(sigma))

    axs[i, 2].imshow(lena512color_filtered, cmap='gray')
    axs[i, 2].set_title('Lena512color (σ={})'.format(sigma))

    axs[i, 3].imshow(mandril_color_filtered, cmap='gray')
    axs[i, 3].set_title('Mandril_color (σ={})'.format(sigma))

# 移除空的子图
if num_rows * num_cols > len(sigmas) * 4:
    for j in range(len(sigmas) * 4, num_rows * num_cols):
        fig.delaxes(axs[j // num_cols, j % num_cols])

# 调整子图之间的间距
plt.tight_layout()

# 显示绘图
plt.show()