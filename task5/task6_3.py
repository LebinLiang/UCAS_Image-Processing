import cv2
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


# 定义sigma值
sigma = 7

# 使用OpenCV加载图像
image_filenames = ['mandril_color.tif', 'lena512color.tiff']
images = [cv2.imread(filename, cv2.IMREAD_GRAYSCALE) for filename in image_filenames]

# 使用自定义函数进行高斯模糊
filtered_images_zero = []
filtered_images_replicate = []
for image in images:
    kernel = gaussKernel(sigma) #设置高斯核
    filtered_image_zero = twodConv(image, kernel,padding="zero") #补0
    filtered_image_replicate = twodConv(image, kernel,padding="replicate") #像素复制
    filtered_images_replicate.append(filtered_image_replicate) #处理后的图像进入队列
    filtered_images_zero.append(filtered_image_zero)

# 计算差异图像
diff_images = [np.abs(custom_img - opencv_img) for custom_img, opencv_img in zip(filtered_images_zero, filtered_images_replicate)]


# 创建画布用于显示图像
plt.figure(figsize=(12, 8))

for i in range(len(images)):
    # 绘制原始图像
    plt.subplot(4, 4, i*4 + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title('raw')

    # 绘制zero滤波结果
    plt.subplot(4, 4, i*4 + 2)
    plt.imshow(filtered_images_zero[i], cmap='gray')
    plt.title('zero')

    # 绘制replicate滤波结果
    plt.subplot(4, 4, i*4 + 3)
    plt.imshow(filtered_images_replicate[i], cmap='gray')
    plt.title('replicate')

    # 绘制差异图像
    plt.subplot(4, 4, i*4 + 4)
    plt.imshow(diff_images[i], cmap='gray')
    mse = np.mean(diff_images[i]**2)
    similarity = 1 / mse
    print(image_filenames[i]+":") 
    print(similarity)
    plt.title('diff')

plt.tight_layout()
plt.show()