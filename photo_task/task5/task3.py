import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


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

# 加载图像
mandril_color = np.array(Image.open('mandril_color.tif'))
lena512color = np.array(Image.open('lena512color.tiff'))

# 转换为灰度图像
mandril_gray_average = rgb1gray(mandril_color, method='average')
mandril_gray_NTSC = rgb1gray(mandril_color, method='NTSC')

lena_gray_average = rgb1gray(lena512color, method='average')
lena_gray_NTSC = rgb1gray(lena512color, method='NTSC')

# 显示原图和灰度图像
plt.figure(figsize=(12, 10))

# 原图像1
plt.subplot(2, 3, 1)
plt.imshow(mandril_color)
plt.title('Original - mandril_color.tif')
plt.axis('off')

# 灰度图像1
plt.subplot(2, 3, 2)
plt.imshow(mandril_gray_average[:, :], cmap='gray')
plt.title('Average - mandril_color.tif')
plt.axis('off')

# 灰度图像2
plt.subplot(2, 3, 3)
plt.imshow(mandril_gray_NTSC[:, :], cmap='gray')
plt.title('NTSC - mandril_color.tif')
plt.axis('off')

# 原图像2
plt.subplot(2, 3, 4)
plt.imshow(lena512color)
plt.title('Original - lena512color.tiff')
plt.axis('off')

# 灰度图像3
plt.subplot(2, 3, 5)
plt.imshow(lena_gray_average[:, :], cmap='gray')
plt.title('Average - lena512color.tiff')
plt.axis('off')

# 灰度图像4
plt.subplot(2, 3, 6)
plt.imshow(lena_gray_NTSC[:, :], cmap='gray')
plt.title('NTSC - lena512color.tiff')
plt.axis('off')

plt.tight_layout()
plt.show()