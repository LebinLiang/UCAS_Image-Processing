import numpy as np

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

    f_padded = pad_image(f, pad_width, padding) #调用自定义pad函数

    print(f_padded)

    g = np.zeros_like(f) #输出图像与输入图像尺寸一样

    for i in range(1, f.shape[0] + 1):  #遍历图像元素进行卷积操作
        for j in range(1, f.shape[1] + 1):
            g[i - 1, j - 1] = np.sum(f_padded[i - 1:i + w.shape[0] - 1, j - 1:j + w.shape[1] - 1] * w) #根据卷积核大小来提取子图

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
    rows, cols = image.shape  # 获取图像尺寸
    padded_image = np.zeros((rows + 2 * pad_width, cols + 2 * pad_width), dtype=image.dtype) # 根据填补宽度，创建一个全零数组
    padded_image[pad_width:pad_width + rows, pad_width:pad_width + cols] = image # 将原图像复制到中心区域

    if padding == 'replicate': #如果是复制边界
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


f = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
w = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

print("原数组:")
print(f)
print("【replicate】pad后数组:")
g_replicate = twodConv(f, w, padding='replicate')
print("卷积后后数组:")
print(g_replicate)

print("【zero】pad后数组:")
g_zero = twodConv(f, w, padding='zero')
print("卷积后后数组:")
print(g_zero)