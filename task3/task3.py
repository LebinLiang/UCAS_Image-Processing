import cv2
import numpy as np
import matplotlib.pyplot as plt


# 反谐波均值滤波器函数
def contra_harmonic_mean_filter(image, mask_size, Q):
    # 计算需要进行填充的边界大小
    border_size = (mask_size - 1) // 2

    # 对图像进行边界填充
    padded_image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_REPLICATE)

    # 创建输出图像
    filtered_image = np.zeros_like(image, dtype=np.float32)

    # 对每个像素应用滤波器
    for i in range(border_size, image.shape[0] + border_size):
        for j in range(border_size, image.shape[1] + border_size):
            # 提取滤波器窗口
            window = padded_image[i - border_size: i + border_size + 1, j - border_size: j + border_size + 1]

            # 计算反谐波均值
            numerator = np.sum(np.power(window, Q + 1))
            denominator = np.sum(np.power(window, Q))
            
            # 检查分母是否接近零
            if np.isclose(denominator, 0):
                filtered_image[i - border_size, j - border_size] = 0
            else:
                filtered_image[i - border_size, j - border_size] = numerator / denominator

    # 将输出图像转换为 8 位无符号整数类型
    filtered_image = cv2.convertScaleAbs(filtered_image)

    return filtered_image

# 谐波均值滤波器函数
def harmonic_mean_filter(image, mask_size):
    result_image = np.copy(image)
    pad_size = mask_size // 2

    # 循环遍历图像像素
    for i in range(pad_size, image.shape[0] - pad_size):
        for j in range(pad_size, image.shape[1] - pad_size):
            values = []

            # 创建基于掩模大小的像素窗口
            for m in range(-pad_size, pad_size + 1):
                for n in range(-pad_size, pad_size + 1):
                    values.append(1 / (image[i + m, j + n] + 1e-18))

            # 使用谐波均值滤波器公式计算结果
            result_image[i, j] = (mask_size**2) / np.sum(values)

    return result_image

# 自适应中值滤波器函数
def adaptive_median_filter(img, max_size=7):
    # 图像类型检查和转换为灰度图像
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img.copy()

    # 生成输出图像容器
    m, n = gray_img.shape
    Nmax = max_size // 2
    result_image = np.zeros_like(gray_img)

    # 循环遍历图像像素
    for i in range(Nmax, m - Nmax):
        for j in range(Nmax, n - Nmax):
            window_size = 3
            while window_size <= max_size:
                window = gray_img[i - window_size // 2:i + window_size // 2 + 1,
                                  j - window_size // 2:j + window_size // 2 + 1]

                # 使用排序数组来计算中值
                sorted_window = np.sort(window.flatten())
                median_index = len(sorted_window) // 2
                median = sorted_window[median_index]

                # 检查中值是否在窗口范围内
                if sorted_window[0] < median < sorted_window[-1]:
                    # 检查当前像素值是否为脉冲噪声
                    if sorted_window[0] < gray_img[i, j] < sorted_window[-1]:
                        result_image[i, j] = gray_img[i, j]
                    else:
                        result_image[i, j] = median
                    break
                else:
                    window_size += 2

    return result_image

# 计算 MSE 和 SNR 的函数
def calculate_metrics(original_image, processed_image):
    mse = np.mean((original_image - processed_image) ** 2)
    original_energy = np.sum(original_image ** 2)
    noise_energy = np.sum((original_image - processed_image) ** 2)
    snr = 10 * np.log10(original_energy / noise_energy)

    return mse, snr

# 读取图像
fig01 = cv2.imread('Fig01.tif', cv2.IMREAD_GRAYSCALE)
fig02 = cv2.imread('Fig02.tif', cv2.IMREAD_GRAYSCALE)
fig03 = cv2.imread('Fig03.tif', cv2.IMREAD_GRAYSCALE)
fig04 = cv2.imread('Fig04.tif', cv2.IMREAD_GRAYSCALE)

# 对图像应用滤波器
# 对第一个图像使用反谐波均值滤波
filtered_fig02_pepper = contra_harmonic_mean_filter(fig02, mask_size=3, Q=1.5)
# 对第二个图像使用谐波均值滤波
filtered_fig03_salt = harmonic_mean_filter(fig03, mask_size=3)
# 对第三个图像使用自适应中值滤波
filtered_fig04 = adaptive_median_filter(fig04, 8)

# 计算评估指标
mse_fig02_pepper, snr_fig02_pepper = calculate_metrics(fig01, filtered_fig02_pepper)
mse_fig03_salt, snr_fig03_salt = calculate_metrics(fig01, filtered_fig03_salt)
mse_fig04, snr_fig04 = calculate_metrics(fig01, filtered_fig04)

# 显示原始图像和滤波后的图像，并在标题中添加 MSE 和 SNR
plt.figure(figsize=(15, 10))

# 显示原始图像 Fig01
plt.subplot(421), plt.imshow(fig01, cmap='gray'), plt.title('Original Fig01')

# 显示原始图像 Fig02 和 滤波后的 Fig02
plt.subplot(423), plt.imshow(fig02, cmap='gray'), plt.title('Original Fig02')
plt.subplot(424), plt.imshow(filtered_fig02_pepper, cmap='gray'), plt.title(f'Filtered Fig02\nMSE: {mse_fig02_pepper:.4f}, SNR: {snr_fig02_pepper:.2f} dB')

# 显示原始图像 Fig03 和 滤波后的 Fig03
plt.subplot(425), plt.imshow(fig03, cmap='gray'), plt.title('Original Fig03')
plt.subplot(426), plt.imshow(filtered_fig03_salt, cmap='gray'), plt.title(f'Filtered Fig03\nMSE: {mse_fig03_salt:.4f}, SNR: {snr_fig03_salt:.2f} dB')

# 显示原始图像 Fig04 和 滤波后的 Fig04
plt.subplot(427), plt.imshow(fig04, cmap='gray'), plt.title('Original Fig04')
plt.subplot(428), plt.imshow(filtered_fig04, cmap='gray'), plt.title(f'Filtered Fig04\nMSE: {mse_fig04:.4f}, SNR: {snr_fig04:.2f} dB')

plt.tight_layout()
plt.show()
