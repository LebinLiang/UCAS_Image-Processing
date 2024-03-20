import numpy as np
import cv2
import matplotlib.pyplot as plt

# 定义计算2D FFT的函数
def dft2D(f):
    # 对每一行进行一维FFT
    f_row = np.fft.fft(f, axis=1)
    
    # 对每一列进行一维FFT
    F = np.fft.fft(f_row, axis=0)
    
    return F

# 读取图像并进行必要的填充，确保尺寸是2的整数次幂
image_files = ['house.tif', 'house02.tif', 'lena_gray_512.tif', 'lunar_surface.tif', 'characters_test_pattern.tif']

# 创建一个5x2的子图布局
fig, axes = plt.subplots(5, 2, figsize=(10, 15))

for i, image_file in enumerate(image_files):
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    
    # 计算填充后的图像尺寸
    padded_height = 2**int(np.ceil(np.log2(image.shape[0])))
    padded_width = 2**int(np.ceil(np.log2(image.shape[1])))
    
    # 创建一个空白的填充后图像
    padded_image = np.zeros((padded_height, padded_width), dtype=float)
    
    # 将原始图像复制到填充后图像的左上角
    padded_image[:image.shape[0], :image.shape[1]] = image
    
    # 计算图像的二维傅里叶变换
    F = dft2D(padded_image)
    
    # 计算频谱的幅度（绝对值）
    magnitude = np.abs(F)
    
    # 显示原始图像
    axes[i, 0].imshow(image, cmap='gray')
    axes[i, 0].set_title('Original Image')
    axes[i, 0].axis('off')
    
    # 显示傅里叶变换的幅度谱
    axes[i, 1].imshow(np.log(1 + magnitude), cmap='gray')
    axes[i, 1].set_title('Spectrum Image (Log)')
    axes[i, 1].axis('off')

plt.tight_layout()
plt.show()



