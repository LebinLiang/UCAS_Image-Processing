import numpy as np
import cv2
import matplotlib.pyplot as plt

def dft2D(f):
    # 对每一行进行一维FFT
    f_row = np.fft.fft(f, axis=1)
    
    # 对每一列进行一维FFT
    F = np.fft.fft(f_row, axis=0)
    
    return F

def idft2D(F):
    # 获取频域图像的形状
    M, N = F.shape
    
    # 对每一列进行一维逆FFT
    f_row = np.fft.ifft(F, axis=0)
    
    # 对每一行进行一维逆FFT
    f = np.fft.ifft(f_row, axis=1)
    
    # 取实部部分，因为逆FFT的结果可能包含复数
    f = np.real(f)
    
    return f

# 创建一个空白的图像，大小为512x512，初始值为0（全黑）
image_size = 512
image = np.zeros((image_size, image_size), dtype=float)

# 定义矩形的参数：位于图像中心，60像素长，10像素宽
rect_width = 10
rect_height = 60
rect_center_x = image_size // 2
rect_center_y = image_size // 2

# 在图像上绘制矩形
image[rect_center_y - rect_height // 2: rect_center_y + rect_height // 2,
      rect_center_x - rect_width // 2: rect_center_x + rect_width // 2] = 1.0

# 计算图像的二维傅里叶变换
F = dft2D(image)

# 计算频谱的幅度（绝对值）
magnitude = np.abs(F)

# 不中心化的频谱图
F_unshifted = np.fft.ifftshift(F)

# 对幅度进行对数缩放以便于可视化
S_log = np.log(1 + np.abs(F_unshifted))

# 显示原始图像
plt.subplot(141)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# 显示傅里叶变换的幅度谱
plt.subplot(142)
plt.imshow(magnitude, cmap='gray')
plt.title('Spectrum Image (Unshifted, No Log)')
plt.axis('off')

# 显示未中心化的频谱图
plt.subplot(143)
plt.imshow(np.abs(F_unshifted), cmap='gray')
plt.title('Spectrum Image (Shifted, No Log)')
plt.axis('off')

# 显示对数化的频谱图
plt.subplot(144)
plt.imshow(S_log, cmap='gray')
plt.title('Spectrum Image (Log)')
plt.axis('off')

plt.tight_layout()
plt.show()
