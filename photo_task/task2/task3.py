import cv2
import numpy as np
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


# 读取.tif图像
image = cv2.imread('rose512.tif', cv2.IMREAD_GRAYSCALE)

# 将像素值范围归一化到[0, 1]
f = image.astype(float) / 255.0

# 使用dft2D将图像转换为频域图像
F = dft2D(f)

# 使用idft2D将频域图像转换回空域
g = idft2D(F)

# 转换后的图像可能包含复数部分，取实部
g = np.real(g)

# 计算误差图像
d = f - g

# 将还原的图像范围缩放到[0, 255]
g_scaled = (g * 255).astype(np.uint8)

# 创建一个画板并显示原始图像、还原图像和误差图像
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow((f * 255).astype(np.uint8), cmap='gray')
plt.title('Original Image (f)')
plt.axis('off')

plt.subplot(132)
plt.imshow(g_scaled, cmap='gray')
plt.title('Reconstructed Image (g)')
plt.axis('off')

plt.subplot(133)
plt.imshow((d * 255).astype(np.uint8), cmap='gray')
plt.title('Error Image (d)')
plt.axis('off')

plt.show()