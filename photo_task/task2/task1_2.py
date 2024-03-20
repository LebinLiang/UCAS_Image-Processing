import cv2
import numpy as np

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
normalized_image = image.astype(float) / 255.0

# 使用dft2D将图像转换为频域图像
F = dft2D(normalized_image)

# 使用idft2D将频域图像转换回空域
reconstructed_image = idft2D(F)

# 转换后的图像可能包含复数部分，取实部
reconstructed_image = np.real(reconstructed_image)

# 显示原始图像和转换后的图像
cv2.imshow('Original Image', image)
cv2.imshow('Reconstructed Image', (reconstructed_image * 255).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()