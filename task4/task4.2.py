import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像并转换为灰度图像
image = cv2.imread('Fig01.tif')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 获取图像尺寸
height, width = gray_image.shape

# 提高下三分之一部分的曝光
brightened_lower_third = gray_image.copy()
brightened_lower_third[2*height//3:height, :] = cv2.addWeighted(brightened_lower_third[2*height//3:height, :], 1.2, np.zeros_like(brightened_lower_third[2*height//3:height, :]), 0, 0)

# 对曝光后的图像进行平滑处理
blurred_image = cv2.GaussianBlur(brightened_lower_third, (5, 5), 0)

# 大津阈值法
_, otsu_thresholded_image = cv2.threshold(brightened_lower_third, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 形态学操作
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
morphology_image = cv2.morphologyEx(otsu_thresholded_image, cv2.MORPH_OPEN, kernel)

# 轮廓查找
contours, _ = cv2.findContours(morphology_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 统计米粒个数和计算平均大小
rice_count = 0
total_area = 0

for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if 50 < area < 5000:  # 根据实际情况调整面积阈值
        rice_count += 1
        total_area += area

print("米粒个数：", rice_count)

# 计算平均米粒大小
average_rice_size = total_area / rice_count
print("平均米粒大小：", average_rice_size)

# 在图像上绘制轮廓
for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if 50 < area < 5000:
        # 获取轮廓中心位置
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # 在contour_image上绘制轮廓
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)


# 显示灰度直方图和多个图像
plt.figure(figsize=(15, 8))

# 显示灰度直方图
plt.subplot(2, 2, 1)
plt.hist(brightened_lower_third.flatten(), 256, [0, 256], color='r')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('Equalized Grayscale Histogram')

# 显示contour_image
plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Contour Image')

# 显示otsu_thresholded_image
plt.subplot(2, 2, 3)
plt.imshow(otsu_thresholded_image, cmap='gray')
plt.title('Otsu Thresholded Image')

# 显示brightened_lower_third
plt.subplot(2, 2, 4)
plt.imshow(brightened_lower_third, cmap='gray')
plt.title('Brightened Image')

plt.show()
