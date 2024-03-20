import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像并转换为灰度图像
image = cv2.imread('Fig01.tif')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 大津阈值法
otsu_thresholded_image = cv2.adaptiveThreshold(gray_image,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,103, 1)

# 形态学操作：开运算
kernel_open = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))
opened_image = cv2.morphologyEx(otsu_thresholded_image, cv2.MORPH_OPEN, kernel_open)

# 轮廓查找
contours, _ = cv2.findContours(opened_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

# 在图像上绘制轮廓和中心标号数字
for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if 50 < area < 5000:
        # 获取轮廓中心位置
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # 在原图上绘制轮廓
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

        # 在轮廓中心位置添加红点
        cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)

# 显示灰度直方图和多个图像
plt.figure(figsize=(15, 8))

# 显示灰度直方图
plt.subplot(2, 2, 1)
plt.hist(gray_image.flatten(), 256, [0, 256], color='r')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('Equalized Grayscale Histogram')

# 显示contour_image
plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Contour Image')

# 显示otsu_thresholded_image
plt.subplot(2, 2, 3)
plt.imshow(opened_image, cmap='gray')
plt.title('Opened Image (After Otsu Thresholding)')

# 显示brightened_lower_third
plt.subplot(2, 2, 4)
plt.imshow(otsu_thresholded_image, cmap='gray')
plt.title('Brightened Image')

plt.show()
