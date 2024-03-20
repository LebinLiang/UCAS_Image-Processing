import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像并转换为灰度图像
image = cv2.imread('Fig01.tif')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 形态学操作：开运算
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
opened_image = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel_open)

# 计算背景图像
background_image = cv2.dilate(opened_image, kernel_open, iterations=1)

# 原图减去背景图
processed_image = cv2.absdiff(gray_image, background_image)

# 大津阈值法
_, processed_thresholded_image = cv2.threshold(processed_image, 0, 255,  cv2.THRESH_OTSU)

# 腐蚀操作
kernel_erode = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
processed_eroded_image = cv2.erode(processed_thresholded_image, kernel_erode, iterations=2)

# 轮廓查找
contours, _ = cv2.findContours(processed_eroded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 统计米粒个数和计算平均大小
rice_count = 0
total_area = 0

for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if 20 < area < 5000:  
        rice_count += 1
        total_area += area

print("米粒个数：", rice_count)

# 在图像上绘制轮廓和中心红点
for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if 20 < area < 5000:
        # 获取轮廓中心位置
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # 在原图上绘制轮廓
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

        # 在轮廓中心位置添加红点
        cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)

plt.figure(figsize=(16, 8))

# 显示原图
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

# 显示processed_image
plt.subplot(2, 2, 2)
plt.imshow(processed_image, cmap='gray')
plt.title('processed Image')

# 显示processed_eroded_image
plt.subplot(2, 2, 3)
plt.imshow(processed_eroded_image, cmap='gray')
plt.title('processed_eroded Image')

# 显示contour_image
plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title(f'Image with Contours (Rice Count: {rice_count})')

plt.show()