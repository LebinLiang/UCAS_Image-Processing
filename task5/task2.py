
import numpy as np
from PIL import Image 
import matplotlib.pyplot as plt

"""
    提取图像指定行或列的像素值，并返回其均值。

    参数：
    - f: 输入的图像数组
    - I: 行或列的索引，可以是整数或包含两个整数的列表
    - loc: 指定是提取行像素还是列像素，可选值为 "row" 或 "column"

    返回值：
    - sOut: 提取行或列像素的均值，当输入为奇数行/列时返回整数，偶数行/列时返回浮点数

"""

def scanLine4e(f,I,loc):
    try:
        if type(I) is int:          # 判断I的类型，当行/列是奇数时，I为INT类型，返回该行/列像素的均值
            if loc == "row":
                sOut = f[I,:]
            elif loc == "column":
                sOut = f[:,I]

        else:                     # 判断I的类型，当行/列是偶数时，I为list类型，返回中间两行像素的均值
            if loc == "row":
                sOut = np.sum(f[I[0]:I[1]+1,:],axis=0) / 2
                sOut = np.int16(sOut)
            elif loc == "column":
                sOut = np.sum(f[:,I[0]:I[1]+1],axis=1) / 2
                sOut = np.int16(sOut)

        return sOut 

    except:
        print("ERROR01:参数值错误")


"""
    获取图像的中心行或中心列的索引。

    参数：
    - length: 图像的宽度或长度

    返回值：
    - index: 中心行或中心列的索引，如果图像宽度或长度为偶数，则返回包含两个整数的列表，否则返回一个整数
"""
def getCenterIndex(length):
    if length % 2 == 0:
        index = [int(length / 2) - 1, int(length / 2)]  #偶数长度取中间两行/列的索引
    else:
        index = int(length / 2)  #奇数长度取中间行/列的索引
    
    return index


"""
    提取图像中心行或中心列的像素值。

    参数：
    - image: 输入的图像数组
    - axis: 指定提取中心行或中心列的像素值，可选值为 "row" 或 "column"

    返回值：
    - index_out: 中心行或中心列的像素值，如果图像宽度或长度为偶数，则返回浮点数数组，否则返回整数数组
"""
def extractCenterLinePixels(image, axis):
    try:
        length = image.shape[0] if axis == "row" else image.shape[1]  #获取图像的长度或宽度
        index = getCenterIndex(length)  #获取中心行或中心列的索
        index_out = scanLine4e(image,index,axis)
        return index_out
    except:
        print("ERROR02:参数值错误")


cameraman = np.array(Image.open('./cameraman.tif'))# 读入图片cameraman
einstein = np.array(Image.open('./einstein.tif'))# 读入 图片einstein
H1,W1 = cameraman.shape # 获取图片长宽
H2,W2 = einstein.shape
print("cameraman.tif的尺寸为({},{})".format(H1,W1)) # 打印图片长宽
print("einstein.tif的尺寸为({},{})".format(H2,W2))


cameraman_row = extractCenterLinePixels(cameraman,"row")#列提取
cameraman_row = cameraman_row.reshape(1,W1) # 一维转二维 方便显示
cameraman_column = extractCenterLinePixels(cameraman,"column") #列提取
cameraman_column = cameraman_column.reshape(1,H1)  # 一维转二维

einstein_row = extractCenterLinePixels(einstein,"row")#列提取
einstein_row = einstein_row.reshape(1,W2)  # 一维转二维
einstein_column = extractCenterLinePixels(einstein,"column")#列提取
einstein_column = einstein_column.reshape(1,H2)  # 一维转二维


plt.figure(figsize=(30,10))  # 创建显示窗口尺寸

plt.subplot(2,1,1)
plt.title("Gray sequence of cameraman.tif")
plt.plot(cameraman_row[0,:],label = 'central row')
plt.plot(cameraman_column[0,:],label = 'central column')
plt.legend()

plt.subplot(2,1,2)
plt.title("Gray sequence of einstein.tif")
plt.plot(einstein_row[0,:],label = 'central row')
plt.plot(einstein_column[0,:],label = 'central column')
plt.legend()
plt.show()

