"""
Created on Sun Oct  8 15:09:42 2023
@author: Administrator
"""
import numpy as np


from osgeo import gdal
import matplotlib.pyplot as plt


# 以下为三种拉伸方式，如果不拉伸，图像太黑，拉伸完显示的图像更好看
def optimized_linear(arr):
    a, b = np.percentile(arr, (2.5, 99))
    c = a - 0.1 * (b - a)
    d = b + 0.5 * (b - a)
    arr = (arr - c) / (d - c) * 255
    arr = np.clip(arr, 0, 255)
    return np.uint8(arr)


def percent_linear(arr, percent=2):
    arr_min, arr_max = np.percentile(arr, (percent, 100 - percent))
    arr = (arr - arr_min) / (arr_max - arr_min) * 255
    arr = np.clip(arr, 0, 255)
    return np.uint8(arr)


def linear(arr):
    arr_min, arr_max = arr.min(), arr.max()
    arr = (arr - arr_min) / (arr_max - arr_min) * 255
    arr = np.clip(arr, 0, 255)
    return np.uint8(arr)


path = r"C:\Users\zhuba\Desktop\RunLog\GAN-STFM\CIA\PRED_2002_076_0317-2002_092_0402.tif"
data = gdal.Open(path)  # 读取tif文件
num_bands = data.RasterCount  # 获取波段数
print(num_bands)
tmp_img = data.ReadAsArray()  # 将数据转为数组
print(tmp_img.shape)
img_rgb = tmp_img.transpose(1, 2, 0)  # 由波段、行、列——>行、列、波段

img_rgb = np.array(img_rgb, dtype=np.uint16)  # 设置数据类型，np.unit8可修改
# img_rgb = np.array(img_rgb)
r = img_rgb[:, :, 3]
g = img_rgb[:, :, 2]
b = img_rgb[:, :, 1]

img_rgb = np.dstack((percent_linear(r), percent_linear(g), percent_linear(b)))  # 波段组合

# img_rgb = np.array(img_rgb, dtype=np.uint8)

# plt.imshow(img_rgb)
# plt.show()
# 通过调用plt.axis（“ off”），可以删除编号的轴
plt.figure(dpi=800)
plt.axis("off")
plt.imshow(img_rgb)
plt.show()