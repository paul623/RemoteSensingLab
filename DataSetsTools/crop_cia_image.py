'''
https://zhuanlan.zhihu.com/p/407049191
参照论文
Remote Sensing Image Spatiotemporal Fusion Using a Generative Adversarial Network，确定裁剪大小（行1792，列1280，波段6
'''
import os

# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
"""
针对CIA的异常数据进行裁剪
没什么用~
"""

def crop2tif(path, save_path, row, col):
    top = 20
    left = 100
    img = gdal.Open(path)
    bands = img.RasterCount
    scol = img.RasterXSize
    srow = img.RasterYSize
    image_geotrans = img.GetGeoTransform()  # 获取仿射矩阵信息
    image_projetion = img.GetProjection()  # 获取投影信息
    img_data = img.ReadAsArray()
    imgarr = img_data[:, top:row + top, left:col + left]
    bands, r, c = imgarr.shape
    datatype = gdal.GDT_UInt16
    driver = gdal.GetDriverByName("GTiff")
    datas = driver.Create(save_path, c, r, bands, datatype)

    # 设置地理坐标和仿射变换信息,注意这里源图像没有添加坐标和仿射变换信息，所以继承了源图像，存储后的图像不能使用ENVI打开
    datas.SetGeoTransform(image_geotrans)
    datas.SetProjection(image_projetion)

    for i in range(bands):
        datas.GetRasterBand(i + 1).WriteArray(imgarr[i])
    del datas


rows = 1792
cols = 1280

root_dir = r'D:\CodeLab\CIA0\CIA-v2\all'
cropped_dir = r'D:\CodeLab\CIA0\CIA-v2\cropped'
if __name__ == "__main__":
    if not os.path.exists(cropped_dir):
        os.makedirs(cropped_dir)
    list_all = os.listdir(root_dir)
    for folder in list_all:
        if not os.path.exists(os.path.join(cropped_dir, folder)):
            os.makedirs(os.path.join(cropped_dir, folder))
        for filename in os.listdir(os.path.join(root_dir, folder)):
            if filename.endswith('tif'):
                path = os.path.join(root_dir, folder, filename)
                save_path = os.path.join(cropped_dir, folder, filename)
                crop2tif(path, save_path, rows, cols)
                print(f"path:{path}转换成功")
