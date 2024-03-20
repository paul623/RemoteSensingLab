# -*- coding: utf-8 -*-
import os
from pathlib import Path

import numpy as np
from osgeo import gdal
"""
CIA、LGC数据集转换工具
转成tif
"""

# 依据BIL存储规则，按照存储完一行的所有波段再存储下一行，进行提取并存入数组。
def read_as_bil(dataarr, bands, rows, col):
    imgarr = np.zeros((bands, rows, col))
    for r in range(rows):  # 取出一行的所有波段
        start = r * col * bands
        end = start + col * bands
        arrtem = dataarr[start:end]
        for b in range(bands):  # 取出每个波段
            start2 = b * col
            end2 = start2 + col
            imgarr[b, r, :] = arrtem[start2:end2]  # 存入数组对应位置
    return imgarr


# 依据BSQ存储规则，按照存储完单波段整幅图像后再存储下一波段的存储方法进行提取并存入数组。
def read_as_bsq(dataarr, bands, rows, col):
    imgarr = np.zeros((bands, rows, col))
    for b in range(bands):  # 取出每个波段
        start = b * rows * col
        end = start + rows * col
        arrtem = dataarr[start:end]
        for r in range(rows):  # 一维数组按行取出，存入对应三维数组。
            start2 = r * col
            end2 = start2 + col
            imgarr[b, r, :] = arrtem[start2:end2]
    return imgarr


# 依据BIP存储规则，按照一个像素所有波段进行存储完，再存储下一个像素所有波段的存储方法进行提取并存入数组。
def read_as_bip(dataarr, bands, rows, col):
    imgarr = np.zeros((bands, rows, col))
    for r in range(rows):  # 按行列遍历每个像元
        for c in range(col):
            if r == 0:
                pix = c
            else:
                pix = r * col + c
            start = pix * bands
            end = start + bands
            arrtem = dataarr[start:end]  # 从一维数组中取出每个像元的全波段元素（6个）
            for b in range(bands):
                imgarr[b, r, c] = arrtem[b]  # 赋值给对应数组
    return imgarr




bands, rows, col = 6, 2040, 1720

def readInt(path):
    f = open(path, 'rb')
    fint = np.fromfile(f, dtype=np.int16)
    return read_as_bsq(fint, bands, rows, col)

def readBil(path):
    f = open(path, 'rb')
    fint = np.fromfile(f, dtype=np.int16)
    return read_as_bil(fint, bands, rows, col)

def saveImg(path, imgarr):
    datatype = gdal.GDT_UInt16
    bands, high, width = imgarr.shape
    driver = gdal.GetDriverByName("GTiff")
    datas = driver.Create(path, col, rows, bands, datatype)
    for i in range(bands):
        datas.GetRasterBand(i + 1).WriteArray(imgarr[i])
    del datas


def read_data_dir(root):
    list_all = os.listdir(root)
    for folder in list_all:
        year = folder.split('_')[0]
        date = folder.split('_')[2]
        for filename in os.listdir(os.path.join(root, folder)):
            if filename.endswith('.int'):
                path = os.path.join(root, folder, filename)
                save_path = os.path.join(root, folder, filename.split('.')[0]+".tif")
                saveImg(save_path, readInt(path))
            elif filename.endswith('.bil'):
                path = os.path.join(root, folder, filename)
                save_path = os.path.join(root, folder, year+date+'_TM.tif')
                saveImg(save_path, readBil(path))
        print(filename+"  转换成功")

def delTrashFiles(root):
    list_all = os.listdir(root)
    for folder in list_all:
        for filename in os.listdir(os.path.join(root, folder)):
            if filename.endswith('xml') or filename.endswith('.int') or filename.endswith('.bil'):
                path = os.path.join(root, folder, filename)
                os.remove(path)
                print(f"path:{path}删除成功")

root_dir = r"D:\CodeLab\CIA0\CIA-v2\all"
if __name__ == "__main__":
    # read_data_dir(root_dir)
    delTrashFiles(root_dir)
