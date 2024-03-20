import argparse
import os
from pathlib import Path
import numpy as np
import torch
from osgeo import gdal_array
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt

from skimage.metrics import structural_similarity as compare_ssim

from sewar import sam, rmse

from Config import ConfigForEvaluation, ConfigForEvaluationForSwin


def uiqi(im1, im2, block_size=64, return_map=False):
    if im1.shape[0] == 6:  # 调整成标准的[长，宽，通道]
        im1 = im1.transpose(1, 2, 0)
        im2 = im2.transpose(1, 2, 0)
    if len(im1.shape) == 3:
        return np.array(
            [uiqi(im1[:, :, i], im2[:, :, i], block_size, return_map=return_map) for i in range(im1.shape[2])])
    delta_x = np.std(im1, ddof=1)
    delta_y = np.std(im2, ddof=1)
    delta_xy = np.sum((im1 - np.mean(im1)) * (im2 - np.mean(im2))) / (im1.shape[0] * im1.shape[1] - 1)
    mu_x = np.mean(im1)
    mu_y = np.mean(im2)
    q1 = delta_xy / (delta_x * delta_y)
    q2 = 2 * mu_x * mu_y / (mu_x ** 2 + mu_y ** 2)
    q3 = 2 * delta_x * delta_y / (delta_x ** 2 + delta_y ** 2)
    q = q1 * q2 * q3
    return q


def calculate_ergas(real_images, predicted_images):
    """
    计算增强的灰度相似性指数（ERGAS）。

    参数:
    real_images -- 真实图像的列表，每个元素是一个通道。
    predicted_images -- 预测图像的列表，每个元素是一个通道。

    返回:
    ergas -- ERGAS指标的值。
    """
    ergas_sum = 0.0
    num_channels = len(real_images)
    # 遍历所有通道
    for real_img, pred_img in zip(real_images, predicted_images):
        # 计算RMSE
        channel_rmse = rmse(real_img, pred_img)

        # 计算图像平均亮度
        mean_brightness = np.mean(real_img)

        # 避免除以零
        mean_brightness_squared = max(mean_brightness ** 2, 1e-100)

        # 计算ERGAS值
        channel_ergas = (channel_rmse ** 2) / mean_brightness_squared

        # 累加ERGAS值
        ergas_sum += channel_ergas

    # 计算平均ERGAS值
    average_ergas = ergas_sum / num_channels

    # 缩放ERGAS值
    scaled_ergas = np.sqrt(average_ergas) * 6

    return scaled_ergas


def evaluate(y_true, y_pred, func):
    assert y_true.shape == y_pred.shape
    if y_true.ndim == 2:
        y_true = y_true[np.newaxis, :]
        y_pred = y_pred[np.newaxis, :]
    metrics = []
    for i in range(y_true.shape[0]):
        metrics.append(func(y_true[i], y_pred[i]))
    return metrics


def rmse_loss(y_true, y_pred):
    return evaluate(y_true, y_pred,
                    lambda x, y: sqrt(mean_squared_error(x.ravel(), y.ravel())))


def ssim(y_true, y_pred, data_range=1):
    return evaluate(y_true, y_pred,
                    lambda x, y: compare_ssim(x, y, data_range=data_range))


config = ConfigForEvaluationForSwin("LGC",
                                    save_dir_name="/home/zbl/datasets/STFusion/RunLog/FinePainterNet/LGC/2024-03-17/test/")


def getMean(data):
    return sum(data) / len(data)


def cc(real_image, predicted_image):
    """
    计算两个图像的相关系数。

    参数:
    real_image -- 真实图像，形状为 (channels, height, width)
    predicted_image -- 预测图像，形状应与 real_image 相同

    返回:
    cc_array -- 一个数组，包含每个通道的相关系数
    """
    # 确保输入图像的形状相同
    if real_image.shape != predicted_image.shape:
        raise ValueError("The shapes of real_image and predicted_image must be the same.")

    # 计算每个通道的相关系数
    cc_array = []
    for i in range(real_image.shape[0]):  # 遍历所有通道
        real_channel = real_image[i]
        pred_channel = predicted_image[i]

        # 计算均值
        mu_x = np.mean(real_channel)
        mu_y = np.mean(pred_channel)

        # 计算协方差和标准差
        cov_xy = np.sum((real_channel - mu_x) * (pred_channel - mu_y))
        var_x = np.sum(np.square(real_channel - mu_x))
        var_y = np.sum(np.square(pred_channel - mu_y))

        # 计算相关系数
        cc = cov_xy / (np.sqrt(var_x * var_y) + 1e-100)  # 添加一个小的常数以避免除以零

        cc_array.append(cc)

    return np.array(cc_array)


def trans_sam(real_image, predicted_image):
    return sam(real_image.transpose(1, 2, 0), predicted_image.transpose(1, 2, 0)) * 180 / np.pi


if __name__ == '__main__':

    # for idx, img in enumerate(config.predict_img_names):
    #     predict_dir = os.path.join(config.predict_dir, img)
    #     ground_truth_dir = os.path.join(config.ground_truth_dir, config.ref_img_names[idx])
    #     print(f"--------------[{idx + 1}/{len(config.predict_img_names)}]IMAGE_NAME:{img}----------------------------")
    #     ix = gdal_array.LoadFile(predict_dir).astype(np.int32)
    #     iy = gdal_array.LoadFile(ground_truth_dir).astype(np.int32)
    #     if config.choice == 'CIA':
    #         ix[iy == 0] = 0
    #     scale_factor = 0.0001
    #     xx = ix * scale_factor
    #     yy = iy * scale_factor
    #     print('RMSE', rmse_loss(yy, xx))
    #     print('SSIM', ssim(yy, xx))
    #     print('UIQI', uiqi(xx, yy))
    #     print('CC', cc(yy, xx))
    #     print('SAM', trans_sam(iy, ix))  # 在原论文中，只有sam是真实数据比的，其他指标都是放缩后再比的
    #     print('ERGAS', calculate_ergas(yy, xx))

    # predict_dir = "/home/zbl/codeLab/remotePython/RemoteSensingLab/fake.tif"
    # ground_truth_dir = "/home/zbl/datasets_paper/CIA-swinSTFM/val/2002_005_0105-2002_012_0112/20020112_TM.tif"
    predict_dir = "/home/zbl/codeLab/remotePython/RemoteSensingLab/fake.tif"
    ground_truth_dir = "/home/zbl/datasets_paper/LGC-swinSTFM/val/2004_331_1126-2004_347_1212/20041212_TM.tif"
    ix = gdal_array.LoadFile(predict_dir).astype(np.int32)
    iy = gdal_array.LoadFile(ground_truth_dir).astype(np.int32)
    # if config.choice == 'CIA':
    #     ix[iy == 0] = 0
    scale_factor = 0.0001
    xx = ix * scale_factor
    yy = iy * scale_factor
    print('RMSE', rmse_loss(yy, xx))
    print('SSIM', ssim(yy, xx))
    print('UIQI', uiqi(xx, yy))
    print('CC', cc(yy, xx))
    print('SAM', trans_sam(iy, ix))  # 在原论文中，只有sam是真实数据比的，其他指标都是放缩后再比的
    print('ERGAS', calculate_ergas(yy, xx))
