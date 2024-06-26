import os.path

import numpy as np
import torch
import torch.nn as nn
import time
# import skimage.measure as  sm
import skimage.metrics as sm
import cv2
from osgeo import gdal, gdalconst
import matplotlib.pyplot as plt
from tqdm import tqdm

import DataHelper


###img read tool###############################################################
def imgread(file, mode='gdal'):
    if mode == 'cv2':
        img = cv2.imread(file, -1) / 10000.  # /10000.
    if mode == 'gdal':
        img = gdal.Open(file).ReadAsArray() / 10000.  # /10000.
    return img


###weight caculate tools######################################################
def weight_caculate(data):
    return torch.log((abs(data) * 10000 + 1.00001))


def caculate_weight(l1m1, m1m2):
    # atmos difference
    wl1m1 = weight_caculate(l1m1)
    # time deference
    wm1m2 = weight_caculate(m1m2)
    return wl1m1 * wm1m2


###space distance caculate tool################################################
def indexdistance(window):
    # one window, one distance weight matrix
    [distx, disty] = np.meshgrid(np.arange(window[0]), np.arange(window[1]))
    centerlocx, centerlocy = (window[0] - 1) // 2, (window[1] - 1) // 2
    dist = 1 + (((distx - centerlocx) ** 2 + (disty - centerlocy) ** 2) ** 0.5) / ((window[0] - 1) // 2)
    return dist


###threshold select tool######################################################
def weight_bythreshold(weight, data, threshold):
    # make weight tensor
    weight[data <= threshold] = 1
    return weight


def weight_bythreshold_allbands(weight, l1m1, m1m2, thresholdmax):
    # make weight tensor
    weight[l1m1 <= thresholdmax[0]] = 1
    weight[m1m2 <= thresholdmax[1]] = 1
    allweight = (weight.sum(0).view(1, weight.shape[1], weight.shape[2])) / weight.shape[0]
    allweight[allweight != 1] = 0
    return allweight


###initial similar pixels tools################################################
def spectral_similar_threshold(clusters, NIR, red):
    thresholdNIR = NIR.std() * 2 / clusters
    thresholdred = red.std() * 2 / clusters
    return (thresholdNIR, thresholdred)


def caculate_similar(l1, threshold, window):
    # read l1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    l1 = nn.functional.unfold(l1, window)
    # caculate similar
    weight = torch.zeros(l1.shape, dtype=torch.float32).to(device)
    centerloc = (l1.size()[1] - 1) // 2
    weight = weight_bythreshold(weight, abs(l1 - l1[:, centerloc:centerloc + 1, :]), threshold)
    return weight


def classifier(l1):
    '''not used'''
    return


###similar pixels filter tools#################################################
def allband_arrayindex(arraylist, indexarray, rawindexshape):
    shape = arraylist[0].shape
    datalist = []
    for array in arraylist:
        newarray = torch.zeros(rawindexshape, dtype=torch.float32).cuda()
        for band in range(shape[1]):
            newarray[0, band] = array[0, band][indexarray]
        datalist.append(newarray)
    return datalist


def similar_filter(datalist, sital, sitam):
    [l1, m1, m2] = datalist
    l1m1 = abs(l1 - m1)
    m1m2 = abs(m2 - m1)
    #####
    l1m1 = nn.functional.unfold(l1m1, (1, 1)).max(1)[0] + (sital ** 2 + sitam ** 2) ** 0.5
    m1m2 = nn.functional.unfold(m1m2, (1, 1)).max(1)[0] + (sitam ** 2 + sitam ** 2) ** 0.5
    return (l1m1, m1m2)


###starfm for onepart##########################################################
def starfm_onepart(datalist, similar, thresholdmax, window, outshape, dist):
    #####param and data
    [l1, m1, m2] = datalist
    bandsize = l1.shape[1]
    outshape = outshape
    blocksize = outshape[0] * outshape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #####img to col
    l1 = nn.functional.unfold(l1, window)
    m1 = nn.functional.unfold(m1, window)
    m2 = nn.functional.unfold(m2, window)
    l1 = l1.view(bandsize, -1, blocksize)
    m1 = m1.view(bandsize, -1, blocksize)
    m2 = m2.view(bandsize, -1, blocksize)
    l1m1 = abs(l1 - m1)
    m1m2 = abs(m2 - m1)
    #####caculate weights
    # time and space weight
    w = caculate_weight(l1m1, m1m2)
    w = 1 / (w * dist)
    # similar pixels: 1:by threshold 2:by classifier
    wmask = torch.zeros(l1.shape, dtype=torch.float32).to(device)

    # filter similar pixels  for each band: (bandsize,windowsize,blocksize)
    # wmasknew=weight_bythreshold(wmask,l1m1,thresholdmax[0])
    # wmasknew=weight_bythreshold(wmasknew,m1m2,thresholdmax[1])

    # filter similar pixels for all bands: (1,windowsize,blocksize)
    wmasknew = weight_bythreshold_allbands(wmask, l1m1, m1m2, thresholdmax)
    # mask
    w = w * wmasknew * similar
    # normili
    w = w / (w.sum(1).view(w.shape[0], 1, w.shape[2]))
    #####predicte and trans
    # predicte l2
    l2 = (l1 + m2 - m1) * w
    l2 = l2.sum(1).reshape(1, bandsize, l2.shape[2])
    # col to img
    l2 = nn.functional.fold(l2.view(1, -1, blocksize), outshape, (1, 1))
    return l2


###starfm for allpart#########################################################
def starfm_main(l1r, m1r, m2r,
                param={'part_shape': (140, 140),
                       'window_size': (31, 31),
                       'clusters': 5,
                       'NIRindex': 3, 'redindex': 2,
                       'sital': 0.001, 'sitam': 0.001}):
    # get start time
    time_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # read parameters
    parts_shape = param['part_shape']
    window = param['window_size']
    clusters = param['clusters']
    NIRindex = param['NIRindex']
    redindex = param['redindex']
    sital = param['sital']
    sitam = param['sitam']
    # caculate initial similar pixels threshold
    threshold = spectral_similar_threshold(clusters, l1r[:, NIRindex:NIRindex + 1], l1r[:, redindex:redindex + 1])
    print('similar threshold (NIR,red)', threshold)
    ####shape
    imageshape = (l1r.shape[1], l1r.shape[2], l1r.shape[3])
    print('datashape:', imageshape)
    row = imageshape[1] // parts_shape[0] + 1
    col = imageshape[2] // parts_shape[1] + 1
    padrow = window[0] // 2
    padcol = window[1] // 2
    #####padding constant for conv;STARFM use Inverse distance weight(1/w),better to avoid 0 and NAN(1/0),or you can use another distance measure
    constant1 = 10
    constant2 = 20
    constant3 = 30
    l1 = torch.nn.functional.pad(l1r, (padrow, padcol, padrow, padcol), 'constant', constant1)
    m1 = torch.nn.functional.pad(m1r, (padrow, padcol, padrow, padcol), 'constant', constant2)
    m2 = torch.nn.functional.pad(m2r, (padrow, padcol, padrow, padcol), 'constant', constant3)
    # split parts , get index and  run for every part
    row_part = np.array_split(np.arange(imageshape[1]), row, axis=0)
    col_part = np.array_split(np.arange(imageshape[2]), col, axis=0)
    print('Split into {} parts,row number: {},col number: {}'.format(len(row_part) * len(row_part), len(row_part),
                                                                     len(row_part)))
    dist = nn.functional.unfold(
        torch.tensor(indexdistance(window), dtype=torch.float32).reshape(1, 1, window[0], window[1]), window).to(device)

    for rnumber, row_index in tqdm(enumerate(row_part), desc=f'生成图像中', total=len(row_part)):
        for cnumber, col_index in enumerate(col_part):
            ####run for part: (rnumber,cnumber)
            # print('now for part{}'.format((rnumber, cnumber)))
            ####output index
            rawindex = np.meshgrid(row_index, col_index)
            ####output shape
            rawindexshape = (col_index.shape[0], row_index.shape[0])
            ####the real parts_index ,for reading the padded data
            row_pad = np.arange(row_index[0], row_index[len(row_index) - 1] + window[0])
            col_pad = np.arange(col_index[0], col_index[len(col_index) - 1] + window[1])
            padindex = np.meshgrid(row_pad, col_pad)
            padindexshape = (col_pad.shape[0], row_pad.shape[0])
            ####caculate initial similar pixels
            NIR_similar = caculate_similar(l1[0, NIRindex][padindex].view(1, 1, padindexshape[0], padindexshape[1]),
                                           threshold[0], window)
            red_similar = caculate_similar(l1[0, redindex][padindex].view(1, 1, padindexshape[0], padindexshape[1]),
                                           threshold[1], window)
            similar = NIR_similar * red_similar
            ####caculate threshold used for similar_pixels_filter
            thresholdmax = similar_filter(
                allband_arrayindex([l1r, m1r, m2r], rawindex, (1, imageshape[0], rawindexshape[0], rawindexshape[1])),
                sital, sitam)
            ####Splicing each col at rnumber-th row
            if cnumber == 0:
                rowdata = starfm_onepart(
                    allband_arrayindex([l1, m1, m2], padindex, (1, imageshape[0], padindexshape[0], padindexshape[1])),
                    similar, thresholdmax, window, rawindexshape, dist
                )

            else:
                rowdata = torch.cat((rowdata,
                                     starfm_onepart(allband_arrayindex([l1, m1, m2], padindex, (
                                         1, imageshape[0], padindexshape[0], padindexshape[1])),
                                                    similar, thresholdmax, window, rawindexshape, dist)), 2)
                ####Splicing each row
        if rnumber == 0:
            l2_fake = rowdata
        else:
            l2_fake = torch.cat((l2_fake, rowdata), 3)

    l2_fake = l2_fake.transpose(3, 2)
    # time cost
    time_end = time.time()
    print('now over,use time {:.4f}'.format(time_end - time_start))
    return l2_fake


def trans(datafile):
    datashape = datafile.shape
    for index in tqdm(range(datashape[2]), desc="转换中"):
        for i in range(datashape[0]):
            for j in range(datashape[1]):
                datafile[i][j][index] *= 10000
    # datafile.transpose(1, 2, 0)
    return datafile


def starfm(paths, root, name):
    ##three band datas(sorry,just find them at home,i cant recognise the spectral response range of each band,'NIR' and 'red' are only examples)
    l1file = paths[1]
    l2file = paths[3]
    m1file = paths[0]
    m2file = paths[2]

    ##param
    param = {'part_shape': (75, 75),
             'window_size': (31, 31),
             'clusters': 5,
             'NIRindex': 1, 'redindex': 0,
             'sital': 0.001, 'sitam': 0.001}

    ##read images from files(numpy)
    l1 = imgread(l1file)
    m1 = imgread(m1file)
    m2 = imgread(m2file)
    l2_gt = imgread(l2file)

    ##numpy to tensor
    shape = l1.shape
    l1r = torch.tensor(l1.reshape(1, shape[0], shape[1], shape[2]), dtype=torch.float32)
    m1r = torch.tensor(m1.reshape(1, shape[0], shape[1], shape[2]), dtype=torch.float32)
    m2r = torch.tensor(m2.reshape(1, shape[0], shape[1], shape[2]), dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    l1r = l1r.to(device)
    m1r = m1r.to(device)
    m2r = m2r.to(device)

    ##predicte(tensor input —> tensor output)
    l2_fake = starfm_main(l1r, m1r, m2r, param)
    print(f'l2_fake.shape:{l2_fake.shape}')

    ##tensor to numpy
    if device.type == 'cuda':
        l2_fake = l2_fake[0].cpu().numpy()
    else:
        l2_fake = l2_fake[0].numpy()

        ##show results
    # transform:(chanel,H,W) to (H,W,chanel)
    l2_fake = l2_fake.transpose(1, 2, 0)
    l2_gt = l2_gt.transpose(1, 2, 0)
    l1 = l1.transpose(1, 2, 0)
    m1 = m1.transpose(1, 2, 0)
    m2 = m2.transpose(1, 2, 0)
    # plot
    # plt.figure('landsat:t1')
    # plt.imshow(l1)
    # plt.figure('landsat:t2_fake')
    # plt.imshow(l2_fake)
    # plt.figure('landsat:t2_groundtrue')
    # plt.imshow(l2_gt)

    ##evaluation
    # psnr = 10. * np.log10(1. / np.mean((l2_fake - l2_gt) ** 2))
    # ssim1 = sm.structural_similarity(l2_fake, l2_gt, data_range=1, multichannel=True)
    # ssim2 = sm.structural_similarity(l1, l2_gt, data_range=1, multichannel=True)
    # ssim3 = sm.structural_similarity(l1 + m2 - m1, l2_gt, data_range=1, multichannel=True)
    # print('psnr:{:.4f};with-similarpixels ssim: {:.4f};landsat_t1 ssim: {:.4f};non-similarpixels ssim: {:.4f}'.format(
    #     psnr, ssim1, ssim2, ssim3))

    trans(l2_fake)
    targetfile_name = f"PRED_{name}.tif"
    path = os.path.join(root, targetfile_name)
    writetif(l2_fake, path, l2file)

    return


def writetif(dataset, target_file, reference_file):
    reference = gdal.Open(reference_file, gdalconst.GA_ReadOnly)
    band_count = dataset.shape[2]  # 波段数
    print("波段数：", band_count)
    band1 = dataset[0]
    # data_type = band1.DataType
    target = gdal.GetDriverByName("GTiff").Create(target_file, xsize=dataset.shape[1],
                                                  ysize=dataset.shape[0],
                                                  bands=band_count,
                                                  eType=reference.GetRasterBand(1).DataType)
    geotrans = list(reference.GetGeoTransform())
    target.SetProjection(reference.GetProjection())  # 设置投影坐标
    target.SetGeoTransform(geotrans)  # 设置地理变换参数
    total = band_count + 1
    for index in tqdm(range(1, total), desc="写入中"):
        # data = dataset.GetRasterBand(index).ReadAsArray(buf_xsize=dataset.shape[0], buf_ysize=dataset.shape[1])
        out_band = target.GetRasterBand(index)
        # out_band.SetNoDataValue(dataset.GetRasterBand(index).GetNoDataValue())
        out_band.WriteArray(dataset[:, :, index - 1])  # 写入数据到新影像中
        out_band.FlushCache()
        out_band.ComputeBandStats(False)  # 计算统计信息
    print("写入完成")
    del dataset


save_path = r"/home/zbl/RunLog/STARFM/LGC/"
if __name__ == '__main__':
    list_dirs, names = DataHelper.getDataLoader(option="LGC")
    for i in range(len(list_dirs)):
        starfm(list_dirs[i], save_path, names[i])
