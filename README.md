# 遥感图像时空融合方法汇总

本项目已经包含以传统方法以及验证指标：RMSE，SSIM，UIQI，CC，SAM，ERGAS

DataSetsTools放了一些常用的CIA和LGC数据集的处理代码

自己整理的一些关于遥感图像时空融合的相关知识：[遥感图像时空融合看板](https://www.yuque.com/basailuonadeyuhui/lczi48/ur61mu8fgbmum727?singleDoc# 《遥感图像 时空融合知识库 看板》)

全部传统方法请参考这个仓库：[Free-shared-Spatiotemporal-method-of-remote-sensing](https://github.com/max19951001/Free-shared-Spatiotemporal-method-of-remote-sensing-)

## 传统方法

### STARFM

[endu111/remote-sensing-images-fusion: remote sensing images fusion,a topic similar to super resolution (github.com)](https://github.com/endu111/remote-sensing-images-fusion/tree/master)

[文献阅读：STARFM - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/412081033)

[STARFM（Python版）_starfm库-CSDN博客](https://blog.csdn.net/qq_43873392/article/details/127990068)

### ESTARFM/FSDAF

[遥感 如何获取时空融合-starfm\estarfm\fsdaf 算法的python代码(自带测试数据)_estarfm融合-CSDN博客](https://blog.csdn.net/Nieqqwe/article/details/124341403)

[FSDAF效果始终不如STARFM的原因和解决办法（在LGC和CIA数据集上） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/436387889)

### Fit-FC

[HoucaiGuo/Fit-FC-Python: Python implementation of the Fit-FC algorithm for spatiotemporal fusion of remote sensing images. (github.com)](https://github.com/HoucaiGuo/Fit-FC-Python)

## 深度学习的方法

### STFCNN

论文：[Spatiotemporal Satellite Image Fusion Using Deep Convolutional Neural Networks](https://ieeexplore.ieee.org/abstract/document/8291042)

代码：[setneicum/stfcnn](https://github.com/setneicum/stfcnn)

### GANSTFM

论文：[A Flexible Reference-Insensitive Spatiotemporal Fusion Model for Remote Sensing Images Using Conditional Generative Adversarial Network | IEEE Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/9336033)

代码：[theonegis/ganstfm: A Flexible Spatiotemporal Fusion Model for Remote Sensing Images With Conditional Generative Adversarial Network (github.com)](https://github.com/theonegis/ganstfm)

### MSNet【未找到】

论文：[Remote Sensing | Free Full-Text | MSNet: A Multi-Stream Fusion Network for Remote Sensing Spatiotemporal Fusion Based on Transformer and Convolution (mdpi.com)](https://www.mdpi.com/2072-4292/13/18/3724)

### SwinSTFM

论文：[SwinSTFM: Remote Sensing Spatiotemporal Fusion Using Swin Transformer | IEEE Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/9795183)

代码：[LouisChen0104/swinstfm: Code of SwinSTFM: Remote Sensing Spatiotemporal Fusion using Swin Transformer (github.com)](https://github.com/LouisChen0104/swinstfm)

### CycleGAN-STF 【未找到】

论文：[CycleGAN-STF: Spatiotemporal Fusion via CycleGAN-Based Image Generation | IEEE Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/9206067)

### StfNet【未找到】

论文：[StfNet: A Two-Stream Convolutional Neural Network for Spatiotemporal Image Fusion | IEEE Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/8693668)

### EDCSTFN

论文：[Remote Sensing | Free Full-Text | An Enhanced Deep Convolutional Model for Spatiotemporal Image Fusion (mdpi.com)](https://www.mdpi.com/2072-4292/11/24/2898?ref=https://githubhelp.com)

代码：[theonegis/edcstfn: An Enhanced Deep Convolutional Model for Spatiotemporal Image Fusion (github.com)](https://github.com/theonegis/edcstfn)

### MLFF-GAN

论文：[MLFF-GAN: A Multilevel Feature Fusion With GAN for Spatiotemporal Remote Sensing Images | IEEE Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/9781347/)

代码：[songbingze/MLFF-GAN (github.com)](https://github.com/songbingze/MLFF-GAN)

### ECPW-STFN

论文：[Enhanced wavelet based spatiotemporal fusion networks using cross-paired remote sensing images - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S092427162400176X)

代码：[lixinghua5540/ECPW-STFN: Enhanced wavelet based spatiotemporal fusion networks using cross-paired remote sensing images, 2024](https://github.com/lixinghua5540/ECPW-STFN)

### STFDiff

论文：[STFDiff: Remote sensing image spatiotemporal fusion with diffusion models - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1566253524002835)

代码：[prowDIY/STF](https://github.com/prowDIY/STF)

注：模型路径在<u>src.model.stfdiff.model6_GN_SiLU</u>

### STM-STFNet

论文：[[A Dual-Perspective Spatiotemporal Fusion Model for Remote Sensing Images by Discriminative Learning of the Spatial and Temporal Mapping](https://ieeexplore.ieee.org/abstract/document/10595407)]

代码：[zhonhua/STM-STFNet](https://github.com/zhonhua/STM-STFNet)

## 联系我

如果有代码贡献欢迎联系我。

## 声明

本仓库仅供学习交流使用，请勿用于非法用途，如有侵权请联系我。
