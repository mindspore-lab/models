import imgaug as ia
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import torch
from torchvision import transforms, datasets
import os
import numpy as np
import cv2
import imgaug.augmenters as iaa
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
class CustomTransform():

    def __init__(self, prob=0.5):
        #super(CustomTransform, self).__init__(keys)
        self.prob = prob
        self.aug=iaa.Sequential([
            iaa.Fliplr(0.5),
            # 	水平镜面翻转
            iaa.Flipud(0.3), # 垂直翻转
            #   随机裁剪图片边长比例的0~0.1
            iaa.Crop(percent=(0,0.2)),

            #Sometimes是指指针对50%的图片做处理
            iaa.Sometimes(
                0.5,
                #高斯模糊
                iaa.GaussianBlur(sigma=(0,0.5))
            ),

            #增强或减弱图片的对比度
            iaa.LinearContrast((0.75,1.5)),

            #添加高斯噪声
            #对于50%的图片,这个噪采样对于每个像素点指整张图片采用同一个值
            #剩下的50%的图片，对于通道进行采样(一张图片会有多个值)
            #改变像素点的颜色(不仅仅是亮度)
            iaa.AdditiveGaussianNoise(loc=0,scale=(0.0,0.05*255),per_channel=0.5),

            #让一些图片变的更亮,一些图片变得更暗
            #对20%的图片,针对通道进行处理
            #剩下的图片,针对图片进行处理
            iaa.Multiply((0.8,1.2),per_channel=0.2),

            #仿射变换
            iaa.Affine(
                #缩放变换
                scale={"x":(0.7,1.3),"y":(0.7,1.3)},
                #平移变换
                translate_percent={"x":(-0.3,0.3),"y":(-0.3,0.3)},
                #旋转
                rotate=(-25,25),
                #剪切
                shear=(-8,8),
                # 以下是指定这些新的像素点的生成方法,这种指定通过设置cval和mode两个参数来实现。参数order用来设置插值方法。
                order=[0, 1],
                cval=(0, 255),
                mode=ia.ALL
            ),

            # iaa.Sometimes(
            #     0.5,
            #     # 浮雕效果
            #     iaa.Emboss(alpha=(0, 0.3), strength=(0, 2.0)),
            # ),

            iaa.Sometimes(
                0.5,
                # 锐化
                iaa.Sharpen(alpha=(0, 0.3), lightness=(0.75, 1.5)),
            ),

            iaa.Add((-10, 10), per_channel=0.5),

            #使用随机组合上面的数据增强来处理图片
            ],random_order=True
        )

    def __call__(self, img):
        img = np.array(img)
        img=self.aug.augment_image(img)
        return img.copy()

