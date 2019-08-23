# -*- coding:utf-
import threading, os, time
import tarfile

from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFile, ImageChops
import tempfile
from six.moves import urllib

import tensorflow as tf
import random
import cv2
import shutil

import random
import logging
import math

class DataAugmentation:
    """
    包含数据增强的6种方式
    """
 
    def __init__(self):
        pass
 
    @staticmethod
    def openImage(image):
        return Image.open(image, mode="r")
 
    @staticmethod
    def randomFlip(image, mode=Image.FLIP_LEFT_RIGHT):
        """
        对图像进行上下左右四个方面的随机翻转
        :param image: PIL的图像image
        :param model: 水平或者垂直方向的随机翻转模式,默认右向翻转
        :return: 翻转之后的图像
        """
        random_model = np.random.randint(0, 2)
        flip_model = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]
        return image.transpose(flip_model[random_model])
        # return image.transpose(mode)
 
    @staticmethod
    def randomShift(image):
    #def randomShift(image, xoffset, yoffset=None):
        """
        对图像进行平移操作
        :param image: PIL的图像image
        :param xoffset: x方向向右平移
        :param yoffset: y方向向下平移
        :return: 平移之后的图像
        """
        random_xoffset = np.random.randint(0, math.ceil(image.size[0]*0.2))
        random_yoffset = np.random.randint(0, math.ceil(image.size[1]*0.2))
        return ImageChops.offset(image, random_xoffset)
 
    @staticmethod
    def randomRotation(image, mode=Image.BICUBIC):
        """
         对图像进行随机任意角度(0~360度)旋转
        :param mode 邻近插值,双线性插值,双三次B样条插值(default)
        :param image PIL的图像image
        :return: 旋转转之后的图像
        """
        image2 = image.convert('RGBA')
        random_angle = np.random.randint(1, 360)
        rot = image2.rotate(random_angle, mode)
        white = Image.new('RGBA', rot.size, (255,255,255,255)) 
        out = Image.composite(rot, white, mask=rot)
        return out.convert(image.mode)
 
    @staticmethod
    def randomCrop(image):
        """
        对图像随意剪切,裁剪图像大小宽和高的2/3
        :param image: PIL的图像image
        :return: 剪切之后的图像
        """
        image_width = image.size[0]
        image_height = image.size[1]
        crop_image_width = math.ceil(image_width*2/3)
        crop_image_height = math.ceil(image_height*2/3)
        x = np.random.randint(0, image_width - crop_image_width)
        y = np.random.randint(0, image_height - crop_image_height) 
        random_region = (x, y, x + crop_image_width, y + crop_image_height)
        return image.crop(random_region)
 
    @staticmethod
    def randomColor(image):
        """
        对图像进行颜色抖动
        :param image: PIL的图像image
        :return: 有颜色色差的图像image
        """
        # random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        random_factor = np.random.randint(0, 15) / 10.
        color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
        #random_factor = np.random.randint(10, 21) / 10.  # 随机因子
        random_factor = np.random.randint(10, 12) / 10.  # 随机因子
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
        #random_factor = np.random.randint(10, 21) / 10.  # 随机因子
        random_factor = np.random.randint(10, 12) / 10.  # 随机因1子
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
        # random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        random_factor = np.random.randint(0, 15) / 10.
        return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度
 
    @staticmethod
    def randomGaussian(image, mean=0.2, sigma=0.3):
        """
         对图像进行高斯噪声处理
        :param image:
        :return:
        """
 
        def gaussianNoisy(im, mean=0.2, sigma=0.3):
            """
            对图像做高斯噪音处理
            :param im: 单通道图像
            :param mean: 偏移量
            :param sigma: 标准差
            :return:
            """
            for _i in range(len(im)):
                im[_i] += random.gauss(mean, sigma)
            return im
 
        # 将图像转化成数组
        img = np.asarray(image)
        img.flags.writeable = True  # 将数组改为读写模式
        width, height = img.shape[:2]
        try:
            img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
            img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
            img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
            img[:, :, 0] = img_r.reshape([width, height])
            img[:, :, 1] = img_g.reshape([width, height])
            img[:, :, 2] = img_b.reshape([width, height])
        except:
            img = img
        return Image.fromarray(np.uint8(img))
 
    @staticmethod
    def saveImage(image, path):
#         try:
        image.save(path)
#         except:
#             print('not save img: ', path)
#             pass
        
    
times = 5  #重复次数
imgs_dir = './sample/'
result_dir = './sample_au/'
imgs = os.listdir(imgs_dir)
funcMap = {"flip": DataAugmentation.randomFlip,
           "shift": DataAugmentation.randomShift,
            "rotation": DataAugmentation.randomRotation,
            "crop": DataAugmentation.randomCrop,
            "color": DataAugmentation.randomColor,
            "gaussian": DataAugmentation.randomGaussian
            }
# funcLists = {"flip", "shift", "rotation", "crop", "color", "gaussian"}
funcLists = {"flip", "shift","color", "gaussian"}
    
global _index
for i in range(len(imgs)):
    print('Processing {} image'.format(i))
    img = imgs[i]        
    img_name = img.split('.')[0]
    postfix = img.split('.')[1]   #后缀 
    if postfix.lower() in ['jpg', 'jpeg', 'png', 'bmp']:
        image = DataAugmentation.openImage(imgs_dir + img)
        _index = 1
        for i in range(times):
            for func in funcLists:
                new_image = funcMap[func](image)
                img_path = os.path.join(result_dir, img_name + '_new_' + str(_index) + '.' + postfix)
                DataAugmentation.saveImage(new_image, img_path)
                _index += 1     