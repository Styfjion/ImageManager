# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------------------------
# Copyright (C), Data Department of Software Service Center, SiChuan Changhong Electronics Co.Ltd
# ------------------------------------------------------------------------------------------------
# @File          : demo.py
# @Time          : 6/13/19 2:37 PM
# @Author        : X.T.Xiao
# @Email         : xinting.xiao@changhong.com
# @PythonVersion :
# @Function      :
# ------------------------------------------------------------------------------------------------
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np

import cv2

IMAGE_PATH = "demo/item"
TARGET_PATH = "demo/item_result"
ERROR_PATH = "demo/item_error"
JSON_PATH = 'demo/threshold_result.json'

imagePathList = os.listdir(IMAGE_PATH)
errorPathList = set(os.listdir(ERROR_PATH))

# this makes our figures bigger
pylab.rcParams['figure.figsize'] = 20, 12

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

#config_file = "configs/e2e_mask_rcnn_X_101_32x8d_FPN_1x.yaml"
config_file = "configs/e2e_ms_rcnn_R_101_FPN_1x.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda:0"])
#cfg.MODEL.WEIGHT="models/e2e_mask_rcnn_X_101_32x8d_FPN_1x.pth"
cfg.MODEL.WEIGHT="models/model_0090000.pth"


def load(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    #response = requests.get(url)
    #pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    pil_image = Image.open(BytesIO(url)).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")

# from http://cocodataset.org/#explore?id=345434
# image = load("http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg")
# image = load("http://f.hiphotos.baidu.com/image/pic/item/fc1f4134970a304ed0b72229dbc8a786c8175c48.jpg")
#image = load("http://a.hiphotos.baidu.com/image/pic/item/f603918fa0ec08fa3139e00153ee3d6d55fbda5f.jpg")

def processing(threshold,jsonFile):

    coco_demo = COCODemo(
        cfg,
        min_image_size=800,
        confidence_threshold=threshold
    )

    imageList = []
    for i,imagePath in enumerate(imagePathList):
        print("loading:{}".format(i))
        imagefilepath = os.path.join(IMAGE_PATH,imagePath)
        image = Image.open(imagefilepath).convert("RGB")
        image = np.array(image)[:, :, [2, 1, 0]]
        imageList.append(image)

    # compute predictions
    predictions,errorList,successList = coco_demo.run_on_imageList(imageList,imagePathList,jsonFile)

    for i,prediction in enumerate(predictions):
        print("writting:{}".format(i))
        targetPath = os.path.join(TARGET_PATH,successList[i])
        cv2.imwrite(targetPath, prediction)
        imagePathList.remove(successList[i])
        if successList[i] in errorPathList:
            errorImagePath = os.path.join(ERROR_PATH,successList[i])
            errorPathList.remove(successList[i])
            os.remove(errorImagePath)

    for i,name in enumerate(errorList):
        print("writting:{}".format(i))
        errorImagePath = os.path.join(IMAGE_PATH,name)
        errorImage = cv2.imread(errorImagePath)
        targetPath = os.path.join(ERROR_PATH,name)
        cv2.imwrite(targetPath, errorImage)

if __name__ == '__main__':
    jsonFile = open(JSON_PATH,'w')
    processing(0.5,jsonFile)
    processing(0.3,jsonFile)
    processing(0.1,jsonFile)
