# -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np

import cv2
import time
# 测试图片文件夹 
IMAGE_PATH = "picture"
# 成功检测的文件夹 
RESULT_PATH = "ptest/"
# 未能成功检测的文件夹
ERROR_PATH = "perror/"
# 居中度和倾斜度结果存放文件 
JSON_PATH = 'result.json'

if not os.path.isdir(RESULT_PATH):
    os.mkdir(RESULT_PATH)
if not os.path.isdir(ERROR_PATH):
    os.mkdir(ERROR_PATH)

# this makes our figures bigger
pylab.rcParams['figure.figsize'] = 20, 12

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

# 在config文件夹下选择对应主干网络的类型
#config_file = "configs/e2e_mask_rcnn_X_101_32x8d_FPN_1x.yaml"
config_file = "configs/e2e_ms_rcnn_R_101_FPN_1x.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
# 选择是否使用GPU  
cfg.merge_from_list(["MODEL.DEVICE", "cuda:0"])

#选择对应的权重
#cfg.MODEL.WEIGHT="models/e2e_mask_rcnn_X_101_32x8d_FPN_1x.pth"
cfg.MODEL.WEIGHT="models/model_0090000.pth"


coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.1
)


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


imagePathList = os.listdir(IMAGE_PATH)
imageList = []
jsonFile = open(JSON_PATH,'w')

for i,imagePath in enumerate(imagePathList):
    print("loading:{}".format(i))
    imagefilepath = os.path.join(IMAGE_PATH,imagePath)
    image = Image.open(imagefilepath).convert("RGB")
    image = np.array(image)[:, :, [2, 1, 0]]
    imageList.append(image)


# compute predictions
start = time.time()
predictions,errorList,successList = coco_demo.run_on_imageList(imageList,imagePathList,jsonFile)

for i,prediction in enumerate(predictions):
    print("writting:{}".format(i))
    targetPath = os.path.join(RESULT_PATH,successList[i])
    cv2.imwrite(targetPath, prediction)

for i,name in enumerate(errorList):
    print("writting:{}".format(i))
    errorImagePath = os.path.join(IMAGE_PATH,name)
    errorImage = cv2.imread(errorImagePath)
    targetPath = os.path.join(ERROR_PATH,name)
    cv2.imwrite(targetPath, errorImage)

end = time.time()
print("run time is {}s".format(end-start))
