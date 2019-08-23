import cv2
import numpy as np
import os
import random

IMAGES_PATH = './picture/'
IMAGE_SIZE = 300  # 每张小图片的大小
IMAGE_ROW = 1  # 图片间隔，也就是合并成一张图后，一共有几行
IMAGE_COLUMN = 2  # 图片间隔，也就是合并成一张图后，一共有几列
IMAGE_SAVE_PATH = './drawresult/'  # 图片转换后的地址
if not os.path.isdir(IMAGE_SAVE_PATH):
    os.mkdir(IMAGE_SAVE_PATH)

def drawline(img,color,thickness):
    startList = []
    endList = []
    row, column, channel = img.shape
    sub_column = column//IMAGE_COLUMN # 200
    sub_row = row//IMAGE_ROW # 400
    
    for i in range(1,IMAGE_ROW):
        startList.append((0, i*sub_row))
        endList.append((column, i*sub_row))
    for j in range(1,IMAGE_COLUMN):
        startList.append((j*sub_column, 0))
        endList.append((j*sub_column, row))
    lineType = 4
    for i in range(len(startList)):
        cv2.line(img, startList[i], endList[i], color, thickness, lineType)
    return img

if __name__ == "__main__":
    imagePathList = os.listdir(IMAGES_PATH)
    random_idx= random.sample(range(len(imagePathList)),50)
    image_names = [imagePathList[i] for i in random_idx]
    thickness_list = [8,10,12,14,16,18]
    for i, path in enumerate(image_names):
        if i <= len(image_names)//3:
            color = (0,0,0)
        else:
            color = (255,255,255)
        thickness = thickness_list[i%len(thickness_list)]
        pintu = cv2.imread(IMAGES_PATH+path)
        result = drawline(pintu,color,thickness)
        cv2.imwrite(IMAGE_SAVE_PATH+'pine-two2sss-'+path,result)
        print(i)

