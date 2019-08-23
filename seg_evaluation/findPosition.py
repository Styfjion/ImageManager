import cv2
import os
import shutil
import json
import numpy as np

source_path = './picture/'
target_path = './wihte_result_200/'
error_path = './white_error_200/'
if not os.path.isdir(target_path):
    os.mkdir(target_path)
if not os.path.isdir(error_path):
    os.mkdir(error_path)
img_list = os.listdir(source_path)
dictionaryList = []
for i, name in enumerate(img_list):
    dictPosition = {}
    img = cv2.imread(source_path+name)
    imageHeight, imageWidth = img.shape[:2]
    # 图像预处理
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binImg = cv2.threshold(grayImg, 200, 255, cv2.THRESH_BINARY)
    # 寻找轮廓
    contours, hierarchy = cv2.findContours(binImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 画出轮廓
    if contours:
        point_array = None
        contour_area = [cv2.contourArea(contour) for contour in contours]
        del contours[contour_area.index(max(contour_area))]
        if contours:
            cv2.drawContours(img,contours,-1,(0,200,0),2)
            for contour in contours:
                pointset = np.reshape(contour,(contour.shape[0],-1))
                if point_array is None:
                    point_array = pointset
                else:
                    point_array = np.concatenate((point_array,pointset), axis = 0)
            rect = cv2.minAreaRect(point_array)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img,[box],0,(0,0,200),3)
            width,height = rect[0]
            reWidth,reHeight = rect[1]
            angle = rect[2]
            angle = rect[2]
            dictPosition['name'] = name
            dictPosition['width_position'] = width/imageWidth
            dictPosition['height_position'] = height/imageHeight
            if reWidth <= reHeight:
                dictPosition['angle'] = round(-angle,2)
            else:
                dictPosition['angle'] = round(-angle+90,2)
            cv2.imwrite(target_path+name,img)
        else:
            shutil.copy(source_path+name, error_path+name)
    else:
        shutil.copy(source_path+name, error_path+name)
    dictionaryList.append(dictPosition)
    print(i)
with open("white_result.json","w") as f:
    json.dump(dictionaryList,f)


