import cv2
import os
import shutil
import json
import numpy as np



class White:
    def white_process(self,img):
        imageHeight, imageWidth = img.shape[:2]
        # 图像预处理
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binImg = cv2.threshold(grayImg, 250, 255, cv2.THRESH_BINARY)
        # 寻找轮廓
        _, contours, hierarchy = cv2.findContours(binImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
                withrate = width/imageWidth
                lengthrate = height/imageHeight
                if reWidth <= reHeight:
                    angle = round(-angle,2)
                else:
                    angle = round(-angle+90,2)ss
                return withrate,lengthrate,angle
            else:
                return None,None,None
        else:
            return None,None,None




