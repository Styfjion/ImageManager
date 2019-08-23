import os
import cv2
import json
import numpy as np

PICTURE_PATH = './picture'
MASK_PATH = './mask'
N = 32

class Metric:
    def __init__(self,n):
        self.level = n
        self.sigma = ((1-1/n)**2 + ((1/n)**2)*(n-1))**0.5

    #计算HSV图像的量化直方图
    def compute_hist(self,image,mask):
        hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        histH = cv2.calcHist([hsv*2],[0],mask,[N],[0,361])
        histS = cv2.calcHist([hsv],[1],mask,[N],[0,256])
        histV = cv2.calcHist([hsv],[2],mask,[N],[0,256])
        return histH,histS,histV

    #计算单个通道的均匀度
    def compute_metric(self,hist):
        res = 0
        hist_sum  = sum(hist)
        for value in hist:
            res += (value/hist_sum - 1/self.level)**2
        metric = 1 - res**0.5/self.sigma
        return metric

    #计算平均均匀度
    def compute_average(self,histH,histS,histV):
        result = (self.compute_metric(histH)+self.compute_metric(histS)+self.compute_metric(histV))/3
        return result



if __name__ == "__main__":

    picture_list = os.listdir(PICTURE_PATH)
    mask_list = os.listdir(MASK_PATH)
    metric = Metric(N)
    record_file = open("result.txt","w")
    for name in picture_list:
        image_path = os.path.join(PICTURE_PATH,name)
        image = cv2.imread(image_path)
        mask_path = os.path.join(MASK_PATH,name)
        mask = cv2.imread(mask_path,0)
        mask_inv = cv2.bitwise_not(mask)  # 加入mask计算背景颜色分布均匀度 
        if sum(sum(mask_inv)):  # 无背景的图片记为0
            histList = metric.compute_hist(image,mask_inv)
            result = metric.compute_average(*histList)
        else:
            result = 0
        print(name+' '+str(result),file=record_file)
