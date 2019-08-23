#-*-coding:utf-8-*-
import os
import cv2
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt

PICTURE_PATH = './picture'
MASK_PATH = './mask'

class Entropy:
    hlist = [15, 25, 45, 55, 80, 108, 140, 165, 190, 220, 255, 275, 290, 316, 330, 345, 360]
    svlist = [int(255*0.15),int(255*0.4),int(255*0.75),255]
    binNumber = 16*(len(hlist)-1)+5*(len(svlist)-1)

    def quantilize(self, h, s, v):
        #hsv直方图量化
        # value : [21, 144, 23] h, s, v
        h = h * 2
        for i in range(len(self.hlist)):
            if h <= self.hlist[i]:
                h = i % (len(self.hlist)-1)
                break
        for i in range(len(self.svlist)):
            if s <= self.svlist[i]:
                s = i
                break
        for i in range(len(self.svlist)):
            if v <= self.svlist[i]:
                v = i
                break
        return 16 * h + 4 * s + v
    

    def colors(self, img, mask):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # 由于frompyfunc函数返回结果为对象，所以需要转换类型
        quantilize_ufunc = np.frompyfunc(self.quantilize, 3, 1)
        nhsv = quantilize_ufunc(hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]).astype(np.uint8)
        hist = cv2.calcHist([nhsv], [0], mask, [self.binNumber+1], [0,self.binNumber])
        hist = hist.reshape(1, hist.shape[0]).astype(np.int32).tolist()[0]
        return hist

    
    #计算图片的颜色熵
    def compute_entropy(self, hist):
        hist_sum = sum(hist)
        hist_rate = [unit/hist_sum for unit in hist]
        res = 0
        for value in hist_rate:
            if value:
                res -= value*np.log2(value)
        res /= np.log2(self.binNumber)
        return res


if __name__ == '__main__':
    picture_list = os.listdir(PICTURE_PATH)
    mask_list = os.listdir(MASK_PATH)
    record_file = open("entropy.txt","w")
    entropy = Entropy()
    for i,name in enumerate(picture_list):
        image_path = os.path.join(PICTURE_PATH,name)
        image = cv2.imread(image_path)
        mask_path = os.path.join(MASK_PATH,name)
        mask = cv2.imread(mask_path,0)
        mask_inv = cv2.bitwise_not(mask)  # 加入mask计算背景颜色熵值       
        if sum(sum(mask_inv)):
            hist = entropy.colors(image,mask_inv)
            color_entropy = entropy.compute_entropy(hist)
        else:
            color_entropy = 0
        print(name+' '+str(color_entropy),file=record_file)
        print(i)