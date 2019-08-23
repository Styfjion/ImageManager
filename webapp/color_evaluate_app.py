from flask import Flask, jsonify, render_template, request
from color_entropy import *
from color_metric import *
from whitebar_detect import *
from clarity_assessment import *
from binary_pinjie import *
from split_pictrure_operation import *
from keras import models
import cv2

WEIGHT_FILE = './weights-1-vgg16-00100-0.97.hdf5'

app = Flask(__name__)

@app.route('/',methods = ['POST'])

def color_evaluate():
    if request.files.get('file1') and request.files.get('file2'):
        file1 = request.files.get('file1')
        file1.save('./picture/'+file1.filename)
        file2 = request.files.get('file2')
        file2.save('./picture/'+file2.filename)
        image = cv2.imread('./picture/'+file1.filename)

        #计算图像清晰度
        image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        clarity = energy(image_gray)

        #评估有无白边
        image_crop, row_top, raw_down, col_top, col_down = corp_margin(image)
        clone_rate = image_crop.shape[0]/image.shape[0]
        row_rate = image_crop.shape[1]/image.shape[1]
        if clone_rate > 0.9 and row_rate > 0.9:
            white = False
        else:
            white = True
        image = image_crop
    
        # 加入mask计算背景颜色分布均匀度 
        mask = cv2.imread('./picture/'+file2.filename,0)
        mask = mask[row_top:raw_down+1,col_top:col_down+1]
        mask_inv = cv2.bitwise_not(mask)  
        

        #计算背景颜色分布均匀度
        metric = Metric(32)
        if sum(sum(mask_inv)):
            histList = metric.compute_hist(image,mask_inv)
            metric_result = metric.compute_average(*histList)
        else:
            metric_result = 0
        
        
        #计算背景颜色熵
        entropy = Entropy()
        if sum(sum(mask_inv)):
            hist = entropy.colors(image,mask_inv)
            entropy_result = entropy.compute_entropy(hist)
        else:
            entropy_result = 0

        #判断是否为拼图
        prediction = load_model(WEIGHT_FILE,image)

        #qianjing分块颜色平均度和颜色熵
        avg_metric, avg_entropy = split_processing(image,mask)

        dict_trans = {True:'有',False:'无'}
        return "拼接置信度：{:.4f} 清晰度：{:.2f} 有无白边：{} 背景颜色分布均匀度：{:.4f} 背景颜色熵：{:.4f} 前景分块颜色分布均匀度：{:.4f} 前景分块颜色熵：{:.4f}".format(prediction[0][0], clarity, dict_trans[white], metric_result[0], entropy_result,avg_metric,avg_entropy)

    else:
        return "ERROR"


if __name__ == "__main__":
    app.run()
