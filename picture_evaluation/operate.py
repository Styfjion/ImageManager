from color_entropy import *
from color_metric import *
from clarity_assessment import *
from split_pictrure_operation import *
from whitebar_detect import *
import cv2
import json





def operation(image,mask,name):
    dict_result = {}
    dict_result['name'] = name
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
    mask = mask[row_top:raw_down+1,col_top:col_down+1]
    mask_inv = cv2.bitwise_not(mask)  
    

    #计算背景颜色分布均匀度
    metric = Metric(32)
    if sum(sum(mask_inv))>0:
        histList = metric.compute_hist(image,mask_inv)
        metric_result = metric.compute_average(*histList)
    else:
        metric_result = [0]
    
    
    #计算背景颜色熵
    entropy = Entropy()
    if sum(sum(mask_inv))>0:
        hist = entropy.colors(image,mask_inv)
        entropy_result = entropy.compute_entropy(hist)
    else:
        entropy_result = 0


    #前景分块颜色平均度和颜色熵
    avg_metric, avg_entropy = split_processing(image,mask)
    
    dict_result['white'] = str(white)
    dict_result['clearity'] = round(float(clarity),4)
    dict_result['metric_result'] = round(float(metric_result[0]),4)
    dict_result['entropy_result'] = round(float(entropy_result),4)
    dict_result['avg_metric'] = round(float(avg_metric),4)
    dict_result['avg_entropy'] = round(float(avg_entropy),4)

    return dict_result


if __name__ == "__main__":
    dict_list = []
    picture_list = os.listdir('./prime/')
    for i,name in enumerate(picture_list):
        image = cv2.imread('./prime/'+name)
        mask = cv2.imread('./mask/'+name,0)
        dict_result = operation(image,mask,name)
        dict_list.append(dict_result)
        print(i)
    with open('result.json','w') as f:
        json.dump(dict_list,f)

