import cv2
import os
import numpy as np
from color_metric import Metric
from color_entropy import Entropy

PICTURE_PATH = './picture/'
MASK_PATH = './mask/'
N = 32

def split_processing(image,mask,n=32):
    rows,cols,_ = image.shape
    subrows = rows//3
    subcols = cols//3
    metric_list = []
    entropy_list = []
    # merge_list = []
    for i in range(3):
        for j in range(3):
            subimage = image[i*subrows:(i+1)*subrows,j*subcols:(j+1)*subcols]
            submask = mask[i*subrows:(i+1)*subrows,j*subcols:(j+1)*subcols]
            if submask is not None and sum(sum(submask)) > 0:
                metric_processer = Metric(n)
                histList = metric_processer.compute_hist(subimage,submask)
                metric_list.append(metric_processer.compute_average(*histList))
                entropy_processer = Entropy()
                qulitify_hist = entropy_processer.colors(subimage,submask)
                entropy_list.append(entropy_processer.compute_entropy(qulitify_hist))
                # merge_list.append(Metric(Entropy.binNumber+1).compute_metric(qulitify_hist))
    if metric_list:
        avg_me = np.mean(metric_list)
    else:
        avg_me = 0
    if entropy_list:
        avg_en = np.mean(entropy_list)
    else:
        avg_en = 0
    return avg_me, avg_en


if __name__ == "__main__":
    imagePathList = os.listdir(PICTURE_PATH)
    for name in imagePathList:
        image = cv2.imread(PICTURE_PATH+name)
        mask = cv2.imread(MASK_PATH+name,0)
        metric = Metric(N)
        splitresult1,splitresult2 = split_processing(image,mask)
        print('metric:'+ name+ ' ' + 'split ' + str(splitresult1))
        print('entropy:'+ name+ ' ' + 'split ' + str(splitresult2))


        '''
        print('metric:'+ name+ ' ' + 'prime '+ str(primeresult[0]))
        print('metric:'+ name+ ' ' + 'split ' + str(splitresult1))
        
        entropy = Entropy()
        qualitify_hist = entropy.colors(image)
        prime_color_entropy = entropy.compute_entropy(qualitify_hist)
        print('entropy:'+ name+ ' ' + 'prime ' + str(prime_color_entropy))
        print('entropy:'+ name+ ' ' + 'split ' + str(splitresult2))

        
        prime_merge_metric = Metric(Entropy.binNumber+1).compute_metric(qualitify_hist)
        print('merge:'+ name+ ' ' + 'prime ' + str(prime_merge_metric))
        print('merge:'+ name+ ' ' + 'split ' + str(splitresult3))
        '''
        


    