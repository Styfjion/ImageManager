from keras import models
import os
import cv2
import numpy as np
import argparse

FILE_PATH = 'weights-1-vgg16-00100-0.97.hdf5'
PICTURE_PATH = './picture'


def loadImg(path):
    inImage = cv2.imread(path)
    info = np.iinfo(inImage.dtype)
    inImage = inImage.astype(np.float) / info.max
    iw = inImage.shape[1]
    ih = inImage.shape[0]
    if iw < ih:
        inImage = cv2.resize(inImage, (600, int(600 * ih/iw)))
    else:
        inImage = cv2.resize(inImage, (int(600 * iw / ih), 600))
    inImage = inImage[0:600, 0:600]
    inImage = 2 * inImage - 1
    shape = inImage.shape
    inImage = np.reshape(inImage,(shape[0],shape[1],3))
    return inImage

if __name__ == "__main__":
    model = models.load_model(FILE_PATH)
    '''
    picture = loadImg('test16.jpg')
    picture = np.reshape(picture,(1,picture.shape[0],picture.shape[1],picture.shape[2]))
    model_out = model.predict(picture)
    print(model_out)
    '''
    parser = argparse.ArgumentParser() 
    parser.add_argument('-i', type=str, help='input path of weight file')
    args = parser.parse_args()
    if args.i:    
        FILE_PATH = args.i
    imagePathList = os.listdir(PICTURE_PATH)
    imageList = []
    for path in imagePathList:
        imagePath = os.path.join(PICTURE_PATH,path)
        image = loadImg(imagePath)
        imageList.append(image)
    imageList = np.asarray(imageList)
    model_out = model.predict(imageList,batch_size=32,verbose=1)
    count = 0
    print(model_out)
    print(len(model_out))
    for unit in model_out:
        if unit[0] >= 0.5:
            count += 1
        #测试负样本时改为unit[0] < 0.5
    print(count/len(model_out))