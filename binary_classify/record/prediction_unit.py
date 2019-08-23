from keras import models
import os
import shutil
import cv2
import numpy as np

FILE_PATH = './record of vgg-2 with pooling/weights-1-vgg16-2-00300-0.88.hdf5'
POSITIVE_PATH = './ptest'
NEGETIVE_PATH = './ntest'
SUCCESS_PATH = './nsuccess_vgg_300'
ERROR_PATH = './nerroe_vgg_300'


def loadImg(path):
    inImage = cv2.imread(path)
    info = np.iinfo(inImage.dtype)
    inImage = inImage.astype(np.float) / info.max
    iw = inImage.shape[1]
    ih = inImage.shape[0]
    if iw < ih:
        inImage = cv2.resize(inImage, (224, int(224 * ih/iw)))
    else:
        inImage = cv2.resize(inImage, (int(224 * iw / ih), 224))
    inImage = inImage[0:224, 0:224]
    inImage = 2 * inImage - 1
    shape = inImage.shape
    inImage = np.reshape(inImage,(shape[0],shape[1],3))
    return inImage

if __name__ == "__main__":
    model = models.load_model(FILE_PATH)
    count = 0
    recordfile = open('result.txt','w')
    imagePathList = os.listdir(POSITIVE_PATH)
    if not os.path.isdir(SUCCESS_PATH):
        os.mkdir(SUCCESS_PATH)
    if not os.path.isdir(ERROR_PATH):
        os.mkdir(ERROR_PATH)
    for path in imagePathList:
        imagePath = POSITIVE_PATH + '/' + path
        picture = loadImg(imagePath)
        picture = np.reshape(picture,(1,picture.shape[0],picture.shape[1],picture.shape[2]))
        model_out = model.predict(picture)
        if FILE_PATH.find('merge') >= 0:
            model_out = model_out[2]
        print(path + ' ' + str(model_out))
        print(path + ' ' + str(model_out),file=recordfile)
        
        if model_out[0][0] < 0.5:
            newPath = SUCCESS_PATH + '/' + path
            shutil.copy(imagePath,newPath)
            count += 1
        else:
            newPath = ERROR_PATH + '/' + path
            shutil.copy(imagePath,newPath)

    imagePathList = os.listdir(NEGETIVE_PATH)
    for path in imagePathList:
        imagePath = NEGETIVE_PATH + '/' + path
        picture = loadImg(imagePath)
        picture = np.reshape(picture,(1,picture.shape[0],picture.shape[1],picture.shape[2]))
        model_out = model.predict(picture)
        if FILE_PATH.find('merge') >= 0:
            model_out = model_out[2]
        print(path + ' ' + str(model_out))
        print(path + ' ' + str(model_out),file=recordfile)
        
        if model_out[0][0] < 0.5:
            newPath = SUCCESS_PATH + '/' + path
            shutil.copy(imagePath,newPath)
            count += 1
        else:
            newPath = ERROR_PATH + '/' + path
            shutil.copy(imagePath,newPath)
    
    print(count/len(imagePathList),file=recordfile)
    recordfile.close()
