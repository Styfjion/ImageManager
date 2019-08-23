from keras import models
import os
import shutil
import cv2
import numpy as np


def loadImg(inImage):
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

def load_model(model_path,image):
    model = models.load_model(model_path)
    picture = loadImg(image)
    picture = np.reshape(picture,(1,picture.shape[0],picture.shape[1],picture.shape[2]))
    model_out = model.predict(picture)
    return model_out
