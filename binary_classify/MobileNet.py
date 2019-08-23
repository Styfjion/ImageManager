# -*- coding: utf-8 -*-
# set the matplotlib backend so figures can be saved in the background
# import the necessary packages
from keras.applications import *
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
from keras import  regularizers
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import os
import cv2
import random
import numpy as np

# initialize the number of epochs to train for, initial learning rate,
# and batch size
batch_size = 32
epochs = 2000
periods = 100
TRUE_PATH = './positive'
FALSE_PATH = './negetive'

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
    
    


def main():
    
    trueList = os.listdir(TRUE_PATH)
    falseList = os.listdir(FALSE_PATH)
    data_set = []
    for i in range(len(trueList)):
        data_set.append((loadImg(os.path.join(TRUE_PATH,trueList[i])),1))
    for i in range(len(falseList)):
        data_set.append((loadImg(os.path.join(FALSE_PATH,falseList[i])),0))
    np.random.seed(1024)
    random.shuffle(data_set)
    data_data = [unit[0] for unit in data_set]
    data_label = [unit[1] for unit in data_set]
    
    train_data = data_data[len(data_data)//4:]
    train_label = data_label[len(data_data)//4:]

    validate_data = data_data[:len(data_data)//4]
    validate_label = data_label[:len(data_data)//4]
    conv_base = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    # model.summary()
    model.layers[0].trainable = False
    
    model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                loss=losses.binary_crossentropy,
                metrics=[metrics.binary_accuracy])
    filepath="./record/weights-1-mobilenetv2-{epoch:05d}-{val_binary_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_binary_accuracy', verbose=1,period=periods)
    filepath2="./record/weights-1-mobilenetv2-best.hdf5"
    checkpoint2 = ModelCheckpoint(filepath2, monitor='val_binary_accuracy', verbose=1, save_best_only=True,mode='max')
    callbacks_list = [checkpoint,checkpoint2,TensorBoard(log_dir='./log1')]
    train_data = np.asarray(train_data)
    train_lable = np.asarray(train_label)
    validate_data = np.asarray(validate_data)
    validate_label = np.asarray(validate_label)
    model.fit(train_data,train_label,epochs=epochs,batch_size=batch_size,validation_data=(validate_data,validate_label),shuffle=True,callbacks=callbacks_list)
    model.save('result_resnet.h5')


#python train.py --dataset_train ../../traffic-sign/train --dataset_test ../../traffic-sign/test --model traffic_sign.model
if __name__=='__main__':
    main()