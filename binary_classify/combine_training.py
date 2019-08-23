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
import keras.backend as K
import os
import cv2
import random
import numpy as np
batch_size = 64
epochs = 1000
periods = 50
TRUE_PATH_1 = './1'
TRUE_PATH_2 = './2'
TRUE_PATH_3 = './3'
FALSE_PATH = './0'


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

def model_generate():
    #conv_base = MobileNetV2(weights='imagenet',include_top=False,pooling='avg')
    conv_base = VGG16(weights='imagenet',include_top=False,pooling='avg')
    x = conv_base.output
    x = layers.Dense(256,activation = 'relu', name='fn_1')(x)
    class1 = layers.Dense(1,activation = 'sigmoid',name = 'class1')(x)
    class2 = layers.Dense(1,activation = 'sigmoid',name = 'class2')(x)
    class3 = layers.Dense(1,activation = 'sigmoid',name = 'class3')(x)
    model = models.Model(inputs = conv_base.input, outputs = [class1,class2,class3])
    for layer in conv_base.layers:
        layer.trainable = False
    return model

def main():
    trueList1 = os.listdir(TRUE_PATH_1)
    trueList2 = os.listdir(TRUE_PATH_2)
    trueList3 = os.listdir(TRUE_PATH_3)
    falseList = os.listdir(FALSE_PATH)
    data_set = []
    for i in range(len(trueList1)):
        data_set.append((loadImg(os.path.join(TRUE_PATH_1,trueList1[i])),1,0,0))
    for i in range(len(trueList2)):
        data_set.append((loadImg(os.path.join(TRUE_PATH_2,trueList2[i])),0,1,0))
    for i in range(len(trueList3)):
        data_set.append((loadImg(os.path.join(TRUE_PATH_3,trueList3[i])),0,0,1))
    for i in range(len(falseList)):
        data_set.append((loadImg(os.path.join(FALSE_PATH,falseList[i])),0,0,0))
    np.random.seed(1024)
    random.shuffle(data_set)
    
    data_data = [unit[0] for unit in data_set]
    data_label_1 = [unit[1] for unit in data_set]
    data_label_2 = [unit[2] for unit in data_set]
    data_label_3 = [unit[3] for unit in data_set]
    
    data_data = np.asarray(data_data)
    data_label_1 = np.asarray(data_label_1)
    data_label_2 = np.asarray(data_label_2)
    data_label_3 = np.asarray(data_label_3)
    
    train_data = data_data[len(data_data)//4:]
    train_label_1 = data_label_1[len(data_data)//4:]
    train_label_2 = data_label_2[len(data_data)//4:]
    train_label_3 = data_label_3[len(data_data)//4:]

    validate_data = data_data[:len(data_data)//4]
    validate_label_1 = data_label_1[:len(data_data)//4]
    validate_label_2 = data_label_2[:len(data_data)//4]
    validate_label_3 = data_label_3[:len(data_data)//4]
    model = model_generate()
    model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                  loss=[losses.binary_crossentropy,losses.binary_crossentropy,losses.binary_crossentropy],
                  loss_weights = [1,1,1],
                  metrics=['accuracy'])
    filepath="./record/weights-1-merge-vgg-{epoch:05d}-{val_class3_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_class3_acc', verbose=1,period=periods)
    filepath2="./record/weights-1-merge-vgg-best.hdf5"
    checkpoint2 = ModelCheckpoint(filepath2, monitor='val_loss', verbose=1, save_best_only=True,mode='min')
    callbacks_list = [checkpoint,checkpoint2,TensorBoard(log_dir='./log1')]
    
    model.fit(train_data,[train_label_1,train_label_2,train_label_3], epochs=epochs,batch_size=batch_size,validation_data=(validate_data,[validate_label_1,validate_label_2,validate_label_3]),shuffle=True,callbacks=callbacks_list)
    model.save('result-merge1.h5')

if __name__ == "__main__":
    '''
    model = model_generate()
    model.summary()
    '''
    main()