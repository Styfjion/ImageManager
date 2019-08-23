from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
from keras import  regularizers
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import numpy as np
import os
import cv2

batch_size = 32
epochs = 1000
periods = 50
TRUE_PATH = './positive'
FALSE_PATH = './negetive'

# 载入训练好的全连接层和分类器
weight_path = 'record/weights-1-vgg16-2-00300-0.88.hdf5'

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
    np.random.shuffle(data_set)
    data_data = [unit[0] for unit in data_set]
    data_label = [unit[1] for unit in data_set]

    train_data = data_data[len(data_data)//4:]
    train_label = data_label[len(data_data)//4:]

    validate_data = data_data[:len(data_data)//4]
    validate_label = data_label[:len(data_data)//4]
    
    model = models.load_model(weight_path)
    set_trainable = False
    for layer in model.layers[0].layers:
        if layer.name == 'block5_conv1':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
    model.summary()
    model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
    loss=losses.binary_crossentropy,
    metrics=[metrics.binary_accuracy])
    filepath="./record/weights-2-vgg16-2-{epoch:05d}-{val_binary_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_binary_accuracy', verbose=1,period=periods)
    filepath2="./record/weights-2-vgg16-2-best.hdf5"
    filepath3="./record/weights-2-vgg16-2-loss-best.hdf5"
    checkpoint2 = ModelCheckpoint(filepath2, monitor='val_binary_accuracy', verbose=1, save_best_only=True,mode='max')
    checkpoint3 = ModelCheckpoint(filepath3, monitor='val_loss', verbose=1, save_best_only=True,mode='min')
    callbacks_list = [checkpoint,checkpoint2,checkpoint3,TensorBoard(log_dir='./log2')]
    train_data = np.asarray(train_data)
    train_label = np.asarray(train_label)
    validate_data = np.asarray(validate_data)
    validate_label = np.asarray(validate_label)
    model.fit(train_data,train_label,epochs=epochs,batch_size=batch_size,validation_data=(validate_data,validate_label),shuffle=True,callbacks=callbacks_list)

if __name__ == "__main__":
    main()