import numpy as np
import sklearn.metrics as metrics
from keras.models import model_from_json, load_model
from keras.datasets import cifar10
import keras.callbacks as callbacks
import keras.utils.np_utils as kutils
from keras.preprocessing.image import ImageDataGenerator
#from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Add, Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from keras import backend as K

import tensorflow as tf

batch_size = 128
nb_epoch = 350
img_rows, img_cols = 32, 32

(trainX, trainY), (testX, testY) = cifar10.load_data()

trainX = trainX.astype('float32')
trainX = (trainX - trainX.mean(axis=0)) / (trainX.std(axis=0))
testX = testX.astype('float32')
testX = (testX - testX.mean(axis=0)) / (testX.std(axis=0))

trainY = kutils.to_categorical(trainY)
testY = kutils.to_categorical(testY)

model = tf.contrib.keras.models.load_model('./ALLCNNparameter_OG-no-bn.h5')

# 驗證模型
score = model.evaluate(testX, testY, verbose=0)

# 輸出結果
print('Test loss:', score[0])
print('Test accuracy:', score[1]*100)
print('Test error:', (1-score[1])*100)
