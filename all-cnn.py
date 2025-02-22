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

from keras.optimizers import SGD
from keras import backend as K

import matplotlib.pyplot as plt  
def show_train_history(train_history, train, validation):  
    plt.plot(train_history.history[train])  
    plt.plot(train_history.history[validation])  
    plt.title('Train History')  
    plt.ylabel(train)  
    plt.xlabel('Epoch')  
    plt.legend(['train', 'validation'], loc='upper left')  
    plt.show()

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

generator = ImageDataGenerator(rotation_range=10,
                               width_shift_range=5./32,
                               height_shift_range=5./32,)

init_shape = (3, 32, 32) if K.image_dim_ordering() == 'th' else (32, 32, 3)

inputs = Input(shape=(32, 32, 3))
x = Conv2D(96, (3, 3), padding='same')(inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Conv2D(96, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Conv2D(96, (3, 3), strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Conv2D(192, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Conv2D(192, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Conv2D(192, (3, 3), strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Conv2D(192, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Conv2D(192, (1, 1), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)

x = Conv2D(10, (1, 1), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)

x = GlobalAveragePooling2D(dim_ordering='default')(x)

predictions = Dense(10, activation='softmax')(x)

# 这部分创建了一个包含输入层和三个全连接层的模型
#model = Model(inputs=inputs, outputs=predictions)


#plot_model(model, "WRN-16-8.png", show_shapes=False)
model = Model(inputs=inputs, outputs=predictions)
sgd = SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)
model.compile(loss="categorical_crossentropy",
              optimizer=sgd, metrics=["acc"])
print("Finished compiling")

#model = load_model("RetrainALLCNN_pruned_one_with_mean_L1.h5")
model.summary()

print("Model loaded.")
def step_decay(epoch):
    if epoch < 200:
        return 0.25
    elif epoch < 250:
        return 0.1
    elif epoch < 300:
        return 0.05
    else:
        return 0.01

train_history = model.fit_generator(generator.flow(trainX, trainY, batch_size=batch_size), steps_per_epoch=len(trainX) // batch_size, epochs=nb_epoch,
                    callbacks=[callbacks.ModelCheckpoint("./ALLCNNparameter_OG-bn.h5",
                                                         monitor="val_acc",
                                                         save_best_only=True,
                                                         verbose=1), callbacks.LearningRateScheduler(step_decay)],
                    validation_data=(testX, testY),
                    validation_steps=testX.shape[0] // batch_size,)

yPreds = model.predict(testX)
yPred = np.argmax(yPreds, axis=1)
yPred = kutils.to_categorical(yPred)
yTrue = testY

accuracy = metrics.accuracy_score(yTrue, yPred) * 100
error = 100 - accuracy
print("Accuracy : ", accuracy)
print("Error : ", error)

show_train_history(train_history,'acc','val_acc')
# model.save('./ALLCNN_OG.h5')
