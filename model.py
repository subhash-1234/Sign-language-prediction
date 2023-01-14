import numpy as np
import matplotlib.pyplot as plot
# import pandas as pd
import cv2 as c
import os
import random
import time
import pickle
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D

dir = "/home/subhash/Documents/project"

label = ["1","2","3","4"]


#print(img_array.shape)

img_size = 50
# new_array = c.resize(img_array,(img_size,img_size))
# plot.imshow(new_array,cmap="gray")
# plot.show()

training_data = []
def creating_training_data():
    for l in label:
        path = os.path.join(dir, l)
        class_num = label.index(l)
        for img in os.listdir(path):
            try:
                img_array = c.imread(os.path.join(path, img),c.IMREAD_GRAYSCALE)
                new_array = c.resize(img_array, (img_size, img_size))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
creating_training_data()
print(len(training_data))


X = []
Y = []

for features,label in   training_data:
    X.append(features)
    Y.append(label)
X = np.array(X).reshape(-1,img_size,img_size,1)
# pickle_out = open("X.pickle","wb")
# pickle.dump(X,pickle_out)
# pickle_out.close()
#
# pickle_out = open("Y.pickle","wb")
# pickle.dump(Y,pickle_out)
# pickle_out.close()


X = pickle.load(open("X.pickle","rb"))
Y = pickle.load(open("Y.pickle","rb"))
X = X/255
X  = np.array(X)
Y = np.array(Y)
dense_layers = [1]
layer_sizes = [64]
conv_layers = [3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)

            model = Sequential()

            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())

            for _ in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'],
                          )

            model.fit(X, Y,
                      batch_size=32,
                      epochs=10,
                      validation_split=0.3,
                      callbacks=[tensorboard])

model.save('64x3-CNN2.model')