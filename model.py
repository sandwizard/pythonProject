import time
import tensorflow as tf

import numpy
import cv2
from tensorflow import keras
from keras.layers import LeakyReLU
import pickle
import data_preprocess as d
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Activation
from tensorflow.keras.callbacks import TensorBoard




################################optimization area ####################################################
# test different image sizes
img_size = (255, 255)
#########################################################################
size_path = '_' + str(img_size[0]) + '_' + str(img_size[1]) + '.pickle'

try:
    X = pickle.load(open('./X' + size_path, 'rb'))

    y = pickle.load(open('./y' + size_path, 'rb'))

except Exception as e:
    # this creates new data based on img size
    d.create_trainning_set(img_size)
    X = pickle.load(open('./X' + size_path, 'rb'))

    y = pickle.load(open('./y' + size_path, 'rb'))


    ################################### optimization area feaarure scaling ############################
    # 1. mean normalization{ (val - avg)/max}
    # 2 .{(val - avg)/staandard deviation}
    # 3. divide by max val/max
    # for now no3

X = X / 255.0






print(X.shape[1:])
dense_layers = [0]
layer_sizes = [8]

conv_layers = [3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:

        for conv_layer in conv_layers:
            Name = '{}-conv-{}-nodes-{}-dense-{}'.format(conv_layer, layer_size, dense_layer, int(time.time()))
            tensorBoard = TensorBoard(log_dir='logs/{}'.format(Name))
                    #########################################
            model = Sequential()
            model.add(Conv2D(layer_size, (3,3),data_format="channels_last",input_shape=X.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))

            for k in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())




                    # no softmax get stuck at minima here same use

            model.add(Dense(1))
            model.add(Activation('sigmoid'))
                    # try find best optimizer i.e the function which brings us closer to the theta values eg gradient descent
                    # sigmoid may case sub optimal local minima
                    # use SGD if i get stuck in local minima has momentum an goes wrong directiotion initially
                    # nag quickly regaing correct direction
                    # try adelta/ adam as firt maybe best choices



                    # caterogical crossentropy best cost func for multiclass

            model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='Adam')
            model.summary()


            model.fit(X, y, batch_size=16, epochs=10,validation_split=0.3, callbacks=[tensorBoard])


model.save('./16x3-CNN-l_mod3.model')

#l_mod best
# note model check pioint can be used to stop mid trainning when we have best loss or etc..