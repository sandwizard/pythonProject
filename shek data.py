
import time
import tensorflow as tf
from imblearn.over_sampling import SMOTE
import numpy
from PIL import  Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
img_size = (255,255)
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

withmask =0
mask_worn_incorrect =0
without_maks =0
total = len(y)
for i in y:
    p = numpy.where(i == i.max())
    if p[0][0] == 0:
        withmask+=1

    if p[0][0] == 1:
        mask_worn_incorrect +=1

    if p[0][0] == 2:
        without_maks +=1
print("withmask ="+ str(withmask) +" percentage "+ str(str(withmask/total)))
print("incorrect_mask ="+ str(mask_worn_incorrect)+" percentage "+ str(str(mask_worn_incorrect/total)))
print("without_mask ="+ str(without_maks)+" percentage  "+ str(str(without_maks/total)))










