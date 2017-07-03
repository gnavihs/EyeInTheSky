import os, os.path
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import cv2
import collections
import re
import sys
from random import shuffle
from random import randint
from numpy import array
import random
exec(open("./File_Paths.py").read())

############################################################################
random.seed(1234)
positive_label      = re.compile(r'^car')

#Do NOT change these
image_size          = 224
num_classes         = 2

# Intialize lists and variables
file_paths_train    = []
file_paths_test     = []

def crop_center(img,cropx,cropy,shift):
    y,x,c = img.shape
    x_shift = randint(-shift,shift)
    y_shift = randint(-shift,shift)

    startx = x//2 - cropx//2 + x_shift
    starty = y//2 - cropy//2 + y_shift  
    return img[starty:starty+cropy, startx:startx+cropx, :]

def preprocess_input(x):
    x = np.true_divide(x, 255)
    # x /= 255.
    x -= 0.5
    x *= 2.
    return x

for dir in dirs:
    path_train  = dir + 'train/'
    path_test   = dir + 'test/'

    for root, di, files in os.walk(path_train):
        file_names = [os.path.join(path_train, f) for f in os.listdir(path_train) if os.path.isfile(os.path.join(path_train, f))]
        file_paths_train.extend(file_names)

    for root, di, files in os.walk(path_test):
        file_names = [os.path.join(path_test, f) for f in os.listdir(path_test) if os.path.isfile(os.path.join(path_test, f))]
        file_paths_test.extend(file_names)

#Shuffle to randomize training
shuffle(file_paths_train)
shuffle(file_paths_test)

file_paths_train = array(file_paths_train)
file_paths_test = array(file_paths_test)
print("No. of training images: ", len(file_paths_train))
print("No. of testing images: ", len(file_paths_test))


#Read all the training images and labels
X_train = []
labels_train = []

for aFile in file_paths_train:
    #Image decoding
    input_value         = cv2.imread(aFile)
    input_value         = preprocess_input(input_value)
    #Get a 224x224 from 256x256 image which is cropped randomly around center
    input_value_crop    = np.zeros((image_size, image_size, 3), dtype=np.float32)
    input_value_crop    = crop_center(input_value,image_size,image_size, 12)

    X_train.append(input_value_crop)
    #Label decoding
    #'1' for car, '0' for not car
    aFileName = os.path.basename(aFile)
    mo = positive_label.search(aFileName)
    if mo:
        labels_train.append(1)
    else:
        labels_train.append(0)

#Converting labels into one hot array
Y_train    = np.zeros((len(labels_train), num_classes))
Y_train[np.arange(len(labels_train)),labels_train] = 1

X_train = array(X_train)
# print("labels_id: ",labels_train)
# print("Y_train: ",Y_train)
# print("file_paths: ",file_paths_train)

#Read all the testing images and labels
X_test = []
labels_test = []

for aFile in file_paths_test:
    #Image decoding
    input_value         = cv2.imread(aFile)
    input_value         = preprocess_input(input_value)
    #Get a 224x224 from 256x256 image which is center cropped
    input_value_crop    = np.zeros((image_size, image_size, 3), dtype=np.float32)
    input_value_crop    = crop_center(input_value,image_size,image_size, 0)

    X_test.append(input_value_crop)
    #Label decoding
    #'1' for car, '0' for not car
    aFileName = os.path.basename(aFile)
    mo = positive_label.search(aFileName)
    if mo:
        labels_test.append(1)
    else:
        labels_test.append(0)

#Converting labels into one hot array
Y_test    = np.zeros((len(labels_test), num_classes))
Y_test[np.arange(len(labels_test)),labels_test] = 1

X_test = array(X_test)
# print("labels_id: ",labels_test)
# print("Y_test: ",Y_test)
# print("file_paths: ",file_paths_test)

print("DATA IMPORT: All the train and test data loaded in X_train, Y_train, X_test, Y_test")