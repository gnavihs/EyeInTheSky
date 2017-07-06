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

####################################################################################################################################
random.seed(1234)
positive_label      = re.compile(r'^car')

# Intialize lists and variables
file_paths_train    = []
file_paths_test     = []

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
print("No. of training images:\t", len(file_paths_train))
print("No. of testing images:\t", len(file_paths_test))

########################################################################################################################################


for aFile in file_paths_train:
    #Label decoding
    #move in car folder for positive_label, move in not_car folder for negative_label
    aFileName = os.path.basename(aFile)
    mo = positive_label.search(aFileName)
    if mo:
        os.rename(aFile, os.path.join(patch_sets_path, train_car_path, aFileName))
    else:
        os.rename(aFile, os.path.join(patch_sets_path, train_not_car_path, aFileName))

for aFile in file_paths_test:
    #Label decoding
    #move in car folder for positive_label, move in not_car folder for negative_label
    aFileName = os.path.basename(aFile)
    mo = positive_label.search(aFileName)
    if mo:
        os.rename(aFile, os.path.join(patch_sets_path, test_car_path, aFileName))
    else:
        os.rename(aFile, os.path.join(patch_sets_path, test_not_car_path, aFileName))


#Checking if there is no overlap of names between images of 6 locations
print("AFTER MOVING:")
file_paths_train    = []
file_paths_test     = []

for root, di, files in os.walk(train_car_path):
    file_names = [os.path.join(train_car_path, f) for f in os.listdir(train_car_path) if os.path.isfile(os.path.join(train_car_path, f))]
    file_paths_train.extend(file_names)

for root, di, files in os.walk(train_not_car_path):
    file_names = [os.path.join(train_not_car_path, f) for f in os.listdir(train_not_car_path) if os.path.isfile(os.path.join(train_not_car_path, f))]
    file_paths_train.extend(file_names)

for root, di, files in os.walk(test_car_path):
    file_names = [os.path.join(test_car_path, f) for f in os.listdir(test_car_path) if os.path.isfile(os.path.join(test_car_path, f))]
    file_paths_test.extend(file_names)

for root, di, files in os.walk(test_not_car_path):
    file_names = [os.path.join(test_not_car_path, f) for f in os.listdir(test_not_car_path) if os.path.isfile(os.path.join(test_not_car_path, f))]
    file_paths_test.extend(file_names)

print("No. of training images:\t", len(file_paths_train))
print("No. of testing images:\t", len(file_paths_test))

