import os, os.path
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import pickle
import numpy as np
import cv2
import collections
from scipy.ndimage import rotate
np.set_printoptions(threshold=np.nan)
import inception
from inception import transfer_values_cache
import re
import sys
from random import shuffle
exec(open("./configuration.py").read())
exec(open("./File_Paths.py").read())
from random import randint
############################################################################

total_train_images  = 0
total_test_images   = 0
all_labels_train    = []
all_labels_test     = []
positive_label      = re.compile(r'^car')
image_size          = 224

#Getting total number of train and test images
for dir in dirs:
    path_train  = dir + 'train/'
    path_test   = dir + 'test/'

    for root, di, files in os.walk(path_train):
        total_train_images += len(files)

    for root, di, files in os.walk(path_test):
        total_test_images += len(files)

print("Total training-images", total_train_images)
print("Total testing-images", total_test_images)

#Getting total number of train and test cache files
total_train_cache_files = total_train_images//cache_batch_size + 1
total_test_cache_files  = total_test_images//cache_batch_size + 1
#############################################################################

file_path_cache_train   = os.path.join(cache_data_path, 'inception_train_')
file_path_cache_test    = os.path.join(cache_data_path, 'inception_test_')
cache_extension         = '.pkl'

inception.maybe_download()
model = inception.Inception()

def crop_center(img,cropx,cropy,shift):
    y,x,c = img.shape
    x_shift = randint(-shift,shift)
    y_shift = randint(-shift,shift)

    startx = x//2 - cropx//2 + x_shift
    starty = y//2 - cropy//2 + y_shift  
    return img[starty:starty+cropy, startx:startx+cropx, :]
############################################################################
#save training images
if not os.path.exists(file_path_cache_train + '0' + cache_extension):
    # Intialize lists and varaibles
    file_paths_train    = []
    file_paths_test     = []
    start   = 0
    end     = start + cache_batch_size

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

    #Fill one cache file at a time
    for i in range(total_train_cache_files):
        print("Saving train-images")
        images = []

        if end < len(file_paths_train):
            # This gives path of number of files in cache_batch_size
            image_paths = file_paths_train[start:end]
        else:
            image_paths = file_paths_train[start:len(file_paths_train)]

        for aFile in image_paths:
            #Image decoding
            input_value         = cv2.imread(aFile)
            #Get a 224x224 from 256x256 image
            input_value_crop    = np.zeros((image_size, image_size, 3), dtype=np.float32)
            input_value_crop    = crop_center(input_value,image_size,image_size, 12)

            images.append(input_value_crop)
            #Label decoding
            aFileName = os.path.basename(aFile)
            mo = positive_label.search(aFileName)
            if mo:
                all_labels_train.append(1)
            else:
                all_labels_train.append(0)

        start = end
        end = start+cache_batch_size
        #Calculating transfer values
        transfer_values_train = transfer_values_cache(images=images,
                                                      model=model)      
        #Storing transfer values
        #This will act as input to new network
        cache_path = file_path_cache_train + str(i) + cache_extension
        with open(cache_path, mode='wb') as file:
            pickle.dump(transfer_values_train, file)
        print("- Data saved to cache-file: " + cache_path)

    #Storing labels for all the images in a single cache file
    cache_path_labels = file_path_cache_train + 'labels' + cache_extension
    # print(all_labels_train)
    with open(cache_path_labels, mode='wb') as file:
        pickle.dump(all_labels_train, file)
    print("- Labels saved to cache-file: " + cache_path_labels)
    # print(all_labels_train)

    start = 0
    end = start+cache_batch_size
    #Fill one cache file at a time
    for i in range(total_test_cache_files):
        print("Saving test-images")
        images = []

        if end < len(file_paths_test):
            # This gives path of number of files in cache_batch_size
            image_paths = file_paths_test[start:end]
        else:
            image_paths = file_paths_test[start:len(file_paths_test)]

        for aFile in image_paths:
            #Image decoding
            input_value = cv2.imread(aFile)
            #Get a 224x224 from 256x256 image
            input_value_crop    = np.zeros((image_size, image_size, 3), dtype=np.float32)
            input_value_crop    = crop_center(input_value,image_size,image_size, 0)
            
            images.append(input_value)
            #Label decoding
            aFileName = os.path.basename(aFile)
            mo = positive_label.search(aFileName)
            if mo:
                all_labels_test.append(1)
            else:
                all_labels_test.append(0)

        start = end
        end = start+cache_batch_size
        #Calculating transfer values
        transfer_values_test = transfer_values_cache(images=images,
                                                      model=model)      
        #Storing transfer values
        #This will act as input to new network
        cache_path = file_path_cache_test + str(i) + cache_extension
        with open(cache_path, mode='wb') as file:
            pickle.dump(transfer_values_test, file)
        print("- Data saved to cache-file: " + cache_path)

    #Storing labels for all the images in a single cache file
    cache_path_labels = file_path_cache_test + 'labels' + cache_extension
    # print(all_labels_test)
    with open(cache_path_labels, mode='wb') as file:
        pickle.dump(all_labels_test, file)
    print("- Labels saved to cache-file: " + cache_path_labels)
else:
    print("Training and testing images already saved")


model.close()