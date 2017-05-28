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
exec(open("./configuration.py").read())
exec(open("./File_Paths.py").read())

############################################################################

total_train_images = 0
total_test_images = 0
cache_batch_size = 10

#Getting total number of train and test images
for dir in dirs:
    path_train = dir + 'train/'
    path_test = dir + 'test/'

    for root, di, files in os.walk(path_train):
        total_train_images += len(files)

    for root, di, files in os.walk(path_test):
        total_test_images += len(files)

print("Total training-images", total_train_images)
print("Total testing-images", total_test_images)

#Getting total number of train and test cache files
total_train_cache_files = total_train_images//cache_batch_size + 1
total_test_cache_files = total_test_images//cache_batch_size + 1
#############################################################################

file_path_cache_train = os.path.join(cache_data_path, 'inception_train_')
file_path_cache_test = os.path.join(cache_data_path, 'inception_test_')
cache_extension = '.pkl'

num_classes = 2
inception.maybe_download()
model = inception.Inception()

if not os.path.exists(file_path_cache_train + '0' + cache_extension):
    print("Saving training-images")
    # Intialize values
    start = 0
    end = start + cache_batch_size
    j = 0

    #Fill one cache file at a time
    for i in range(total_train_cache_files):
        flag = False
        file_paths = []

        # This while loop gives path of number of files in cache_batch_size
        while flag == False:
            if j < len(dirs):
                path_train = dirs[j] + 'train/'
            else:
                break
            file_names = [os.path.join(path_train, f) for f in os.listdir(path_train) if os.path.isfile(os.path.join(path_train, f))]
            if end > len(file_names):
                file_paths.extend(file_names[start:len(file_names)])
                diff = cache_batch_size - (len(file_names) - start)
                start = 0 
                end = start + diff
                j += 1
                flag = False
            else:
                file_paths.extend(file_names[start:end])
                start = end
                end = start + cache_batch_size
                flag = True
        
        images = []
        for aFile in file_paths:
            input_value = cv2.imread(aFile)
            images.append(input_value)
        transfer_values_train = transfer_values_cache(images=images,
                                                      model=model)      

        cache_path = file_path_cache_train + str(i) + cache_extension
        with open(cache_path, mode='wb') as file:
            pickle.dump(transfer_values_train, file)
        print("- Data saved to cache-file: " + cache_path)
else:
    print("Training-images already saved")

# Same thing for test images
if not os.path.exists(file_path_cache_test + '0' + cache_extension):
    print("Saving testing-images")
    # Intialize values
    start = 0
    end = start + cache_batch_size
    j = 0

    #Fill one cache file at a time
    for i in range(total_test_cache_files):
        flag = False
        file_paths = []

        # This while loop gives path of number of files in cache_batch_size
        while flag == False:
            if j < len(dirs):
                path_test = dirs[j] + 'test/'
            else:
                break
            file_names = [os.path.join(path_test, f) for f in os.listdir(path_test) if os.path.isfile(os.path.join(path_test, f))]
            if end > len(file_names):
                file_paths.extend(file_names[start:len(file_names)])
                diff = cache_batch_size - (len(file_names) - start)
                start = 0 
                end = start + diff
                j += 1
                flag = False
            else:
                file_paths.extend(file_names[start:end])
                start = end
                end = start + cache_batch_size
                flag = True
        
        images = []
        for aFile in file_paths:
            input_value = cv2.imread(aFile)
            images.append(input_value)
        transfer_values_test = transfer_values_cache(images=images,
                                                      model=model)      

        cache_path = file_path_cache_test + str(i) + cache_extension
        with open(cache_path, mode='wb') as file:
            pickle.dump(transfer_values_test, file)
        print("- Data saved to cache-file: " + cache_path)
else:
    print("Testing-images already saved")


'''
print("Processing Inception transfer-values for training-images ...")

labels_train =  data.train.labels
labels_test =  data.test.labels

transfer_len = model.transfer_len
'''










'''
Dataset = collections.namedtuple('Dataset', ['images', 'labels', 'cls'])
Datasets = collections.namedtuple('Datasets', ['train', 'test'])
images_test = []
cls_test = []
images_train = []
cls_train = []

#Function for rotating images(augmentation) and adding in data set
def rotate_image(image, angles, cls_label, dataset_type):
    for angle in angles:
        rot_cropped_image = rotate(image, angle, reshape=False)
        if dataset_type == 'test':
            images_test.append(rot_cropped_image)
            cls_test.append(cls_label)
        elif dataset_type == 'train':
            images_train.append(rot_cropped_image)
            cls_train.append(cls_label)


def pre_process_image(image,  cls_label, x, y):
    # This function takes a single image as input, 
    # and returns all 256x256 cropped images
    x0 = x - 128 # x coordinate of top right corner
    y0 = y - 128 # y coordinate of top right corner
    height, width = 256, 256 # height and width of image
    angles = [0, 15, 30, 45] # angles by which to rotate

    if y0 > 0 and x0 > 0:
        if x0+height<image.shape[0] and y0+width<image.shape[1]:
            cropped_image = image[x0:x0+height, y0:y0+width,:]
            if (x%2048 < 1024) and (y%2048 < 1024):
                rotate_image(cropped_image, angles, cls_label, 'test')
            else:
                rotate_image(cropped_image, angles, cls_label, 'train')

################################################################################
#Loop through all set of files to build train and test set            
for single_set_of_file in all_files:
    #Read whole image
    input_value = cv2.imread(single_set_of_file[0])
    #Read positive annotations
    pos_value = cv2.imread(single_set_of_file[1])
    #Read negative annotations
    neg_value = cv2.imread(single_set_of_file[2])

    #Cropping positively annotated images
    pos_annotations_index = np.nonzero(pos_value)
    pos_annotations_index_x = pos_annotations_index[0]
    pos_annotations_index_y = pos_annotations_index[1]
    for i in range(len(pos_annotations_index_x)):
        pre_process_image(input_value, 1, pos_annotations_index_x[i], pos_annotations_index_y[i])

    #Cropping negatively annotated images
    neg_annotations_index = np.nonzero(neg_value)
    neg_annotations_index_x = neg_annotations_index[0]
    neg_annotations_index_y = neg_annotations_index[1]
    for i in range(len(neg_annotations_index_x)):
        pre_process_image(input_value, 0, neg_annotations_index_x[i], neg_annotations_index_y[i])

#############################################################################################################
#Set named tuples
labels = np.zeros((len(cls_train), max(cls_train)+1))
labels[np.arange(len(cls_train)),cls_train] = 1
train = Dataset(images=images_train, labels=labels, cls=cls_train)

labels = np.zeros((len(cls_test), max(cls_test)+1))
labels[np.arange(len(cls_test)),cls_test] = 1
test = Dataset(images=images_test, labels=labels, cls=cls_test)

data = Datasets(train=train, test=test)

#Size of dataset
print("Number of images in train set", len(data.train.images))
# print(len(data.train.labels))
print("Number of images in test set", len(data.test.images))
# print(len(data.test.labels))






print("AAAAAAAAA")
reader = tf.WholeFileReader()
for single_set_of_file in all_files:
    print("BBBBBBBB")
    input_key, input_value = reader.read(single_set_of_file[0])
    pos_key, pos_value = reader.read(single_set_of_file[1])
    neg_key, neg_value = reader.read(single_set_of_file[2])

    my_img = tf.image.decode_png(input_value, channels=0) 
    my_img_pos = tf.image.decode_png(pos_value, channels=1) 
    my_img_neg = tf.image.decode_png(neg_value, channels=1)

    zero = tf.constant(0, dtype=tf.uint8)
    where_pos = tf.not_equal(my_img_pos, zero)
    where_neg = tf.not_equal(my_img_neg, zero)
    indices_pos = tf.where(where_pos)
    indices_neg = tf.where(where_neg)
    print(tf.shape(indices_pos))
    print("CCCCCCCC")
    pre_process_image(my_img, 1, indices_pos)
    # a_patch.append(pre_process_image(my_img, 0, indices_neg))


# shapeOp = tf.shape(a_patch)

'''
