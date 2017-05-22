import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import cv2
import collections
np.set_printoptions(threshold=np.nan)

exec(open("./File_Paths.py").read())

Dataset = collections.namedtuple('Dataset', ['images', 'labels', 'cls'])
Datasets = collections.namedtuple('Datasets', ['train', 'test'])
images_test = []
cls_test = []
images_train = []
cls_train = []

#Function for cropping images and assigning to test or train set
def pre_process_image(image,  annotations_type, x, y):
    # This function takes a single image as input, 
    # and returns all 256x256 cropped images
    x0 = x - 128
    y0 = y - 128
    height, width = 255, 255

    if y0 > 0 and x0 > 0:
        if x0+height<image.shape[0] and y0+width<image.shape[1]:
            cropped_image = image[x0:x0+height, y0:y0+width,:]
            if (x%2048 < 1024) and (y%2048 < 1024):
                images_test.append(cropped_image)
                cls_test.append(annotations_type)
            else:                
                images_train.append(cropped_image)
                cls_train.append(annotations_type)


#Loop through all set of files to build train and test set            
for single_set_of_file in all_files:
    input_value = cv2.imread(single_set_of_file[0])
    pos_value = cv2.imread(single_set_of_file[1])
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


#Set named tuples
labels = np.zeros((len(cls_train), max(cls_train)+1))
labels[np.arange(len(cls_train)),cls_train] = 1
train = Dataset(images=images_train, labels=labels, cls=cls_train)

labels = np.zeros((len(cls_test), max(cls_test)+1))
labels[np.arange(len(cls_test)),cls_test] = 1
test = Dataset(images=images_test, labels=labels, cls=cls_test)

data = Datasets(train=train, test=test)
print(len(data.train.images))
print(len(data.test.images))





'''
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
