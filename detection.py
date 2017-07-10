import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.nan)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
import matplotlib.pyplot as plt

import os
import cv2
from numpy import array
from scipy.stats import threshold
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.models import Model
from keras import layers
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras import backend as K

exec(open("./File_Paths.py").read())
from inception_v3 import InceptionV3

#########################################################################################################################
# The Sampling stride over the image
# window_stride       = 8
# skew placed on probability 
log_power           = 2


window_size     = 255
window_pad      = window_size//2 + 1    #128 pixels for padding
patch_size      = 224
kImage_size     = 192
kImage_pad      = 16    #patch_size = kImage_size + 2*kImage_pad

detection_size  = 2048
detection_size_pad = 2048+window_pad*2
input_shape     =(2048+window_pad*2, 2048+window_pad*2, 3)
# input_shape=(224, 224, 3)

#2048x2048 image paths
patch_paths    = []

#Get all 2048x2048 image paths
for root, di, files in os.walk(patch_paths_dir):
    file_names = [os.path.join(root, f) for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
    patch_paths.extend(file_names)


# Load inception model
inception_model = InceptionV3(include_top=False, 
                    weights='imagenet', 
                    input_shape=input_shape, 
                    pooling='avg',
                    trainable=False,
                    classes=num_classes,
                    second_stage=True,
                    model_weights='imagenet_models/my_model_final.h5')

# Get weights of the last fc layer
final_layer_weights = inception_model.get_layer('my_predictions').get_weights()
final_layer_weights[0] = np.expand_dims(final_layer_weights[0], axis=0)
final_layer_weights[0] = np.expand_dims(final_layer_weights[0], axis=0)
# print(final_layer_weights)

##################################################################################################################################
###############################            Making fully convolutional model        ###############################################
if K.image_data_format() == 'channels_first':
    bn_axis = 1
else:
    bn_axis = 3
# Get input
img_input = inception_model.input

# Get mixed10 output
layer_name = 'mixed10'
intermediate_layer_model = Model(inputs=img_input,
                                 outputs=inception_model.get_layer(layer_name).output)
out_mixed10 = intermediate_layer_model(img_input)

# Average pool 6x6 with stride (1, 1) because this output in case of 256x256 images is 6x6
x_new_AP = AveragePooling2D(
                            pool_size=(6, 6),
                            strides=(1, 1),
                            padding='same',
                            data_format=K.image_data_format())(out_mixed10)

# Define 1x1x2 convolution layer
# And apply softmax activation
x_new_conv = Conv2D(
                    2, (1, 1),
                    strides=(1, 1),
                    padding='same',
                    use_bias=True,
                    name='x_new_conv')(x_new_AP)
x_new_conv = BatchNormalization(axis=bn_axis, scale=False)(x_new_conv)
x_new_conv = Activation('softmax')(x_new_conv)
# Make new model
model = Model(img_input, x_new_conv, name='my_fully_conv_inception_v3')

# Set weights of 'x_new_conv'(conv layer) same as the last fc layer of inception model
model.get_layer('x_new_conv').set_weights(final_layer_weights)

# Heatmaps
heat_maps = [] #21 heat maps to be generated

#Operate on one of the 21 files at a time

for aFile in patch_paths[0:1]:
    print(aFile)
    input_image             = cv2.imread(aFile)
    #Padding the whole image
    input_image_pad         = np.zeros((input_image.shape[0]+window_pad*2, input_image.shape[1]+window_pad*2, 3), dtype=np.float32)
    input_image_pad[window_pad:window_pad+input_image.shape[0],window_pad:window_pad+input_image.shape[1],:] = input_image
    input_image_pad = input_image_pad.reshape((1,detection_size_pad,detection_size_pad,3))
    # get prediction from 'my_fully_conv_inception_v3'
    output = model.predict(x=input_image_pad)
    output = np.squeeze(output, axis=0)

    heat_map = np.zeros((output.shape[0], output.shape[1]), dtype=np.float32)
    # Fill each pixel of heat map with p = (o1 − o2 + 1)^16 /2^16 
    for i in range(heat_map.shape[0]):
            for j in range(heat_map.shape[1]):
                P_final         = pow(output[i][j][0] - output[i][j][1] + 1.0,log_power)/pow(2,log_power)
                heat_map[i][j]  = P_final
    # heat_map = threshold(heat_map, threshmin=0.75, newval=0)

    imgplot = plt.imshow(heat_map[:, :])
    plt.show()
    # print(heat_map)
    heat_maps.append(heat_map)


#########################################################################################################################
# def non_max_suppression(input, window_size):
#     # input: B x W x H x C
#     pooled = tf.nn.max_pool(input, ksize=[1, window_size, window_size, 1], strides=[1,1,1,1], padding='SAME')
#     output = tf.where(tf.equal(input, pooled), input, tf.zeros_like(input))

#     # NOTE: if input has negative values, the suppressed values can be higher than original
#     return output # output: B X W X H x C

# #An example for non maximal suppression
# x = heat_maps[0].reshape([1,heat_maps[0].shape[0], heat_maps[0].shape[1], 1])
# inp = tf.Variable(x)
# out = non_max_suppression(inp, 3)

# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# print("After non maximal suppression of heat map: ")
# # print(out.eval().reshape([heat_maps[0].shape[0],heat_maps[0].shape[1]]))

# sess.close()
