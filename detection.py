import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import cv2
import inception
from inception import transfer_values_cache

inception.maybe_download()
model = inception.Inception()
#########################################################################################################################
# The Sampling stride over the image
window_stride       = 8
# how many patches can we fit in one batch
batch_size          = 64
# skew placed on probability 
log_power           = 16
# numeric threshold for a detection (0 to 255)
threshold           = 196
# The starting layer in your network
start_layer         = 'x:0'
# The final softmax output layer
softmax_layer       = 'softmax_activation/Softmax:0'


window_size     = 255
window_pad      = window_size//2 + 1    #128 pixels for padding
patch_size      = 224
kImage_size     = 192
kImage_pad      = 16    #patch_size = kImage_size + 2*kImage_pad

patches_path_dir    = '/home/shivang/Downloads/ObjectDetection/COWC/gdo152.ucllnl.org/pub/cowc/datasets/testing_scenes/testing_scenes/'
patches_path        = []

#Get all file paths
for root, di, files in os.walk(patches_path_dir):
    file_names = [os.path.join(root, f) for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
    patches_path.extend(file_names)

#Operate on one of the 21 files at a time
#TODO: change file name to filenames
file_name = [file_names[0]]
for aFile in file_name:
    input_image             = cv2.imread(aFile)
    #Padding the whole image
    input_image_pad         = np.zeros((input_image.shape[0]+window_pad*2, input_image.shape[1]+window_pad*2, 3), dtype=np.float64)
    input_image_pad[window_pad:window_pad+input_image.shape[0],window_pad:window_pad+input_image.shape[1],:] = input_image
    
    #Get heat map dimensions
    heat_map_col        = (input_image_pad.shape[1] - kImage_size)//window_stride + 1
    heat_map_row        = (input_image_pad.shape[0] - kImage_size)//window_stride + 1
    heat_map            = np.zeros((heat_map_row, heat_map_col), dtype=np.float64)

    #Get all 192x192 patches from input_image_pad
    #Pad them to make 224x224
    for i in range(heat_map_row):
        aAll_Cropped_Images = []
    
        for j in range(heat_map_col):
            y0  = i*window_stride
            x0  = j*window_stride
            y1  = i*window_stride+kImage_size
            x1  = j*window_stride+kImage_size
            aImage      = input_image_pad[y0:y1,x0:x1,:]
            aImage_pad  = np.zeros((aImage.shape[0]+kImage_pad*2, aImage.shape[1]+kImage_pad*2, 3), dtype=np.float64)
            aImage_pad[kImage_pad:kImage_pad+aImage.shape[0],kImage_pad:kImage_pad+aImage.shape[1],:] = aImage
            aAll_Cropped_Images.append(aImage_pad)

        #Transfer values of all images in a row (265 images)
        transfer_values = transfer_values_cache(images=aAll_Cropped_Images,
                                                model=model)     

        #Pass transfer values to new network
        sess        = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        new_saver   = tf.train.import_meta_graph('./tmp/model.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('./tmp/'))

        graph       = tf.get_default_graph()
        x           = graph.get_tensor_by_name(start_layer)
        y_pred      = graph.get_tensor_by_name(softmax_layer)
        feed_dict   = {x:transfer_values}
        Probabilities = sess.run(y_pred, feed_dict)
        sess.close()
        # print(Probabilities.shape)

        #Fill one row of heat map at a time
        for j in range(heat_map_col):
            P_final         = pow(Probabilities[j][1] - Probabilities[j][0] + 1.0,log_power)/pow(2,log_power)
            heat_map[i][j]  = P_final
    
    # print(heat_map)
    
#########################################################################################################################
def non_max_suppression(input, window_size):
    # input: B x W x H x C
    pooled = tf.nn.max_pool(input, ksize=[1, window_size, window_size, 1], strides=[1,1,1,1], padding='SAME')
    output = tf.where(tf.equal(input, pooled), input, tf.zeros_like(input))

    # NOTE: if input has negative values, the suppressed values can be higher than original
    return output # output: B X W X H x C

x = heatmap.reshape([1,heat_map_row,heat_map_col,1])
inp = tf.Variable(x)
out = non_max_suppression(inp, 3)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
print("After non maximal suppression of heat map: ")
print(out.eval().reshape([heat_map_row,heat_map_col]))

sess.close()



