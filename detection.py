import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#########################################################################################################################
#Read all 2048x2048 patches

patches_path_dir = '/home/shivang/Downloads/ObjectDetection/COWC/gdo152.ucllnl.org/pub/cowc/datasets/testing_scenes/testing_scenes/'
patches_path = [] 

for root, di, files in os.walk(patches_path_dir):
    file_names = [os.path.join(root, f) for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
    patches_path.extend(file_names)

# print(patches_path[0])

#########################################################################################################################
def non_max_suppression(input, window_size):
    # input: B x W x H x C
    pooled = tf.nn.max_pool(input, ksize=[1, window_size, window_size, 1], strides=[1,1,1,1], padding='SAME')
    output = tf.where(tf.equal(input, pooled), input, tf.zeros_like(input))

    # NOTE: if input has negative values, the suppressed values can be higher than original
    return output # output: B X W X H x C

x = np.array([[3,2,1,4,2,3],[1,4,2,1,5,2],[2,2,3,2,1,3]], dtype=np.float32).reshape([1,3,6,1])
inp = tf.Variable(x)
out = non_max_suppression(inp, 3)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
print(out.eval().reshape([3,6]))
'''
[[ 0.  0.  0.  0.  0.  0.]
 [ 0.  4.  0.  0.  5.  0.]
 [ 0.  0.  0.  0.  0.  0.]]
'''

sess.close()



