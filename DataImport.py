import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy
numpy.set_printoptions(threshold=numpy.nan)
#tf.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width)

all_files = []
a_patch = []
single_set_of_files = [tf.train.string_input_producer([r'/home/shivang/Downloads/ObjectDetection/COWC/gdo152.ucllnl.org/pub/cowc/datasets/ground_truth_sets/Columbus_CSUAV_AFRL/EO_Run01_s2_301_15_00_31.99319028-Oct-2007_11-00-31.993_Frame_1-124%.png']),
                        tf.train.string_input_producer([r'/home/shivang/Downloads/ObjectDetection/COWC/gdo152.ucllnl.org/pub/cowc/datasets/ground_truth_sets/Columbus_CSUAV_AFRL/EO_Run01_s2_301_15_00_31.99319028-Oct-2007_11-00-31.993_Frame_1-124%_Annotated_Cars.png']),
                        tf.train.string_input_producer([r'/home/shivang/Downloads/ObjectDetection/COWC/gdo152.ucllnl.org/pub/cowc/datasets/ground_truth_sets/Columbus_CSUAV_AFRL/EO_Run01_s2_301_15_00_31.99319028-Oct-2007_11-00-31.993_Frame_1-124%_Annotated_Negatives.png'])]

all_files.append(single_set_of_files)

'''
def pre_process_image(image,  annotations_type, annotations):
    # This function takes a single image as input, 
    # and returns all 256x256 cropped images
    cropped_images = []
    print(annotations)
    for annotation in annotations:
        offset_height = annotation[0] - 128
        offset_width = annotation[1] - 128
        target_height, target_width = 255, 255

        if offset_width > 0 and offset_height > 0:
            if offset_height+target_height<my_img.get_shape().as_list()[1] and offset_width+target_width<my_img.get_shape().as_list()[2]:
                cropped_image = tf.image.crop_to_bounding_box(image, 
                                                        offset_height,
                                                        offset_width, 
                                                        target_height = 255, 
                                                        target_width = 255)
                cropped_images.append(cropped_image)

    return cropped_images

# dataset of patches and labels
# Dataset = collections.namedtuple('Dataset', ['images', 'labels', 'cls'])
# Datasets = collections.namedtuple('Datasets', ['train', 'test'])
'''

# print(c.eval())


#Read all the files
reader = tf.WholeFileReader()
for a_set_of_file in all_files:
    print("BBBBBBBB")
    input_key, input_value = reader.read(a_set_of_file[0])
    pos_key, pos_value = reader.read(a_set_of_file[1])
    neg_key, neg_value = reader.read(a_set_of_file[2])

    my_img = tf.image.decode_png(input_value, channels=0) 
    my_img_pos = tf.image.decode_png(pos_value, channels=1) 
    my_img_neg = tf.image.decode_png(neg_value, channels=1)

    zero = tf.constant(0, dtype=tf.uint8)
    where_pos = tf.not_equal(my_img_pos, zero)
    where_neg = tf.not_equal(my_img_neg, zero)
    indices_pos = tf.where(where_pos)
    indices_neg = tf.where(where_neg)
    print("BBBBBBBB")
    
    # a_patch.append(pre_process_image(my_img, 1, indices_pos))
    # a_patch.append(pre_process_image(my_img, 0, indices_neg))


# shapeOp = tf.shape(a_patch)


