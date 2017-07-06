import os
import numpy as np

#Do NOT change these
img_size            = 224
original_img_size   = 256
######################################################################################################################################
# Classification training
patch_sets_path = '/home/shivang/Downloads/ObjectDetection/COWC/gdo152.ucllnl.org/pub/cowc/datasets/patch_sets/detection/'
# Data Paths for test and train images

dirs = ["Columbus_CSUAV_AFRL/",
        "Potdam_ISPRS/",
        "Selwyn_LINZ/",
        "Toronto_ISPRS/",
        "Utah_AGRC/",
        "Vaihingen_ISPRS/"
        ]
'''
dirs = ["dummy1/",
        "dummy2/"]
'''
dirs = [os.path.join(patch_sets_path, f) for f in dirs]

# File paths as required by "flow_from_directory"
# for getting data-on-the-fly from the directories
train_car_path      = os.path.join(patch_sets_path, "data/train/car/")
train_not_car_path  = os.path.join(patch_sets_path, "data/train/not_car/")
test_car_path       = os.path.join(patch_sets_path, "data/test/car/")
test_not_car_path   = os.path.join(patch_sets_path, "data/test/not_car/")

if not os.path.exists(train_car_path):
    os.makedirs(train_car_path)
if not os.path.exists(train_not_car_path):
    os.makedirs(train_not_car_path)
if not os.path.exists(test_car_path):
    os.makedirs(test_car_path)
if not os.path.exists(test_not_car_path):
    os.makedirs(test_not_car_path)

######################################################################################################################################
# Detection testing
#Path for 2048x2048 images
patch_paths_dir = '/home/shivang/Downloads/ObjectDetection/COWC/gdo152.ucllnl.org/pub/cowc/datasets/testing_scenes/testing_scenes/'
