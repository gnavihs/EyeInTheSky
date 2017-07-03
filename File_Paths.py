import os
import numpy as np

######################################################################################################################################
# Classification training
patch_sets_path = '/home/shivang/Downloads/ObjectDetection/COWC/gdo152.ucllnl.org/pub/cowc/datasets/patch_sets/detection/'
# Data Paths for test and train images
dirs = ["Columbus_CSUAV_AFRL/",
        "Potdam_ISPRS/",
        "Selwyn_LINZ/",
        "Toronto_ISPRS/",
        "Utah_AGRC/",
        "Vaihingen_ISPRS/"]
'''
dirs = ["dummy1/",
        "dummy2/"]
'''
dirs = [os.path.join(patch_sets_path, f) for f in dirs]


######################################################################################################################################
# Detection testing
#Path for 2048x2048 images
patch_paths_dir = '/home/shivang/Downloads/ObjectDetection/COWC/gdo152.ucllnl.org/pub/cowc/datasets/testing_scenes/testing_scenes/'
