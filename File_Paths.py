import os
import numpy as np
patch_sets_path = '/home/shivang/Downloads/ObjectDetection/COWC/gdo152.ucllnl.org/pub/cowc/datasets/patch_sets/detection/'
# Data Paths for test and train images
'''
dirs = ["Columbus_CSUAV_AFRL/",
        "Potdam_ISPRS/",
        "Selwyn_LINZ/",
        "Toronto_ISPRS/",
        "Utah_AGRC/",
        "Vaihingen_ISPRS/"]
'''

dirs = ["dummy1/",
        "dummy2/"]

dirs = [os.path.join(patch_sets_path, f) for f in dirs]
# print(dirs)

#for training 308,988 patches and testing 79,447 patches  
cache_batch_size = 6


