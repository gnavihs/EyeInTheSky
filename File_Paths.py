

all_files = []
single_set_of_files = [tf.train.string_input_producer([r'/home/shivang/Downloads/ObjectDetection/COWC/gdo152.ucllnl.org/pub/cowc/datasets/ground_truth_sets/Columbus_CSUAV_AFRL/EO_Run01_s2_301_15_00_31.99319028-Oct-2007_11-00-31.993_Frame_1-124%.png']),
                        tf.train.string_input_producer([r'/home/shivang/Downloads/ObjectDetection/COWC/gdo152.ucllnl.org/pub/cowc/datasets/ground_truth_sets/Columbus_CSUAV_AFRL/EO_Run01_s2_301_15_00_31.99319028-Oct-2007_11-00-31.993_Frame_1-124%_Annotated_Cars.png']),
                        tf.train.string_input_producer([r'/home/shivang/Downloads/ObjectDetection/COWC/gdo152.ucllnl.org/pub/cowc/datasets/ground_truth_sets/Columbus_CSUAV_AFRL/EO_Run01_s2_301_15_00_31.99319028-Oct-2007_11-00-31.993_Frame_1-124%_Annotated_Negatives.png'])]

all_files.append(single_set_of_files)