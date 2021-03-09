# !/usr/bin/env python3.6

"""
This script create a file contain the absolute path of all the files in the directories and subdirectories.
The file "train.txt" is used in the darknet-YOLO implementation in order to detect all the images in the test set. 
"""
import os, sys

original_stdout = sys.stdout 

with open('train.txt', 'w') as f:
    sys.stdout = f 
    for root, dirs, files in os.walk(os.path.abspath("/data/yuvalsnir@ef.technion.ac.il/test_set_kaggle_2019/")):
        for file in files:
            print(os.path.join(root, file))
    sys.stdout = original_stdout 