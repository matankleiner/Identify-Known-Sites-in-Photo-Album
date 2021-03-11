# !/usr/bin/env python3.6

"""
This script create a file contain the absolute path of all the images in "test.csv". 
"test.csv" is a csv file that contain images ehich are or out of domain or their classes are both in the test set and in the train set.  
The file "train.txt" is used in the darknet-YOLO implementation in order to detect object in the test set images. 
"""
import sys
import pandas as pd 

PATH = "/data/yuvalsnir@ef.technion.ac.il/test_set_kaggle_2019/"

original_stdout = sys.stdout 

test_df = pd.read_csv("/data/yuvalsnir@ef.technion.ac.il/test_set_kaggle_2019/test.csv") 

with open('train.txt', 'w') as f:
    sys.stdout = f 
    for i in range(test_df.shape[0]):
        print(PATH + test_df['id'][i][0] + "/" + test_df['id'][i][1] + "/" + test_df['id'][i][2] + "/" +  test_df['id'][i] + ".jpg")
    sys.stdout = original_stdout 