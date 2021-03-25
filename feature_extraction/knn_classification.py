# !/usr/bin/env python3.6

"""
In this script we're using KNN Classifier (from scklearn) to predict the correct class for each test set image. 
Because of the very large embedded data set (4.4 Gb of data) and the very large test set (0.33 Gb of data) the prediction
process was very long and took 121028.36078858376 seconds, i.e. 1.4 days. 
"""

import torch
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd 
from tqdm import tqdm 
import time 

start = time.time()

NEIGHBORS = 7
RESULT_PATH = "predicted_class_embedded_test_K=7.csv"

test = torch.load('embedded_test.pt', map_location=lambda storage, loc: storage) 
data = torch.load('embedded_data.pt', map_location=lambda storage, loc: storage)
labels = torch.load('labels.pt', map_location=lambda storage, loc: storage) 

neigh = KNeighborsClassifier(n_neighbors = NEIGHBORS) 
neigh.fit(data, labels)

"""
# if you would like to predict the nearest neighbor for the whole data set unmark this section 
# and mark the next section
predict_class_list = []
print("Predicting the class for each test set image using K-NN:")
for i in tqdm(range(len(test))): 
    predict_class = neigh.predict([test[i]])
    predict_class_list.append(predict_class)

predict_class_df = pd.DataFrame(predict_class_list)
predict_class_df.to_csv(RESULT_PATH) # save results as CSV file 
"""

# predict the nearest neighbors ONLY for the landmarks images
test_df = pd.read_csv("/data/yuvalsnir@ef.technion.ac.il/test_set_kaggle_2019/test.csv")
landmark_indices = test_df["landmarks"] != 0
landmark_indices = landmark_indices[landmark_indices].index

predict_class_list = []
print("Predicting the class for each test set image using K-NN:")
for i in tqdm(landmark_indices):
    predict_class = neigh.predict([test[i]])
    predict_class_list.append(predict_class)

predict_class_df = pd.DataFrame(predict_class_list)
predict_class_df.to_csv(RESULT_PATH) # save results as CSV file

end = time.time()
print("time: {}".format(end-start))