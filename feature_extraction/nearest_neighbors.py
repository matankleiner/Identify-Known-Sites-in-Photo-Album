# !/usr/bin/env python3.6

"""
In this script we're using NearestNeighbors (from scklearn) to calculate the nearest neighbors and the distances 
of each test set image. 
Because of the very large embedded data set (4.4 Gb of data) and the very large test set (0.33 Gb of data) the prediction
process was very long and took 121272.02406287193 seconds, i.e. 1.4 days. 
"""

import torch
from sklearn.neighbors import NearestNeighbors
import pandas as pd 
import numpy as np 
from tqdm import tqdm 
import time 

start = time.time()

NEIGHBORS = 5 
DIST_PATH = "dist_embedded_test.csv" 
NN_PATH = "nearest_neighbor_embedded_test.csv" 

test = torch.load('embedded_test.pt', map_location=lambda storage, loc: storage) 
data = torch.load('embedded_data.pt', map_location=lambda storage, loc: storage)

neigh = NearestNeighbors(n_neighbors=NEIGHBORS)
neigh.fit(data)

dist_list = []
nearest_neighbor_list = []
print("Calculating each test set image nearest neighbors and distances to each neighbor:")
for i in tqdm(range(len(test))): 
    dist, nearest_neighbor = neigh.kneighbors([test[i]])
    dist_list.append(dist)
    nearest_neighbor_list.append(nearest_neighbor)

dist_np = np.array(dist_list)
dist_np.shape = -1, NEIGHBORS # reshape the list
dist_df = pd.DataFrame(dist_np)
dist_df.to_csv(DIST_PATH) # save dist as CSV file 
nearest_neighbor_np = np.array(nearest_neighbor_list)
nearest_neighbor_np.shape = -1, NEIGHBORS # reshape the list
nearest_neighbor_df = pd.DataFrame(nearest_neighbor_np)
nearest_neighbor_df.to_csv(NN_PATH) # save nearest_neighbors as CSV file 

end = time.time()
print("time: {}".format(end-start))