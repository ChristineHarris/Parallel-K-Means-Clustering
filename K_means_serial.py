# Import Python modules
from __future__ import division
import numpy as np
import sklearn.datasets as skl
import matplotlib.pyplot as plt
from multiprocessing import Pool
import timeit
from sklearn.cluster import KMeans



class K_Means(object):
    # Initialize input values n_clusters and max_iter   
    def __init__(self, n_clusters, max_iter):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    # Function that assigns points to a cluster
    def assign_points_to_cluster(self, X):
        # Label points according to the minimum euclidean distance
        self.labels_ = [self._nearest(self.cluster_centers_, x) for x in X]
        # Map labels to data points
        indices=[]
        for j in range(self.n_clusters):
            cluster=[]
            for i, l in enumerate(self.labels_):
                if l==j: cluster.append(i)
            indices.append(cluster)
        X_by_cluster = [X[i] for i in indices]
        return X_by_cluster
    
    # Function that randomly selects initial centroids
    def initial_centroid(self, X):
        initial = np.random.permutation(X.shape[0])[:self.n_clusters]
        return X[initial]

    # Function that updates centroids and repeats 
    # assign_points_to_cluster until convergence or max_iter is reached
    def fit(self, X):  
        # initialize centroids      
        self.cluster_centers_ = self.initial_centroid(X)
        # process of assigning points to clusters until convergence or until max_iter is reached
        for i in range(self.max_iter):
            X_by_cluster = self.assign_points_to_cluster(X)
            # calculate the new centers 
            new_centers=[c.sum(axis=0)/len(c) for c in X_by_cluster]
            new_centers = [arr.tolist() for arr in new_centers]
            old_centers=self.cluster_centers_
            # if the new centroid are the same as the old centroids then the algorithm has converged
            if np.all(new_centers == old_centers): 
                self.number_of_iter=i
                break;
            else: 
                # set self.cluster_centers_ as new centers 
                self.cluster_centers_ = new_centers
        self.number_of_iter=i
        return self
    
    # Function that calculates the minimum euclidean distance
    def _nearest(self, clusters, x):
        return np.argmin([self._distance(x, c) for c in clusters])
    
    # Function to calculate euclidean distance between two points
    def _distance(self, a, b):
        return np.sqrt(((a - b)**2).sum())

    # Function that returns predicted clusters for each point
    def predict(self, X):
        return self.labels_
    


sim1 = []

TEST_CODE1 = """
kmeans = K_Means(n_clusters = 3, max_iter = 500)
kmeans.fit(X)
"""

SETUP_CODE = """
import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=500, centers=3, cluster_std=0.60, random_state=0)
from __main__ import K_Means
"""
sim1.append(timeit.timeit(stmt=TEST_CODE1,setup=SETUP_CODE,number=50)/50)

print ("1")


SETUP_CODE = """
import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=1000, centers=3, cluster_std=0.60, random_state=0)
from __main__ import K_Means
"""
sim1.append(timeit.timeit(stmt=TEST_CODE1,setup=SETUP_CODE,number=50)/50)

print ("2")

SETUP_CODE = """
import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=5000, centers=3, cluster_std=0.60, random_state=0)
from __main__ import K_Means
"""
sim1.append(timeit.timeit(stmt=TEST_CODE1,setup=SETUP_CODE,number=50)/50)

print ("3")

SETUP_CODE = """
import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=10000, centers=3, cluster_std=0.60, random_state=0)
from __main__ import K_Means
"""

sim1.append(timeit.timeit(stmt=TEST_CODE1,setup=SETUP_CODE,number=50)/50)

print ("4")

SETUP_CODE = """
import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=50000, centers=3, cluster_std=0.60, random_state=0)
from __main__ import K_Means
"""
sim1.append(timeit.timeit(stmt=TEST_CODE1,setup=SETUP_CODE,number=50)/50)

print ("5")

SETUP_CODE = """
import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=100000, centers=3, cluster_std=0.60, random_state=0)
from __main__ import K_Means
"""

sim1.append(timeit.timeit(stmt=TEST_CODE1,setup=SETUP_CODE,number=50)/50)

print ("6")

SETUP_CODE = """
import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=200000, centers=3, cluster_std=0.60, random_state=0)
from __main__ import K_Means
"""

sim1.append(timeit.timeit(stmt=TEST_CODE1,setup=SETUP_CODE,number=50)/50)

print ("7")

SETUP_CODE = """
import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=300000, centers=3, cluster_std=0.60, random_state=0)
from __main__ import K_Means
"""

sim1.append(timeit.timeit(stmt=TEST_CODE1,setup=SETUP_CODE,number=50)/50)


print ("8")

SETUP_CODE = """
import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=400000, centers=3, cluster_std=0.60, random_state=0)
from __main__ import K_Means
"""

sim1.append(timeit.timeit(stmt=TEST_CODE1,setup=SETUP_CODE,number=50)/50)

print ("9")

SETUP_CODE = """
import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=500000, centers=3, cluster_std=0.60, random_state=0)
from __main__ import K_Means
"""

sim1.append(timeit.timeit(stmt=TEST_CODE1,setup=SETUP_CODE,number=50)/50)


print ("10")


import pandas as pd
# Creating a dataframe with the results
results = pd.DataFrame(sim1)
# save data to a csv file
results.to_csv('./sim_serial_k3.csv', sep='\t')














