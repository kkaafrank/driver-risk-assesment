import os
from matplotlib.colors import LogNorm
import numpy as np
from numpy.core.fromnumeric import shape
from numpy.linalg.linalg import norm
from numpy.ma.core import getdata
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from seaborn.miscplot import palplot
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, explained_variance_score, confusion_matrix, accuracy_score, classification_report, log_loss
from math import sqrt
import glob

from sklearn.cluster import KMeans, k_means


def getData():
    files = glob.glob('data/VA Heatmaps/driver????.csv')

    data = []

    for file in files:
        driverHeatmap = pd.read_csv(file).to_numpy()
        driverHeatmap = driverHeatmap[np.logical_not(np.isnan(driverHeatmap))]
        data.append(driverHeatmap)

    # print(np.shape(data))

    return data

# main


xData = getData()

# #used to determine the number of clusters
# numClusters = [x for x in range(2, 11)]

# inertias = []
# for clusters in numClusters:
#     kmeans = KMeans(n_clusters=clusters, random_state=10)
#     kmeans = kmeans.fit(xData)
#     inertias.append(kmeans.inertia_)

# for i in range(len(inertias)):
#     print('Inertia for ', i + 2, 'is', inertias[i])

# fig, (ax1) = plt.subplots(1, figsize=(16, 6))
# xx = np.arange(len(numClusters))
# ax1.plot(xx, inertias)
# ax1.set_xticks(xx)
# ax1.set_xticklabels(numClusters)

# plt.show()

# categorizes the VA heatmaps into 4 clusters
kmeans = KMeans(n_clusters=4, random_state=10)
kmeans = kmeans.fit(xData)

# print('Clusters:  ', kmeans.labels_)
# print('Inertia:   ', kmeans.inertia_)

center1 = kmeans.cluster_centers_[0]
center2 = kmeans.cluster_centers_[1]
center3 = kmeans.cluster_centers_[2]
center4 = kmeans.cluster_centers_[3]

center1 = np.reshape(center1, [100, 400])
center2 = np.reshape(center2, [100, 400])
center3 = np.reshape(center3, [100, 400])
center4 = np.reshape(center4, [100, 400])


velBins = np.linspace(0, 40, 401)
header = ""

for vel in velBins:
    header += str(vel) + ","

header = header[:-1]

# saves cluster centers
np.savetxt("Cluster Heatmaps/cluster 1.csv", center1,
           fmt='%d', delimiter=',', header=header)

np.savetxt("Cluster Heatmaps/cluster 2.csv", center2,
           fmt='%d', delimiter=',', header=header)

np.savetxt("Cluster Heatmaps/cluster 3.csv", center3,
           fmt='%d', delimiter=',', header=header)

np.savetxt("Cluster Heatmaps/cluster 4.csv", center4,
           fmt='%d', delimiter=',', header=header)

np.savetxt("Cluster Heatmaps/Driver Labels.csv",
           kmeans.labels_, fmt='%d', delimiter=",")

results = kmeans.labels_
print(results)

results = np.transpose(results)
np.savetxt("data\VA Heatmaps\VA Clusters.csv", results, fmt='%d')
