from matplotlib.colors import LogNorm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter

# # creates heatmap visualizations
# fig, axs = plt.subplots()

# file = "data\VA Heatmaps\driver1410.csv"
# matrix = pd.read_csv(file)
# # print(matrix)

# velBins = np.linspace(0, 40, 401)
# accelBins = np.linspace(5, -5, 101)

# # accelBins = np.delete(accelBins, 180)
# # accelBins = np.delete(accelBins, 179)

# for i in range(len(velBins)):
#     velBins[i] = round(velBins[i], 1)

# for i in range(len(accelBins)):
#     accelBins[i] = round(accelBins[i], 1)

# axs = sns.heatmap(matrix, norm=LogNorm(),
#                   xticklabels=velBins, yticklabels=accelBins, vmax=1000)

# xLabels = axs.get_xticklabels()
# yLabels = axs.get_yticklabels()

# for i, l in enumerate(xLabels):
#     if (i % 10 != 0):
#         xLabels[i] = ''

# for i, l in enumerate(yLabels):
#     if (i % 10 != 0):
#         yLabels[i] = ''

# axs.set_xticklabels(xLabels, rotation=90)
# axs.set_yticklabels(yLabels)

# # matrix = pd.read_csv("data\VA Heatmaps\driver0001.csv")
# # axs = sns.heatmap(matrix, cmap="OrRd", norm=LogNorm(), vmax=5000)

# plt.show()

matrix = pd.read_csv(
    'data\VA Minus 00\orig norm\driver18.csv', header=None).to_numpy()

matrix = matrix[:, 0:398]

plt.matshow(matrix, cmap='Greys')
plt.show()
