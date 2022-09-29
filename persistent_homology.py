import pandas as pd
import numpy as np
import glob
from PIL import Image
import scipy
import sklearn
import persim
import matplotlib.pyplot as plt
import ripser

from typing import List

def lower_star_filtration(data, new_folder_name):
    diagram: List = ripser.lower_star_img(data)
    persim.plot_diagrams(diagram)
    plt.title(file[-14:-4])
    # plt.show()
    plt.savefig(file.replace('VA Heatmap Percent', new_folder_name).replace('.csv', '.png'))
    plt.clf()

    return diagram

def barcodes(persistence_array, new_folder_name, sort_by = 'birth'):
    if sort_by == 'birth':
        persistence_array.sort(key=lambda persistence: persistence[0])
    elif sort_by == 'death':
        persistence_array.sort(key=lambda persistence: persistence[1])
    elif sort_by == 'life':
        persistence_array.sort(key=lambda persistence: persistence[1] - persistence[0])
    prefix = sort_by + ' barcode driver'

    indicies = []
    births = []
    deaths = []
    for i in range(len(persistence_array)):
        indicies.append(i)
        births.append(persistence_array[i][0])
        deaths.append(persistence_array[i][1])

    plt.hlines(indicies, births, deaths, linewidths=.9)
    plt.savefig(file.replace('VA Heatmap Percent', new_folder_name).replace('driver', prefix).replace('.csv', '.png'),
        dpi=300)
    # plt.show()
    plt.clf()

if __name__ == '__main__':
    new_folder = 'VA Heatmap Persistent Homology'
    files = glob.glob('D:/Users/Kevin/Desktop/Apps/Code/Python/Driver Data/derived_data/VA Heatmap Percent/driver????.csv')
    files = files[0:10]

    for file in files:
        print(file[-8:-4])

        heatmap = pd.read_csv(file, header=None).to_numpy()
        persisences = lower_star_filtration(heatmap, new_folder)

        persistence_array = list(persisences)
        barcodes(persistence_array, new_folder, 'life')
