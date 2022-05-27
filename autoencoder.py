from random import random, randrange
from unicodedata import decimal
import keras
from keras import layers
from keras.backend import shape
from keras.preprocessing.image import img_to_array
from matplotlib import image
import numpy as np
import glob
import cv2
from numpy.core.defchararray import decode
from numpy.lib.npyio import genfromtxt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf

"""Given a folder of images, returns an equivalent 3d array of images

Assumes all images are grayscale so red, blue, and green values for each pixel are the same

folderPath: the path to the folder of images (relative or absolute)
returns: a 3d array of floats that correspond to red rgb values
"""


def getImgs(folderPath):
    files = glob.glob(folderPath)

    images = []
    for i in range(len(files)):
        image = cv2.imread(files[i], 0)
        imgArr = img_to_array(image)
        imgArr = (255 - imgArr.astype('float32')) / 255
        images.append(imgArr)

    images = np.squeeze(images)

    return images


"""Given a text file, returns a 2d array of values in the text file

Assumes the values in the text files are comma separated
    and each row is a set of principal components for a unique driver

filePath: the path file containing pca values (relative or absolute)
returns: a 2d array of floats, each row is a set of pca values for a driver
"""


def getPca(filePath):
    pcas = genfromtxt(filePath, delimiter=',')
    pcas = pcas.astype(np.int)

    return pcas


"""Shows original, pca, and predicted graphs

test_imgs: 3d array of floats that represents greyscale rgb values of images
test_pca: 2d array of principal components (floats)
decoded_imgs: 3d array of floats, predicted images of the autoencoder model
"""


def show_rand_pred(test_imgs, test_pca, decoded_imgs):
    # edit to change the number of predictions to show
    numFigs = 5

    dimensions = (100, 400)
    fig, axs = plt.subplots(3, numFigs, figsize=dimensions)
    plt.gray()

    # used for consistent testing, change to see different predictions
    random.seed(1)

    for i in range(numFigs):
        k = randrange(len(test_imgs))

        axs[0, i].imshow(test_imgs[k].reshape(dimensions))
        axs[0, i].title.set_text(str(k + 1) + " original")

        axs[1, i].imshow(test_pca[k].reshape(1, 5))
        axs[1, i].title.set_text("pca" + str(k + 1))
        for j in range(len(test_pca[k])):
            axs[1, i].text(j, 0, test_pca[i, j], color="r",
                           ha="center", va="center")

        axs[2, i].imshow(decoded_imgs[k].reshape(dimensions))
        axs[2, i].title.set_text(str(k + 1) + " decoded")

    plt.show()


"""Shows pca and predicted graphs, sorted by specified principal component

Principal component indexing starts at 0, not 1
The function does not do bounds checking for the PC index,
    May raise an index out of bounds error if sort_pca is not in range

test_pca: 2d array of principal components (floats)
decoded_imgs: 3d array of floats, predicted images of the autoencoder model
sort_pca: integer, the index of the PC to sort by
"""


def show_sorted_pred(test_imgs, test_pca, decoded_imgs, sort_pca):
    decImgIndex = [x for x in (range(len(decoded_imgs)))]
    pcaIndex = [x for x in range(len(test_pca))]
    sortPca1, sortDecImg1, sortPcaIndex = zip(
        *sorted(zip(test_pca[:, sort_pca], decImgIndex, pcaIndex)))

    numFigs = 5
    fig, axs = plt.subplots(4, 5, figsize=(10, 40))
    plt.gray()
    for i in range(numFigs):
        axs[0, i].imshow(decoded_imgs[sortDecImg1[i]].reshape(100, 400))
        axs[0, i].title.set_text(str(sortDecImg1[i]) + " decoded")

        axs[1, i].imshow(test_pca[sortPcaIndex[i]].reshape(1, 5))
        axs[1, i].title.set_text("pca" + str(i + 1))
        for j in range(len(test_pca[sortPcaIndex[i]])):
            axs[1, i].text(j, 0, test_pca[sortPcaIndex[i], j], color="r",
                           ha="center", va="center")

        axs[2, i].imshow(decoded_imgs[sortDecImg1[
            len(decoded_imgs) - i - 1]].reshape(100, 400))
        axs[2, i].title.set_text(
            str(sortDecImg1[len(decoded_imgs) - i - 1]) + " decoded")

        axs[3, i].imshow(
            test_pca[sortPcaIndex[len(sortPcaIndex) - i - 1]].reshape(1, 5))
        axs[3, i].title.set_text("pca" + str(i + 1))
        for j in range(len(test_pca[i])):
            axs[3, i].text(j, 0, test_pca[sortPcaIndex[len(sortPcaIndex) - i - 1], j], color="r",
                           ha="center", va="center")

    fig.suptitle('Sorted by the PC' + str(sort_pca + 1))

    plt.show()


if (__name__ == '__main__'):

    # change this depending on the number of principal components used
    encodingDim = 5

    # change this depending on the dimensions of the data
    numPoints = 32000

    inputImg = keras.Input(shape=(numPoints))
    encoded = layers.Dense(encodingDim, activation="relu")(inputImg)
    decoded = layers.Dense(numPoints, activation="sigmoid")(encoded)

    autoencoder = keras.Model(inputImg, decoded)

    encoder = keras.Model(inputImg, encoded)

    encodedInput = keras.Input(shape=(encodingDim,))
    decoderLayer = autoencoder.layers[-1]
    decoder = keras.Model(encodedInput, decoderLayer(encodedInput))

    autoencoder.compile(optimizer="adam", loss="mean_squared_error")
    decoder.compile(optimizer="adam", loss=tf.losses.BinaryFocalCrossentropy())

    # change folder and file names accordingly to match where your data is located
    folder = 'VelAng Subset'

    images = getImgs("data\\" + folder + "\images\driver*.png")
    pca = getPca("data\\" + folder + "\\" + folder + " PCA.csv")
    images = images.reshape((len(images), np.prod(images.shape[1:])))

    train_imgs, test_imgs, train_pca, test_pca = train_test_split(
        images, pca, test_size=.1, shuffle=False, stratify=None)

    # uncomment if the autoencoder has been run previously
    # change the weight file name accordingly
    # autoencoder.load_weights('Model Weights\\autoencoder2.h5')

    # uncomment to use auto encoder without pca
    # autoencoder.fit(trainImgs, trainImgs,
    #                 epochs=500,
    #                 batch_size=100,
    #                 shuffle=True,
    #                 validation_data=(testImgs, testImgs))

    # uncomment if the decoder has been run previously
    # change the weight file name accordingly
    # decoder.load_weights("Model Weights\VelAng subset focal cross entropy.h5")

    decoder.fit(train_pca, train_imgs,
                epochs=50,
                batch_size=100,
                shuffle=True,
                validation_data=(test_pca, test_imgs))

    # uncomment to use autoencoder without pca
    # encodedImgs = encoder.predict(testImgs)

    decodedImgs = decoder.predict(pca)

    # uncomment if using the autoencoder without pca
    # autoencoder.save("Model Weights\\" + folder + "autoencoder2.h5")

    decoder.save("Model Weights\\" + folder +
                 " decoder focal cross entropy.h5")

    numFigs = 5
    fig, axs = plt.subplots(2, 5)
    plt.gray()
    for i in range(numFigs):
        axs[0, i].imshow(images[i].reshape(80, 400))
        axs[0, i].title.set_text(str(i + 1) + " original")

        axs[1, i].imshow(decodedImgs[i].reshape(80, 400))
        axs[1, i].title.set_text(str(i + 1) + " decoded")

    plt.show()

    decodedImgs = decodedImgs.reshape(1000, 80, 400)
    for i in range(len(decodedImgs)):
        print(i)
        np.savetxt("data\VelAng subset\pred\\driver" + str(i) + ".csv",
                   decodedImgs[i], delimiter=',', fmt='%5f')

    show_rand_pred(test_imgs, test_pca, decodedImgs)
    show_sorted_pred(test_imgs, test_pca, decodedImgs, 0)
