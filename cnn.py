import numpy as np
from numpy.core.defchararray import translate
import pandas as pd
import glob
import re
from shutil import move
from pandas.core.frame import DataFrame
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Dense, Flatten, InputLayer
from tensorflow.keras.utils import to_categorical
from tensorflow.python import keras
from tensorflow.python.keras.backend import conv2d
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split

np.random.seed(1111)

# generator for model


def dataGenerator(fileList, drivers, batchSize=10):
    i = 0
    while True:
        if i*batchSize >= len(fileList):
            i = 0
            np.random.shuffle(fileList)
        else:
            fileChunk = fileList[i * batchSize: (i + 1) * batchSize]
            data = []
            labels = []
            labelClasses = tf.constant(drivers)
            for file in fileChunk:
                temp = pd.read_csv(open(file, 'r'))

                data.append(temp.values.reshape(128, 35, 1))

                pattern = tf.constant(eval("file[33: 38]"))

                temp2 = []
                for j in range(len(labelClasses)):
                    temp2.append(0)

                for j in range(len(labelClasses)):
                    if (re.match(pattern.numpy(), labelClasses[j].numpy())):
                        temp2[j] = 1

                labels.append(temp2)

            data = np.asarray(data).reshape(-1, 128, 35, 1)
            labels = np.asarray(labels)
            yield data, labels

            i += 1


# main

# creates a list of drivers
driverFolders = glob.glob("data/Stat Matricies/driver[0][0-1][0-9][0-9]")
drivers = []
for file in driverFolders:
    temp1 = file.split("\\")[1]
    temp2 = temp1[6:]
    if temp1 not in drivers:
        drivers.append("d" + temp1)

driverFiles = glob.glob(
    "data/Stat Matricies/driver[0][0-1][0-9][0-9]/*.csv")

# creates the data sets
train, test = train_test_split(driverFiles, test_size=200, random_state=0)


trainDataSet = tf.data.Dataset.from_generator(dataGenerator, args=[train, drivers, 100], output_shapes=(
    (None, 128, 35, 1), (None, 124)), output_types=(tf.float64, tf.float64))

testDataSet = tf.data.Dataset.from_generator(dataGenerator, args=[test, drivers, 100], output_shapes=(
    (None, 128, 35, 1), (None, 124)), output_types=(tf.float64, tf.float64))

poolSize = (2, 1)

# creats teh model
model = Sequential([
    Conv2D(32, [5, 35], input_shape=(128, 35, 1)),
    tf.keras.layers.MaxPool2D(poolSize),
    tf.keras.layers.Conv2D(64, [3, 1]),
    tf.keras.layers.MaxPool2D(poolSize),
    tf.keras.layers.Conv2D(64, [3, 1]),
    tf.keras.layers.MaxPool2D(poolSize),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Flatten(),
    Dense(124, activation='softmax')
])

# print(model.summary())

model.compile(
    optimizer='adam',
    loss='SquaredHinge',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5)]
)

stepsPerEpoch = 100
steps = 10

print(model.fit(
    trainDataSet,
    steps_per_epoch=stepsPerEpoch,
    epochs=10
))

model.save_weights('cnn Weights.h5')

#model.load_weights("cnn Weights.h5")

testLoss, testAcc, testTop5Acc = model.evaluate(testDataSet, steps=steps)
print("test loss: ", testLoss, "\ntest accuracy: ",
      testAcc, "\ntest top-5 accuracy: ", testTop5Acc)
