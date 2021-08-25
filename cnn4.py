import numpy as np
from numpy.core.defchararray import translate
from numpy.lib.function_base import append
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
from tensorflow.python.keras.layers.core import Activation
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from tensorflow.python.ops.gen_array_ops import empty

from tensorflow.python.keras import backend as K

# data generator


def dataGenerator(fileList, batchSize=10):

    i = 0
    while True:
        if i*batchSize >= len(fileList):
            i = 0
            np.random.shuffle(fileList)
        else:
            fileChunk = fileList[i * batchSize: (i + 1) * batchSize]
            data = []
            labels = []
            for file in fileChunk:
                temp = pd.read_csv(open(file, 'r'))

                data.append(temp.values.reshape(128, 35, 1))

                accel = temp['Acceleration Norm Max']
                decel = temp['Acceleration Norm Min']
                turn1 = temp['Angular Speed Max']
                turn2 = temp['Angular Speed Min']
                riskOccurence = 0

                # counts number of instances of acceleration or deceleration above 4.5 m/s^2 or below -4.5 m/s^2
                for j in range(len(accel)):
                    if accel[j] >= 4.5:
                        riskOccurence += 1

                    if decel[j] <= -4.5:
                        riskOccurence += 1

                emptyLabel = [0 for n in range(50)]
                if riskOccurence < 50:
                    emptyLabel[riskOccurence] = 1
                else:
                    emptyLabel[49] = 1

                labels.append(emptyLabel)

                # labels.append(temp2)

            data = np.asarray(data).reshape(-1, 128, 35, 1)
            labels = np.asarray(labels)
            yield data, labels

            i += 1


# main

driverFiles = glob.glob("data/Stat Matricies/**/*t[0][0-2][0-9].csv")

# creates the data sets
train, test = train_test_split(driverFiles, test_size=1000, random_state=0)

trainDataSet = tf.data.Dataset.from_generator(dataGenerator, args=[train, 100], output_shapes=(
    (None, 128, 35, 1), (100, 50)), output_types=(tf.float64, tf.float64))

testDataSet = tf.data.Dataset.from_generator(dataGenerator, args=[test, 100], output_shapes=(
    (None, 128, 35, 1), (100, 50)), output_types=(tf.float64, tf.float64))

poolSize = (3, 1)

# creates the model
model = Sequential([
    Conv2D(32, [5, 35], input_shape=(128, 35, 1)),
    tf.keras.layers.MaxPool2D(poolSize),
    tf.keras.layers.Conv2D(64, [3, 1]),
    tf.keras.layers.MaxPool2D(poolSize),
    tf.keras.layers.Conv2D(64, [3, 1]),
    tf.keras.layers.MaxPool2D(poolSize),
    Dense(128, activation='relu'),
    Flatten(),
    Dense(50, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='SquaredHinge',
    metrics=['Accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5)],
)

epochs = 9
stepsPerEpoch = 200
steps = 100

# model = keras.models.load_model('cnn4.h5')

print(model.fit(trainDataSet, steps_per_epoch=stepsPerEpoch, epochs=epochs))

model.save('cnn4.h5')

testLoss, testAcc, topKAcc = model.evaluate(testDataSet, steps=steps)
print("test loss: ", testLoss, "\naccuracy: ",
      testAcc, "\ntop-5 accuracy: ", topKAcc)
