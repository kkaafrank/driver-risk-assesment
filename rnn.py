import numpy as np
from numpy.core.defchararray import translate
from numpy.lib.function_base import append
import pandas as pd
import glob
import random
import re
from shutil import move
from pandas.core.frame import DataFrame
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, InputLayer, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.python import keras
from tensorflow.python.keras.backend import conv2d
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split

np.random.seed(1111)
random.seed(10)

# data generator for the model


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

                data.append(temp.values.reshape(128, 35))

                accel = temp['Acceleration Norm Max']
                decel = temp['Acceleration Norm Min']
                turn1 = temp['Angular Speed Max']
                turn2 = temp['Angular Speed Min']
                riskOccurence = 0

                # counts number of instances of high acceleration/deceleration and angular velocity
                previousIncrement = False
                for j in range(len(accel)):
                    if accel[j] >= 3.6:
                        if not previousIncrement:
                            riskOccurence += 1
                            previousIncrement = False
                        else:
                            riskOccurence += 2
                            previousIncrement = True
                    if decel[j] <= -3.6:
                        if not previousIncrement:
                            riskOccurence += 1
                            previousIncrement = False
                        else:
                            riskOccurence += 2
                            previousIncrement = True

                    if turn1[j] >= 120:
                        if not previousIncrement:
                            riskOccurence += 1
                            previousIncrement = False
                        else:
                            riskOccurence += 2
                            previousIncrement = True
                    if turn2[j] <= -120:
                        if not previousIncrement:
                            riskOccurence += 1
                            previousIncrement = False
                        else:
                            riskOccurence += 2
                            previousIncrement = True

                # simulates presence of a crash
                if riskOccurence <= 6:
                    if (random.random() <= .005):
                        labels.append([0, 1])
                    else:
                        labels.append([1, 0])
                elif riskOccurence <= 15:
                    if (random.random() <= .01):
                        labels.append([0, 1])
                    else:
                        labels.append([1, 0])
                elif riskOccurence <= 21:
                    if (random.random() <= .02):
                        labels.append([0, 1])
                    else:
                        labels.append([1, 0])
                elif riskOccurence <= 26:
                    if (random.random() <= .05):
                        labels.append([0, 1])
                    else:
                        labels.append([1, 0])
                else:
                    if (random.random() <= .08):
                        labels.append([0, 1])
                    else:
                        labels.append([1, 0])

                # labels.append(temp2)

            data = np.asarray(data).reshape(-1, 128, 35)
            labels = np.asarray(labels)
            yield data, labels

            i += 1


# main
driverFiles = glob.glob("data/Stat Matricies/**/*t[0][0-2][0-9].csv")

# creates the data sets
train, test = train_test_split(driverFiles, test_size=1000, random_state=0)

trainDataSet = tf.data.Dataset.from_generator(dataGenerator, args=[train, 100], output_shapes=(
    (None, 128, 35), (None, 2)), output_types=(tf.float64, tf.float64))

testDataSet = tf.data.Dataset.from_generator(dataGenerator, args=[test, 100], output_shapes=(
    (None, 128, 35), (None, 2)), output_types=(tf.float64, tf.float64))

# creates the model
model = Sequential([
    InputLayer(input_shape=(128, 35)),
    LSTM(128),
    Dense(128),
    Flatten(),
    Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='CategoricalHinge',
    metrics=['Accuracy', 'Recall']
)

epochs = 10
stepsPerEpoch = 200
steps = 100

# model = keras.models.load_model('cnn2')

print(model.fit(trainDataSet, steps_per_epoch=stepsPerEpoch, epochs=epochs))

model.save('rnn.h5')

testLoss, testAcc, testRecall = model.evaluate(testDataSet, steps=steps)
print("test loss: ", testLoss, "\ntest accuracy: ",
      testAcc, "\ntest recall: ", testRecall)
