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
            labelClasses = tf.constant([1, 2, 3, 4, 5])
            for file in fileChunk:
                temp = pd.read_csv(open(file, 'r'))

                data.append(temp.values.reshape(128, 35, 1))

                accel = temp['Acceleration Norm Max']
                decel = temp['Acceleration Norm Min']
                turn1 = temp['Angular Speed Max']
                turn2 = temp['Angular Speed Min']
                riskOccurence = 0

                # calculates a risk rating based on angular velocity
                previousIncrement = False
                for j in range(len(accel)):
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

                if riskOccurence <= 2:
                    labels.append([1, 0, 0, 0, 0])
                elif riskOccurence <= 5:
                    labels.append([0, 1, 0, 0, 0])
                elif riskOccurence <= 5:
                    labels.append([0, 0, 1, 0, 0])
                elif riskOccurence <= 8:
                    labels.append([0, 0, 0, 1, 0])
                else:
                    labels.append([0, 0, 0, 0, 1])

                # labels.append(temp2)

            data = np.asarray(data).reshape(-1, 128, 35, 1)
            labels = np.asarray(labels)
            yield data, labels

            i += 1


# main
driverFiles = glob.glob("data/Stat Matricies/**/*t[0][0-2][0-9].csv")

# creates data sets
train, test = train_test_split(driverFiles, test_size=1000, random_state=0)

trainDataSet = tf.data.Dataset.from_generator(dataGenerator, args=[train, 100], output_shapes=(
    (None, 128, 35, 1), (None, 5)), output_types=(tf.float64, tf.float64))

testDataSet = tf.data.Dataset.from_generator(dataGenerator, args=[test, 100], output_shapes=(
    (None, 128, 35, 1), (None, 5)), output_types=(tf.float64, tf.float64))

poolSize = (3, 1)

# creates the model
model = Sequential([
    Conv2D(32, [5, 35], input_shape=(128, 35, 1)),
    tf.keras.layers.MaxPool2D(poolSize),
    tf.keras.layers.Conv2D(64, [3, 1]),
    tf.keras.layers.MaxPool2D(poolSize),
    tf.keras.layers.Conv2D(64, [3, 1]),
    tf.keras.layers.MaxPool2D(poolSize),
    Dense(128),
    Flatten(),
    Dense(5, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='CategoricalHinge',
    metrics=['Accuracy', 'Recall']
)

epochs = 10
stepsPerEpoch = 100
steps = 100

model = keras.models.load_model('cnn5.h5')

print(model.fit(trainDataSet, steps_per_epoch=stepsPerEpoch, epochs=epochs))

model.save('cnn5.h5')

testLoss, testAcc, testRecall = model.evaluate(testDataSet, steps=steps)
print("test loss: ", testLoss, "\ntest accuracy: ",
      testAcc, "\ntest recall: ", testRecall)
