from operator import le
import os
from os import stat
from matplotlib.colors import LogNorm
import numpy as np
from numpy.core.arrayprint import printoptions
from numpy.core.defchararray import isdigit
from numpy.core.fromnumeric import shape
from numpy.core.numeric import NaN
from numpy.lib.function_base import angle, append
import pandas as pd
from numpy.linalg import norm
import statistics
import math
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from seaborn.palettes import color_palette, cubehelix_palette
import glob as glob
# from matplotlib.pyplot import plot, subplots, figure


#calculates basic feature matricies
#Precondition: CSVs of x positions in the first column and y positions in the second column stored in a "data" folder with separate folders for each driver
def calcBasicFeatures():

    #creates a list of folders of drivers
    temp1 = glob.glob("data/?")
    temp2 = glob.glob("data/??")
    temp3 = glob.glob("data/???")
    temp4 = glob.glob("data/????")

    folders = temp1 + temp2 + temp3 + temp4

    #iterates through each driver's folder
    for folder in folders:
        
        #creates a list of trips for the specified driver
        trip = []
        files = glob.glob(folder + "/*.csv")

        #iterates through each trip for the driver
        for file in files:
            trip = pd.read_csv(file)

            xDiff = []
            yDiff = []

            for j in range(len(trip.x) - 1):
                xDiff.append(trip.x[j + 1] - trip.x[j])

            for j in range(len(trip.y) - 1):
                yDiff.append(trip.y[j + 1] - trip.y[j])

            #calculates the normal of the velocity vector
            velocityVectorList = []
            velocityNormList = []
            for j in range(len(xDiff)):
                velocityVector = [xDiff[j], yDiff[j]]
                velocityVectorList.append(velocityVector)
                velocityNorm = norm(velocityVectorList[j])
                velocityNormList.append(velocityNorm)

            #calculates the difference between velocity norms
            velocityNormDiffList = [0]
            for j in range(len(velocityNormList) - 1):
                velocityNormDiffList.append(
                    velocityNormList[j + 1] - velocityNormList[j])

            #calculates the normal of acceleration
            accelNormList = [0]
            for j in range(len(velocityVectorList) - 1):
                VelVector1 = velocityVectorList[j]
                VelVector2 = velocityVectorList[j + 1]
                accelX = VelVector2[0] - VelVector1[0]
                accelY = VelVector2[1] - VelVector1[1]
                accelVector = [accelX, accelY]

                accelNormList.append(norm(accelVector))

            #calculates the difference of acceleration normals
            accelNormDiffList = [0]
            for j in range(len(accelNormList) - 1):
                accelNormDiffList.append(
                    accelNormList[j + 1] - accelNormList[j])

            #calculates angular speed
            invTan = np.arctan2(trip.y, trip.x)
            angularSpeed = []

            k = 1
            while k < len(invTan):
                angularSpeed.append((invTan[k] - invTan[k - 1]) * 180 / np.pi)

                k += 1

            angularSpeed[0] = 0

            #puts all of the features together
            indexes = [j for j in range(len(velocityNormList))]
            matrix = np.array([indexes, velocityNormList, velocityNormDiffList,
                               accelNormList, accelNormDiffList, angularSpeed])

            basicStatMatrix = np.round(matrix, 4)

            #saves the matrix
            temp5 = file.split("\\")
            driverNum = temp5[1]
            tripNum = temp5[2].split(".")[0]

            if not os.path.exists('data\Basic Matricies\\' + driverNum):
                os.mkdir('data\Basic Matricies\\' + driverNum)

            np.savetxt('data\Basic Matricies\\' + driverNum + '\\trip' + tripNum +
                       ".csv", basicStatMatrix, fmt='%.5f', delimiter=',')

#calculates statistical features for each basic feature
#statstical features include: mean, max, min, 25th percentile, 50th percentile, 75th percentile, and standard deviation
#Precondition: CSVs of basic feature matricies stored in "data/Basic Matricies" with individual folders for each driver
def calcStatFeatureMatrix():

    #creates a list of all drivers
    folders = glob.glob("data/Basic Matricies/*")

    #iterates through each driver's folder
    for folder in folders:

        #creates a list of each driver's basic feature matrix for each trip
        files = glob.glob(folder + "/*.csv")
        driverNum = folder.split("\\")[-1]

        #iterates through each basic feature matrix
        for file in files:

            #grabs the trip number
            temp1 = file.split("\\")
            temp2 = temp1[2].split(".")
            temp3 = temp2[0]
            tripNum = temp3[4:]

            #reads in the matrix
            readDataFrame = pd.read_csv(file)
            basicMatrix = readDataFrame.to_numpy()

            #creates the empty 2d matrix
            statFeatureMatrix = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [
            ], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], ]

            #calculates the window and step size for each trip
            #results in a matrix with a lenght of 128 
            tripLength = len(basicMatrix[0])
            windowSize = tripLength / 64
            stepSize = windowSize / 2

            #calculates statistical features for each window
            windowStartCounter = 0
            while windowStartCounter < tripLength:
                velocityNorms = basicMatrix[0]
                velocityNormDiffs = basicMatrix[1]
                accelNorms = basicMatrix[2]
                accelNormDiffs = basicMatrix[3]
                angularSpeed = basicMatrix[4]

                windowStart = math.floor(windowStartCounter)

                #calculates window end
                if (windowStartCounter + windowSize > tripLength):
                    windowEnd = tripLength
                else:
                    windowEnd = math.floor(windowStartCounter + windowSize)

                window = [velocityNorms[windowStart: windowEnd],
                          velocityNormDiffs[windowStart: windowEnd],
                          accelNorms[windowStart: windowEnd],
                          accelNormDiffs[windowStart: windowEnd],
                          angularSpeed[windowStart: windowEnd],
                          ]

                for j in range(len(window)):
                    row = 7*j
                    statFeatureMatrix[row].append(statistics.mean(window[j]))
                    statFeatureMatrix[row + 1].append(min(window[j]))
                    statFeatureMatrix[row + 2].append(max(window[j]))
                    statFeatureMatrix[row +
                                      3].append(np.quantile(window[j], .25))
                    statFeatureMatrix[row +
                                      4].append(np.quantile(window[j], .50))
                    statFeatureMatrix[row +
                                      5].append(np.quantile(window[j], .75))
                    statFeatureMatrix[row + 6].append(np.std(window[j]))

                windowStartCounter += stepSize

            #transposes the matrix to make the statistical features columns
            statFeatureMatrix = np.transpose(statFeatureMatrix)

            #saves the matrices
            fileHeader = 'Velocity Norm Mean,Velocity Norm Min,Velocity Norm Max,Velocity Norm 25%,Velocity Norm 50%,Veloctiy Norm 75%,Velocity Norm STD,Velocity Norm Difference Mean,Velocity Norm Difference Min,Velocity Norm Difference Max,Velocity Norm Difference 25%,Velocity Norm Difference 50%,Velocity Norm Difference 75%,Velocity Norm Difference STD,Acceleration Norm Mean,Acceleration Norm Min,Acceleration Norm Max,Acceleration Norm 25%,Acceleration Norm 50%,Acceleration Norm 75%,Acceleration Norm STD,Acceleration Norm Difference Mean,Acceleration Norm Difference Min,Acceleration Norm Difference Max,Acceleration Norm Difference 25%, Acceleration Norm Difference 50%,Acceleration Norm Difference 75%,Acceleration Norm Differene STD,Angular Speed Mean,Angular Speed Min,Angular Speed Max,Angular Speed 25%,Angular Speed 50%,Angular Speed 75%,Angular Speed STD'

            filePath = ".\data\Stat Matricies"
            folderName = ""
            if (int(driverNum) < 10):
                folderName = "driver000" + driverNum
            elif (int(driverNum) < 100):
                folderName = "driver00" + driverNum
            elif (int(driverNum) < 1000):
                folderName = "driver0" + driverNum
            else:
                folderName = "driver" + driverNum

            filePath = os.path.join(filePath, folderName)

            if not os.path.exists(filePath):
                os.mkdir(filePath)

            fileName = ""

            if (int(driverNum) < 10):
                if (int(tripNum) < 10):
                    fileName = filePath + "\d000" + driverNum + "t00" + tripNum + ".csv"
                elif (int(tripNum) < 100):
                    fileName = filePath + "\d000" + driverNum + "t0" + tripNum + ".csv"
                else:
                    fileName = filePath + "\d000" + driverNum + "t" + tripNum + ".csv"
            elif (int(driverNum) < 100):
                if (int(tripNum) < 10):
                    fileName = filePath + "\d00" + driverNum + "t00" + tripNum + ".csv"
                elif (int(tripNum) < 100):
                    fileName = filePath + "\d00" + driverNum + "t0" + tripNum + ".csv"
                else:
                    fileName = filePath + "\d00" + driverNum + "t" + tripNum + ".csv"
            elif (int(driverNum) < 1000):
                if (int(tripNum) < 10):
                    fileName = filePath + "\d0" + driverNum + "t00" + tripNum + ".csv"
                elif (int(tripNum) < 100):
                    fileName = filePath + "\d0" + driverNum + "t0" + tripNum + ".csv"
                else:
                    fileName = filePath + "\d0" + driverNum + "t" + tripNum + ".csv"
            else:
                if (int(tripNum) < 10):
                    fileName = filePath + "\d" + driverNum + "t00" + tripNum + ".csv"
                elif (int(tripNum) < 100):
                    fileName = filePath + "\d" + driverNum + "t0" + tripNum + ".csv"
                else:
                    fileName = filePath + "\d" + driverNum + "t" + tripNum + ".csv"

            np.savetxt(fileName, statFeatureMatrix, fmt='%.5f',
                       delimiter=',', header=fileHeader)

#calculates a heatmap with velocity on the x axis and acceleration on the y axis
#Precondition: CSVs of basic feature matricies stored in "data/Basic Matricies" with individual folders for each driver
def vaHeatCalc():

    #creates a list of driver folders
    folders = glob.glob("data\Basic Matricies\\*")

    #divides velocity and acceleration into bins of .1
    velBins = np.linspace(0, 15, 151)
    accelBins = np.linspace(0, 5, 51)

    #creates a header for the CSV
    header = ""
    for vel in velBins:
        header += str(vel) + ", "

    header = header[:-1]


    #iterates through each driver's folder
    for folder in folders:

        #creates a list of trips for each driver
        files = glob.glob(folder + "\*.csv")
        driverNum = folder.split("\\")[-1]

        print(driverNum)

        #creates a list of all instances of velocity and normal of acceleration
        driverVel = []
        driverAccel = []

        for file in files:
            matrix = pd.read_csv(file).to_numpy()
            tripVel = matrix[0]
            tripAccel = matrix[2]

            driverVel = np.append(driverVel, tripVel)
            driverAccel = np.append(driverAccel, tripAccel)

        binned, binx, biny = np.histogram2d(driverVel, driverAccel, bins=[
                                            velBins, accelBins])

        #creates the heatmap
        heatMap = binned.T

        #saves the heatmap
        fileName = "data/VA Heatmaps/driver"
        if (int(driverNum) < 10):
            fileName = fileName + "000" + driverNum
        elif (int(driverNum) < 100):
            fileName = fileName + "00" + driverNum
        elif (int(driverNum) < 1000):
            fileName = fileName + "0" + driverNum
        else:
            fileName = fileName + driverNum

        fileName = fileName + ".csv"

        np.savetxt(fileName, heatMap, fmt='%d', header=header,
                   delimiter=',')

#calculates a heatmap with velocity on the x axis and acceleration on the y axis
#Precondition: CSVs of basic feature matricies stored in "data/Basic Matricies" with individual folders for each driver
def vaHeatCalc2():

    #divides velocity and acceleration into bins of .1
    velBins = np.linspace(0, 40, 401)
    accelBins = np.linspace(-5, 5, 101)

    header = ""
    for vel in velBins:
        header += str(vel) + ","

    header = header[:-1]

    #creates a list of driver folders
    temp1 = glob.glob("data/?")
    temp2 = glob.glob("data/??")
    temp3 = glob.glob("data/???")
    temp4 = glob.glob("data/????")

    folders = temp1 + temp2 + temp3 + temp4

    #iterates through each driver's folder
    for folder in folders:

        driverNum = folder[5:]
        print(driverNum)

        #creates a list of trips for each driver
        trip = []
        files = glob.glob(folder + "/*.csv")

        #creates a list of all instances of velocity and acceleration 
        velocityNormList = []
        velocityNormDiffList = []

        for file in files:
            trip = pd.read_csv(file)

            xDiff = []
            yDiff = []

            for j in range(len(trip.x) - 1):
                xDiff.append(trip.x[j + 1] - trip.x[j])

            for j in range(len(trip.y) - 1):
                yDiff.append(trip.y[j + 1] - trip.y[j])

            velocityVectorList = []
            tempVelNormList = []
            for j in range(len(xDiff)):
                velocityVector = [xDiff[j], yDiff[j]]
                velocityVectorList.append(velocityVector)
                velocityNorm = norm(velocityVectorList[j])
                velocityNormList.append(velocityNorm)
                tempVelNormList.append(velocityNorm)

            velocityNormDiffList.append(0)
            for j in range(len(tempVelNormList) - 1):
                velocityNormDiffList.append(
                    tempVelNormList[j + 1] - tempVelNormList[j])

        driverVel = np.array(velocityNormList)
        driverAccel = np.array(velocityNormDiffList)

        #creates the heatmap
        binned, binx, biny = np.histogram2d(driverVel, driverAccel, bins=[
                                            velBins, accelBins])

        heatMap = binned.T

        #saves the heatmap
        fileName = "data/VA Heatmaps/driver"
        if (int(driverNum) < 10):
            fileName = fileName + "000" + driverNum
        elif (int(driverNum) < 100):
            fileName = fileName + "00" + driverNum
        elif (int(driverNum) < 1000):
            fileName = fileName + "0" + driverNum
        else:
            fileName = fileName + driverNum

        fileName = fileName + ".csv"

        np.savetxt(fileName, heatMap, fmt='%d', header=header,
                   delimiter=',')

#creates some labels for each driver's VA heatmap
#Precondition: CSVs of x positions in the first column and y positions in the second column stored in a "data" folder with separate folders for each driver
def vaLabelCalc():

    files = glob.glob("data/VA Heatmaps/driver????.csv")

    # print(files)
    drivers = []
    risks = []
    norms = []

    for file in files:

        riskOccurence = 0
        driverNum = file[23:-4]
        print(driverNum)

        #reads in the CSV
        trip = pd.read_csv(file).to_numpy()
        trip = trip[np.logical_not(np.isnan(trip))]
        trip = np.reshape(trip, [100, 400])

        #calculates the l2 norm
        l2Norm = np.linalg.norm(trip)

        # print(trip)

        #calculates a risk rating
        for j in range(100):
            if ((j < 6) or (j > 95)) and (j != 0):
                # print(trip[j])
                for num in trip[j]:
                    if not (math.isnan(num)):
                        riskOccurence += int(num)

        risks.append(riskOccurence)
        drivers.append(driverNum)
        norms.append(l2Norm)

    driverMap = [[], [], []]
    driverMap[0] = drivers
    driverMap[1] = risks
    driverMap[2] = norms

    driverMap = np.transpose(driverMap)

    #saves the labels
    np.savetxt("data/VA Heatmaps/VA Labels.csv", driverMap, fmt=['%s', '%s', '%s'], header="Driver,Risk,L2 Norm",
               delimiter=',')


# main

calcBasicFeatures()
calcStatFeatureMatrix()
vaHeatCalc2()
vaLabelCalc()
