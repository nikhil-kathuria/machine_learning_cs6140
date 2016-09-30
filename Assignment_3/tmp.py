from readData import readSpam, extractMatrix

from normalizedata import normalize

from GDA import computeU, getPrior

import numpy as np

import math



def extractDistribution(traindata):
    holder = list()
    for idx in range(traindata.shape[1]):
        col = traindata[:, idx]
        hmap = dict()
        for data in col:
            if data in hmap:
                hmap[data] += 1
            else:
                hmap[data] = 1

        # Add the completed mapping of feature idx
        holder.append(hmap)
    return holder


def partionMatrix(traindata):
    zero = list()
    one = list()
    for idx in range(traindata.shape[0]):
        if trainlabels[idx] == 0:
            zero.append(idx)
        else:
            one.append(idx)    

    zeromat = meanMat(traindata[zero, :])
    onemat = meanMat(traindata[one, :])
    return (zeromat, onemat)


def getPredictions(data, fldy0, fldy1, labels):
    (py0, py1) = getPrior(labels)
    flen = 
    for idx in range(data.shape[0]):
        row = data[idx, :]
        prowy0 = 1
        prowy1 = 1
        for data in row:
            if data in fldy0[idx]:
                pdataY = (fldy0[idx][data] + 1) / float(len(fldy0[idx]))
                


def runNB(name):
    (bucketmap, full) = readSpam(name)

    sumacc = 0
    for key in bucketmap.keys():
        trainset = list(bucketmap.keys())
        testset = list()
        testset.append(key)
        trainset.remove(key)

        # Get train and test matrix and labels as well
        (traindata, trainlabels) = extractMatrix(bucketmap, trainset, full)
        (testdata, testlabels) = extractMatrix(bucketmap, testset, full)

        # Normalize test and traint data
        # normalize(traindata)
        # normalize(testdata)
        # (U0, U1) = computeU(traindata, trainlabels)

        (zeromat, onemat) = partionMatrix(traindata)

        fdistrY0 = extractDistribution(zeromat)
        fdistrY1 = extractDistribution(onemat)

        predictions = getPredictions(testdata, fdistrY0, fdistrY1,labels)


def main():
    runNB("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_1/spambase.data.txt")


if __name__ == '__main__':
    main()