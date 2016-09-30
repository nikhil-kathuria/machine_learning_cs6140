import numpy as np
from normalizedata import normalize

from readData import readSpam, extractMatrix, readHaar
from scipy.spatial.distance import euclidean, cosine
import math


def gaussian(testv, trainv):
    sigma = 1
    num = euclidean(testv, trainv)
    val = -num / (sigma * sigma)
    return math.exp(val)


def computeKernelSpam(row, zeromat, onemat):
    smap = dict()
    sumzero = 0
    sumone = 0

    zerolen = float(len(zeromat))
    onelen = float(len(onemat))
    totallen = zerolen + onelen

    # For one matrix
    for itr in range(onemat.shape[0]):
        sumone += gaussian(row, onemat[itr])
    sumone = sumone / onelen

    # For zero matrix
    for itr in range(zeromat.shape[0]):
        sumzero += gaussian(row, zeromat[itr])
    sumzero = sumzero / zerolen

    pzero = (zerolen / totallen) * sumzero
    pone = (onelen / totallen) * sumone

    if (pone > pzero):
        return 1
    else:
        return 0


def accCalc(predictions, labels):
    hits = 0
    for idx in range(len(predictions)):
        if predictions[idx] == labels[idx]:
            hits += 1
    acc = hits / float(len(labels))
    print("Accuracy -> " + str(acc * 100))


def partionMatrix(traindata, trainlabels):
    zero = list()
    one = list()
    for idx in range(traindata.shape[0]):
        if trainlabels[idx] == 0:
            zero.append(idx)
        else:
            one.append(idx)

    zeromat = traindata[zero, :]
    onemat = traindata[one, :]
    return (zeromat, onemat)


def performKNNSpam():
    (bucketmap, full) = readSpam("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_1/spambase.data.txt")

    for key in bucketmap.keys():
        trainset = list(bucketmap.keys())
        testset = list()
        testset.append(key)
        trainset.remove(key)

        # Get train and test matrix and labels as well
        (train, trainlabels) = extractMatrix(bucketmap, trainset, full)
        (test, testlabels) = extractMatrix(bucketmap, testset, full)

        # Normalize data
        # normalize(test)
        # normalize(train)
        # print("Done Normalization")

        (zeromat, onemat) = partionMatrix(train, trainlabels)

        labels = [None] * test.shape[0] 
        for itr in range(test.shape[0]):
        	labels[itr] = computeKernelSpam(test[itr], zeromat, onemat)
        	print(labels[itr])

        # Compute accuracy
        accCalc(labels, testlabels)
        exit()



if __name__ == '__main__':
    performKNNSpam()


    # Cosine -> False
    # Euclidina -> False
    # Gaussian - > True
    # Polynomial - > True
