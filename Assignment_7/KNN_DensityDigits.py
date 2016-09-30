import numpy as np
from normalizedata import normalize

from readData import readSpam, extractMatrix, readHaar
from scipy.spatial.distance import euclidean, cosine
import math
from collections import defaultdict


def gaussian(testv, trainv):
    sigma = 1 # Try .25 as well
    num = euclidean(testv, trainv)
    val = -num / (sigma * sigma)
    return math.exp(val)


def ploynomial(testv, trainv):
    val = 0
    gamma = .5 # gamma = 1 / C
    coefficient = .25
    degree = 2
    val = np.dot(testv, trainv)
    return math.pow(val * gamma + coefficient, degree)


def predictDigits(smap):
    val = -1
    mykey = 1
    for key in smap:
        if smap[key] > val:
            val = smap[key]
            mykey = key
            # print(str(val) + " " +  str(mykey))
    return mykey        


def computeKernelDigits(row, mydict, total):
    smap = dict()
    for key in mydict:
        data = mydict[key]
        mysum = 0

        # Iterate over submatrix of Label C
        for itr in range(data.shape[0]):
            # mysum += ploynomial(row, data[itr])
            mysum += gaussian(row, data[itr])
        # Compute prior and normalize sum and add to dict
        length = data.shape[0]
        prior = length / total
        mysum = mysum / length
        smap[key] = prior * mysum

    return predictDigits(smap)


def accCalc(predictions, labels):
    hits = 0
    for idx in range(len(predictions)):
        if predictions[idx] == labels[idx]:
            hits += 1
    acc = hits / float(len(labels))
    print("Accuracy -> " + str(acc * 100))


def partionMatrix(traindata, trainlabels):
    mydict = defaultdict(list)
    for idx in range(traindata.shape[0]):
        mydict[trainlabels[idx]].append(idx)

    finaldict = dict()
    for key in mydict:
        finaldict[key] = traindata[mydict[key], :]

    return finaldict


def performKNNDigits():
    test = readHaar("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/HF/htest.txt")
    testlabels = readHaar("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/HF/htestlabels.txt")
    train = readHaar("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/HF/s20train.txt")
    trainlabels = readHaar("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/HF/s20labels.txt")

    # Normalize data
    # normalize(test)
    # normalize(train)

    length = float(len(train))
    mydict = partionMatrix(train, trainlabels)

    labels = [None] * test.shape[0] 
    for itr in range(test.shape[0]):
        labels[itr] = computeKernelDigits(test[itr], mydict, length)
        print(labels[itr])


    accCalc(labels, testlabels)
    exit()


if __name__ == '__main__':
    performKNNDigits()

    # Cosine -> False
    # Euclidina -> False
    # Do not normalize for ploynomial function, Gamma = .5, coefficient = .25 -> 60 % accuracy
    # Do not normalize for Gaussian function as well