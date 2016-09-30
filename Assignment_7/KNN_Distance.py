import numpy as np
from normalizedata import normalize

from readData import readSpam, extractMatrix, readHaar
from scipy.spatial.distance import euclidean, cosine
import math


def euclidianDistance(testv, trainv):
	return euclidean(testv, trainv)
	dis = float(0)
	for itr in range(len(testv)):
		dis += math.pow(testv[itr] - trainv[itr], 2)
	dis = math.pow(dis, .5)
	return dis


def preSpam(slist, num):
    zeros = 0
    ones = 0

    for itr in range(num):
        if (slist[itr][1] == 0):
           zeros += 1
        else:
           ones += 1
    if (zeros >= ones):
        return 0
    else:
        return 1
    

def predictSpam(smap, num):
    slist = sorted(smap.values())
    k1 = preSpam(slist, num[0])
    k2 = preSpam(slist, num[1])
    k3 = preSpam(slist, num[2])

    return (k1, k2, k3)


def preDigits(slist, num):
    count = dict()
    for itr in range(num):
        if slist[itr][1] in count:
            count[slist[itr][1]] += 1
        else:
            count[slist[itr][1]] = 1

    mymax = -1
    mykey = 1
    
    for key in count:
        val = count[key]
        if (val > mymax):
            mymax = val
            mykey = key
    # print(mykey)
    return mykey



def predictDigits(smap, num, mbool):
    slist = sorted(smap.values(), reverse=mbool)
    k1 = preDigits(slist, num[0])
    k2 = preDigits(slist, num[1])
    k3 = preDigits(slist, num[2])

    return (k1, k2, k3)


def gaussian(testv, trainv):
    sigma = 1
    num = euclidean(testv, trainv)
    val = -num / (sigma * sigma)
    return math.exp(val)


def ploynomial(testv, trainv):
    val = 0
    gamma = .1
    coefficient = .25
    degree = 2
    val = np.dot(testv, trainv)
    return math.pow(val * gamma + coefficient, degree)


def computeKernelSpam(row, data, labels, num):
	smap = dict()
	for itr in range(data.shape[0]):
		smap[itr] = (euclidianDistance(row, data[itr]) , labels[itr])
	return predictSpam(smap, num)


def computeKernelDigits(row, data, labels, num):
    smap = dict()
    # std = np.std(row)
    for itr in range(data.shape[0]):
        distance = ploynomial(row, data[itr])
        # distance =  gaussian(row, data[itr])
        # distance = cosine(row, data[itr])
        smap[itr] = (distance , labels[itr])
    return predictDigits(smap, num, True)


def accCalc(predictions, labels):
    k1 = 0
    k2 = 0
    k3 = 0
    for idx in range(len(predictions)):
        if predictions[idx][0] == labels[idx]:
            k1 += 1
        if predictions[idx][1] == labels[idx]:
            k2 += 1
        if predictions[idx][2] == labels[idx]:
            k3 += 1

    acc = k1 / float(len(labels))
    print("Accuracy -> " + str(acc * 100))

    acc = k2 / float(len(labels))
    print("Accuracy -> " + str(acc * 100))

    acc = k3 / float(len(labels))
    print("Accuracy -> " + str(acc * 100))


def performKNNSpam(num):
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
        normalize(test)
        normalize(train)
        # print("Done Normalization")

        labels = [None] * test.shape[0] 
        for itr in range(test.shape[0]):
        	labels[itr] = computeKernelSpam(test[itr], train, trainlabels, num)
        	print(labels[itr])

        # Compute accuracy
        accCalc(labels, testlabels)
        exit()


def performKNNDigits(num):
    test = readHaar("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/HF/htest.txt")
    testlabels = readHaar("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/HF/htestlabels.txt")
    train = readHaar("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/HF/s20train.txt")
    trainlabels = readHaar("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/HF/s20labels.txt")

    # Normalize data
    # normalize(test)
    # normalize(train)

    labels = [None] * test.shape[0] 
    for itr in range(test.shape[0]):
        labels[itr] = computeKernelDigits(test[itr], train, trainlabels, num)
        print(labels[itr])


    accCalc(labels, testlabels)
    exit()


if __name__ == '__main__':
    performKNNSpam([1, 3, 7])
    # performKNNDigits([1, 3, 7])

    # Cosine -> False
    # Euclidian -> False
    # Gaussian - > True
    # Polynomial - > True

    # Do not Normalize for Spam and user your own euclidean
    # Do not normalize for ploynomial function, gamma = .1, coefficient = .25, K=7 -> 58% accuracy
