import numpy as np
from normalizedata import normalize

from readData import readSpam, extractMatrix, readHaar
from scipy.spatial.distance import euclidean, cosine
import math

# mylist = [34, 37, 50, 53, 55]
# mylist = [52, 51, 56, 15, 6]
mylist = [54, 55, 37, 47, 19]
# mylist = [54, 55, 37, 47, 50] # 53 7 34 23 19


def euclidianDistance(testv, trainv):
    # return euclidean(testv, trainv)
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


def computeKernelSpam(row, data, labels, num):
    smap = dict()
    for itr in range(data.shape[0]):
        smap[itr] = (euclidianDistance(row, data[itr]) , labels[itr])
    return predictSpam(smap, num)


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

        train = train[:,mylist]
        test = test[:,mylist]

        # Normalize data
        # normalize(test)
        # normalize(train)
        # print("Done Normalization")

        labels = [None] * test.shape[0] 
        for itr in range(test.shape[0]):
        	labels[itr] = computeKernelSpam(test[itr], train, trainlabels, num)
        	print labels[itr]

        # Compute accuracy
        accCalc(labels, testlabels)
        exit()


if __name__ == '__main__':
    performKNNSpam([1,2,3])

    # Do not normalize and user your euclidean