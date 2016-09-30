import numpy as np
from normalizedata import normalize

from readData import readSpam, extractMatrix, readHaar
from scipy.spatial.distance import euclidean, cosine
import math


def euclidianDistance(testv, trainv):
	# return euclidean(testv, trainv)
	dis = float(0)
	for itr in range(len(testv)):
		dis += math.pow(testv[itr] - trainv[itr], 2)
	dis = math.pow(dis, .5)
	return dis


def predictSpam(smap, num):
	slist = smap.values()
	zeros = 0
	ones = 0
	for itr in range(len(slist)):
		if (slist[itr][1] == 0):
			zeros += 1
		else:
			ones += 1

	# Return the majority
	if (zeros >= ones):
		return 0
	else:
		return 1


def predictDigits(smap, num, mbool):
    slist = smap.values()
    count = dict()
    # print(len(slist))
    for itr in range(len(slist)):
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


def gaussian(std, testv, trainv):
    num = euclidean(testv, trainv)
    val = -num / (2 * std * std)
    return math.exp(val)


def ploynomial(testv, trainv):
    val = 0
    gamma = 1
    coefficient = .25
    degree = 2
    val = np.dot(testv, trainv)
    return math.pow(val * gamma + coefficient, degree)


def computeKernelSpam(row, data, labels, num):
    smap = dict()
    for itr in range(data.shape[0]):
        distance = euclidianDistance(row, data[itr])
        if distance <= num:
            smap[itr] = (distance, labels[itr])
    return predictSpam(smap, num)


def computeKernelDigits(row, data, labels, num):
    smap = dict()
    for itr in range(data.shape[0]):
        distance = cosine(row, data[itr])
        if distance <= num:
            smap[itr] = (distance , labels[itr])
    return predictDigits(smap, num, False)


def accCalc(predictions, labels):
    hits = 0
    for idx in range(len(predictions)):
        if predictions[idx] == labels[idx]:
            hits += 1
    acc = hits / float(len(labels))
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
        # normalize(test)
        # normalize(train)
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
    # performKNNSpam(2.5)
    performKNNDigits(.83)

    # Cosine -> False
    # Euclidian -> False
    # Gaussian - > True
    # Polynomial - > True
