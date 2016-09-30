from readData import readSmallSpam

import numpy as np

from normalizedata import normalize

from scipy.spatial.distance import euclidean, cosine

import sys

from random import shuffle


'''
def valCalc(mat, y):
    mymin = sys.maxsize
    for itr in range(zeromat.shape[1]):
        if (col == itr):
            continue
        kernel = euclidean(mat[:,col], mat[:,itr])
        if (kernel < mymin):
            mymin = kernel
    return mymin

def perfomUpdate(data, zeromat, onemat, labels):
    weight[] = [0] * data.shape[1]
    for col in range(data.shape[1]):

        weight[col] = weight[col] + valCalc(zeromat, col) + valCalc(onemat, col)

    join = range(data.shape[1])
    zipped = zip(join, weight)
    slist = sorted(zipped, key=lambda x : x[1], reverse=True)
    print(slist)
'''

def genlist(num):
    mylist = range(num)
    shuffle(mylist)
    return mylist


def nearHM(data, row, labels):
    orow = 0
    zrow = 0
    ok = sys.maxsize
    zk = sys.maxsize
    for itr in range(data.shape[0]):
        if (itr == row):
            continue
        kernel = euclidean(data[row], data[itr])

        # Check for when label is 0
        if (labels[row] == 0):
            if (kernel < zk):
                zk = kernel
                zrow = row

        # Check when label is 1
        elif (labels[row] == 1):
            if(kernel < ok):
                ok = kernel
                orow = row

    # Now return one and zero diff matrix
    ones = (data[row] - data[orow]) ** 2
    zeros = (data[row] - data[zrow]) ** 2
    return (zeros, ones)


def reliefAlgo(data, labels):
    mylist = genlist(data.shape[0])
    # mylist = [2,4, 56, 80, 112, 456, 812, 1000, 1223, 1899, 2345, 3245, 3456, 4100]
    weight = np.zeros(data.shape[1])
    # for row in range(data.shape[0]):
    for row in mylist[:1000]:
        (zeros, ones) = nearHM(data, row, labels)
        if (labels[row] == 0):
            weight = weight - zeros + ones
        else:
            weight = weight + zeros - ones

        # Update weights
        print(row)
    weight = weight / len(mylist)
    print(weight)

    join = range(data.shape[1])
    zipped = zip(join, weight)
    slist = sorted(zipped, key=lambda x : x[1])
    print(slist[:10])
    


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


if __name__ == '__main__':
    (data, labels) = readSmallSpam("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_1/spambase.data.txt")
    normalize(data)
    # (zeromat, onemat) = partionMatrix(data, labels)
    # perfomUpdate(data, zeromat, onemat, labels)
    reliefAlgo(data, labels)

