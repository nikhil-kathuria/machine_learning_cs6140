import numpy

import random

from collections import defaultdict


def read(full):
    data = full[:, range(full.shape[1] - 1)]

    labels = full[:, full.shape[1] - 1]

    return (data, labels)


def readHouse(name):
    full = numpy.loadtxt(name, dtype=float)
    return read(full)


def readSmallSpam(name):
    full = numpy.loadtxt(name, delimiter=',', dtype=float)
    return read(full)


def readSpam(name):
    full = numpy.loadtxt(name, delimiter=',', dtype=float)
    slist = list(range(full.shape[0]))

    buckets = 10
    # Shuffle the list
    random.shuffle(slist, random.random)

    # Get count for each bucket
    if (len(slist) % buckets == 0):
        size = int(len(slist) / buckets)
    else:
        size = int(len(slist) / buckets) + 1

    # Dictionary to store rowids of bucket
    bucketmap = defaultdict(list)
    counter = 0
    for itr in range(0, len(slist)):
        if(itr % size == 0):
            counter += 1
        bucketmap[counter].append(slist[itr])

    return (bucketmap, full)


def extractMatrix(bucketmap, keys, full):
    final = list()
    for key in keys:
        final = final + bucketmap[key]

    mat = full[final, :]

    return read(mat)


def readPerceptron(name):
    full = numpy.loadtxt(name, dtype=float)
    return read(full)


def main():
    readHouse("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_1/housing_train.txt")
    (bucketmap, full) = readSpam("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_1/spambase.data.txt")
    for key in bucketmap:
        print(key)
        print(bucketmap[key])

if __name__ == '__main__':
    main()
