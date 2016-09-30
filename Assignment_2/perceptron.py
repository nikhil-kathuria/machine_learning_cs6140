from readData import readPerceptron

import numpy

# Assign a value to lambda
lam = 1


def updatedata(data, labels):
    intercept = numpy.ones((data.shape[0], 1), dtype=float)
    data = numpy.hstack((intercept, data))
    weights = numpy.zeros(data.shape[1])

    return (data, weights)


def missandData(data, labels):
    for row in range(labels.shape[0]):
        if labels[row] == -1:
            data[row, :] = -1 * data[row, :]
    return data


def findmiss(data, weights):
    miss = set()
    for row in range(data.shape[0]):
        if weights.transpose().dot(data[row, :]) < 0:
            miss.add(row)
    return miss


def printweights(weights):
    print("Actual weights " + str(weights))
    for itr in range(1, weights.shape[0]):
        weights[itr] = -1 * (weights[itr] / weights[0])

    print("Normalized weights " + str(weights))


def performupdate(data, labels):
    (data, weights) = updatedata(data, labels)
    data = missandData(data, labels)
    condition = True
    counter = 0

    while(condition):
        counter += 1
        miss = set()
        condition = False
        for row in range(data.shape[0]):
            if (weights.transpose().dot(data[row, :]) <= 0):
                weights = weights + lam * data[row, :]
                condition = True
                miss.add(row)
        print("Iteration " + str(counter) + ", Total Mistakes " + str(len(miss)))
    printweights(weights)


def main():
    (data, labels) = readPerceptron("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_2/perceptronData.txt")
    performupdate(data, labels)


if __name__ == '__main__':
    main()
