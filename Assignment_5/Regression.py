import numpy

numpy.set_printoptions(threshold=numpy.inf)

import sys

import math

# from readData import readHouse, readSpam, extractMatrix, readSmallSpam

# from plotROC import rocAndAuc

# from normalizedata import normalize

# Logistic Spam = .0005
# Linear Spam = .0001
# Linear House = .001
# Iniiate Lambda value
lam = .001
cost = .1


def sigmoid(value):
    return 1 / (1 + math.exp(-value))


def sigmoidVector(data):
    for row in range(data.shape[0]):
            data[row] = sigmoid(data[row])


def sigmoidAll(data):
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            data[row][col] = sigmoid(data[row][col])
    return data


def logistichypoth(weights, data):
    datalin = linearhypoth(weights, data)
    sigmoidVector(datalin)
    return datalin


def linearhypoth(weights, data):
    return data.dot(weights)


def computeCost(weights, data, labels):
    hypoth_V = data.dot(weights)
    sumerror = float(0)

    for row in range(labels.shape[0]):
        sumerror += pow(hypoth_V[row] - labels[row], 2)
    return .5 * sumerror


def likelyhood(weights, data, labels):
    hypoth_V = logistichypoth(weights, data)

    sumlikely = float(0)
    for row in range(labels.shape[0]):
        label = labels[row]
        # print(hypoth_V[row])
        # print(math.log2(hypoth_V[row]) + (1 - label) * math.log2(1 - hypoth_V[row]))
        sumlikely += label * math.log2(hypoth_V[row]) + (1 - label) * math.log2(1 - hypoth_V[row])
    return sumlikely / labels.shape[0]


def ridgeLikelyhood(weights, data, labels):
    hypoth_V = logistichypoth(weights, data)
    sumlikely = float(0)

    for row in range(labels.shape[0]):
        label = labels[row]
        # Compute Likelyhood
        sumlikely += label * math.log2(hypoth_V[row]) + (1 - label) * math.log2(1 - hypoth_V[row])

    # Compute penalty
    penalty = sum(numpy.multiply(weights, weights)) * (cost / (2 * labels.shape[0]))
    sumlikely = sumlikely / labels.shape[0]
    value = sumlikely - penalty
    return value


def updateWeights(hypoth_V, weights, data, labels):
    diff = labels - hypoth_V
    weights = weights + lam * (numpy.transpose(data).dot(diff))
    return weights


def ridgeweights(hypoth_V, weights, data, labels):
    diff = labels - hypoth_V
    weights = weights + lam * (numpy.transpose(data).dot(diff)) + (cost / labels.shape[0]) * weights
    return weights


def checkGradient(data, labels):
    # Add intercpet and assign weights array
    intercept = numpy.ones((data.shape[0], 1), dtype=float)
    data = numpy.hstack((intercept, data))
    weights = numpy.zeros(data.shape[1])

    # Variables for const and exit condition
    j_history = list()
    j_old = sys.maxsize
    j_new = 0
    errordiff = .0001
    counter = 1

    while abs(j_old - j_new) > errordiff:
        j_old = j_new
        hypoth_V = logistichypoth(weights, data)

        weights = updateWeights(hypoth_V, weights, data, labels)

        j_new = likelyhood(weights, data, labels)
        j_history.append(j_new)

        counter += 1

        print("Log likelyhood " + str(j_new))
    return weights


def ridgeGradient(data, labels):
    # Add intercpet and assign weights array
    intercept = numpy.ones((data.shape[0], 1), dtype=float)
    data = numpy.hstack((intercept, data))
    weights = numpy.zeros(data.shape[1])

    # Variables for const and exit condition
    j_history = list()
    j_old = sys.maxsize
    j_new = 0
    errordiff = .0001
    counter = 1

    while abs(j_old - j_new) > errordiff:
        j_old = j_new
        hypoth_V = logistichypoth(weights, data)

        weights = ridgeweights(hypoth_V, weights, data, labels)

        j_new = ridgeLikelyhood(weights, data, labels)
        j_history.append(j_new)

        counter += 1

        print("Ridge Log likelyhood " + str(j_new))
    return weights


def predict(weights, test):
    intercept = numpy.ones((test.shape[0], 1), dtype=float)
    test = numpy.hstack((intercept, test))
    return test.dot(weights)


def msecalc(predictions, labels):
    summse = float(0)
    for row in range(labels.shape[0]):
        summse += pow(predictions[row] - labels[row], 2)
        print("Labels " + str(labels[row]) + " Prediction " + str(predictions[row]))
    return summse / labels.shape[0]


def accCalc(prediction, labels, threshold):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for row in range(labels.shape[0]):
        if (prediction[row] >= threshold):
            if (labels[row] == 1):
                tp += 1
            else:
                fp += 1
        else:
            if (labels[row] == 0):
                tn += 1
            else:
                fn += 1
    print("True Positives " + str(tp))
    print("False Positives " + str(fp))
    print("True Negatives " + str(tn))
    print("Flase Negatives " + str(fn))
    return (float(tp + tn) / labels.shape[0] * 100)


def main():
    # runLogisticRegressin()
    # runLinearRegression()
    # testlogistic()
    pass

if __name__ == '__main__':
    main()
