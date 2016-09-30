from readData import readSpam, extractMatrix

from normalizedata import normalize

import numpy as np

import math


def meanMat(mat):
    return np.mean(mat, axis=0)


def computeU(traindata, trainlabels):
    zero = list()
    one = list()
    for idx in range(traindata.shape[0]):
        if trainlabels[idx] == 0:
            zero.append(idx)
        else:
            one.append(idx)

    U0 = meanMat(traindata[zero, :])
    U1 = meanMat(traindata[one, :])

    return(U0, U1)


def pxGivenY(cov, det, inv, diff, nby2):
    par1 = 1 / (math.pow((2 * math.pi), nby2) * math.pow(det, .5))
    part2 = math.exp(-.5 * diff.dot((inv.dot(diff))))
    return par1 * part2


def getPrior(labels):
    ones = float(0)
    for label in labels:
        if label == 1:
            ones += 1

    zeros = len(labels) - ones

    return(zeros / len(labels), ones / len(labels))


def accCalc(predictions, labels):
    hits = 0
    for idx in range(len(predictions)):
        if predictions[idx] == labels[idx]:
            hits += 1
    acc = hits / float(len(labels))
    print("Accuracy -> " + str(acc * 100))
    return acc * 100


def getPrediction(data, cov, det, inv, U0, U1, labels):
    predictions = list()
    nby2 = float(data.shape[0]) / 2
    (py0, py1) = getPrior(labels)
    for idx in range(data.shape[0]):
        row = data[idx, :]
        diff0 = row - U0
        diff1 = row - U1
        pxy0 = pxGivenY(cov, det, inv, diff0, nby2)
        pxy1 = pxGivenY(cov, det, inv, diff1, nby2)

        if (py0 * pxy0) > (py1 * pxy1):
            predictions.append(0)
        else:
            predictions.append(1)
    return predictions


def runGDA(name):
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
        normalize(traindata)
        normalize(testdata)

        (U0, U1) = computeU(traindata, trainlabels)

        cov = np.cov(traindata.T)

        # np.savetxt('train.txt', traindata, fmt='%10.4f')
        # np.savetxt('cov.txt', cov, fmt='%10.4f')

        det = np.linalg.det(cov)
        inv = np.linalg.pinv(cov)
        predictions = getPrediction(testdata, cov, det, inv, U0, U1, trainlabels)
        sumacc += accCalc(predictions, testlabels)

    print("Average Accuracy : " + str(sumacc / len(bucketmap)))


def main():
    runGDA("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_1/spambase.data.txt")


if __name__ == '__main__':
    main()
    # print(np.cov.__doc__)
