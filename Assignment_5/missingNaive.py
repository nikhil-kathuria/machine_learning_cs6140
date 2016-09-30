import numpy as np
np.set_printoptions(threshold=np.inf)


from readMissing import read
from normalizedata import normalize
from NaiveBayes import partionMatrix, bernoulliDistribution, predictBernoulli, accCalc


def computeMean(data):
    meanlist = list()
    for itr in range(data.shape[1]):
        localsum = float(0)
        count = 0
        col = data[:, itr]

        # Iterate through the column
        for row in range(len(col)):
            if np.isfinite(col[row]):
                localsum += col[row]
                count += 1

        # Caclulate and store mean
        if count == 0:
            meanlist.append(float(0))
        else:
            meanlist.append(localsum / data.shape[0])

    # return the meanlist
    return meanlist


def runMissing(traindata, trainlabels, testdata, testlabels):
    # Normalize test and train data
    normalize(traindata)
    normalize(testdata)

    # print(traindata.shape)
    U = computeMean(traindata)
    (zeromat, onemat) = partionMatrix(traindata, trainlabels)

    fdistrY0 = bernoulliDistribution(zeromat, U)
    fdistrY1 = bernoulliDistribution(onemat, U)

    (values, predictions) = predictBernoulli(testdata, fdistrY0, fdistrY1, U, trainlabels)
    accCalc(predictions, testlabels)


if __name__ == '__main__':
    (traindata, trainlabels) = read("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/missing/train.txt")
    (testdata, testlabels) = read("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/missing/test.txt")

    runMissing(traindata, trainlabels, testdata, testlabels)
    # print(testlabels)
    # val = testdata[0][2]
    # np.isfinite(val)
