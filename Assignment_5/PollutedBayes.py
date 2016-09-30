import sys
import numpy as np

sys.path.insert(0, '/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_3')

from NaiveBayes import computeU, predictGaussian, accCalc
from normalizedata import normalize
from ReadPolluted import readData, readLabels


def runGau(traindata, trainlabels, testdata, testlabels):
    # Normalize test and train data
    normalize(traindata)
    normalize(testdata)

    (U0, U1) = computeU(traindata, trainlabels)

    var = np.var(traindata, 0, dtype=np.float64)

    (values, predictions) = predictGaussian(testdata, U0, U1, var, trainlabels)

    accCalc(predictions, testlabels)


if __name__ == '__main__':
    traindata = readData("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/spam_polluted/train_feature.txt")
    testdata = readData("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/spam_polluted/test_feature.txt")
    trainlabels = readLabels("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/spam_polluted/train_label.txt")
    testlabels = readLabels("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/spam_polluted/test_label.txt")

    runGau(traindata, trainlabels, testdata, testlabels)
