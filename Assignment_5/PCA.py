import sys
import numpy as np

sys.path.insert(0, '/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_3')

from sklearn.decomposition import PCA
from ReadPolluted import readData, readLabels
from NaiveBayes import computeU, predictGaussian, accCalc

percent = 10


def performReduction(traindata, testdata):

    data = np.concatenate((traindata, testdata), axis=0)
    n_comp = int((percent / 100) * data.shape[1])

    trainrange = range(0, traindata.shape[0])
    testrange = range(traindata.shape[0], data.shape[0])

    pca = PCA(n_components=n_comp)
    new = pca.fit_transform(data)

    # Now split back to test and train
    train = new[trainrange, :]
    test = new[testrange, :]

    return (train, test)


def runGau(traindata, trainlabels, testdata, testlabels):
    (U0, U1) = computeU(traindata, trainlabels)

    var = np.var(traindata, 0, dtype=np.float64)

    (values, predictions) = predictGaussian(testdata, U0, U1, var, trainlabels)

    accCalc(predictions, testlabels)


if __name__ == '__main__':
    traindata = readData("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/spam_polluted/train_feature.txt")
    testdata = readData("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/spam_polluted/test_feature.txt")
    trainlabels = readLabels("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/spam_polluted/train_label.txt")
    testlabels = readLabels("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/spam_polluted/test_label.txt")

    (traindata, testdata) = performReduction(traindata, testdata)

    print(traindata.shape)
    print(testdata.shape)

    runGau(traindata, trainlabels, testdata, testlabels)
