# from sklearn import svm
import numpy as np

# from sklearn import svm
import numpy as np
from normalizedata import normalize
import sys

sys.path.insert(0, '/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_6/libsvm-3.20/python/')
from svmutil import *


def accCalc(predictions, labels):
    hits = 0
    for idx in range(len(predictions)):
        if predictions[idx] == labels[idx]:
            hits += 1
    acc = hits / float(len(labels))
    print("Accuracy -> " + str(acc * 100))


def loaddata(features, labels):
    data = np.loadtxt(features, dtype=int)
    labels = np.loadtxt(labels, dtype=int)

    return data, labels


def performSVM(train, trainlabels, test, testlabels):

    '''
    clf = svm.LinearSVC()
    # clf = svm.SVC()
    clf.fit(train, trainlabels)
    predictions = clf.predict(test)
    accCalc(predictions, testlabels)
    '''

    # Normalize data
    # normalize(test)
    # normalize(train)

    print("Done Normalization")

    trainlabels = trainlabels.tolist()
    train = train.tolist()

    testlabels = testlabels.tolist()
    test = test.tolist()

    print("Converted to LIBSVM format")

    prob = svm_problem(trainlabels, train)
    print("Trainining Model")

    model = svm_train(prob, '-t 0')
    print("Test Accuracy ")
    (p_label, p_acc, p_val) = svm_predict(testlabels, test, model)

    print("Train Accuracy ")
    (p_label, p_acc, p_val) = svm_predict(trainlabels, train, model)


if __name__ == '__main__':
    test, testlabels = loaddata("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/HF/htest.txt", "/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/HF/htestlabels.txt")
    train, trainlabels = loaddata("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/HF/s20train.txt", "/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/HF/s20labels.txt")
    performSVM(train, trainlabels, test, testlabels)
