from sklearn import svm
import numpy as np
from readData import readSpam, extractMatrix
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


def performSVM(knl):
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
        normalize(test)
        normalize(train)

        print("Done Normalization")

        # t = 0 Linear
        # t = 1 polynomial
        # t = 2 rbf
        # t = 3 sigmoid

        prob = svm_problem(trainlabels.tolist(), train.tolist())
        model = svm_train(prob, '-t 2 -c 2 -g 2')
        print("Test Accuracy ")
        (p_label, p_acc, p_val) = svm_predict(testlabels.tolist(), test.tolist(), model)
        
        print("Train Accuracy ")
        (p_label, p_acc, p_val) = svm_predict(trainlabels.tolist(), train.tolist(), model)



def performSCI():
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

        print("Done Normalization")
        # clf = svm.LinearSVC()
        clf = svm.SVC(kernel='rbf', C=10, tol=0.001)
        clf.fit(train, trainlabels)
        predictions = clf.predict(test)
        accCalc(predictions, testlabels)
        
        # exit()

if __name__ == '__main__':
    # performSCI()
    performSVM('polynomial')
    #performSVM('linear')
    # performSVM('rbf')
    # performSVM('sigmoid')
