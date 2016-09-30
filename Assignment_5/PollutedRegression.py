import pyliblinear
# import sys
# import numpy as np
# import math


from ReadPolluted import readData, readLabels
from Regression import checkGradient, ridgeGradient, predict, sigmoidVector, accCalc
from normalizedata import normalize


def runRidge(traindata, trainlabels, testdata, testlabels):
    # Normalize test and train data
    normalize(traindata)
    normalize(testdata)

    weights = ridgeGradient(traindata, trainlabels)
    predictions = predict(weights, testdata)
    sigmoidVector(predictions)

    accuracy = accCalc(predictions, testlabels, .45)
    print("Accuracy " + str(accuracy))


def runLogistic(traindata, trainlabels, testdata, testlabels):
    # Normalize test and train data
    normalize(traindata)
    normalize(testdata)

    weights = checkGradient(traindata, trainlabels)
    predictions = predict(weights, testdata)
    sigmoidVector(predictions)

    # Accuracy
    accuracy = accCalc(predictions, testlabels, .5)
    print("Accuracy " + str(accuracy))


def writeToFile(data, labels, filename):
    fname = open(filename, 'w')

    for row in range(data.shape[0]):
        fname.write(str(labels[row]) + "\t")
        for col in range(data.shape[1]):
            fname.write(str(col + 1) + ":" + str(data[row][col]) + "\t")
        fname.write("\n")

    fname.close()


def toIterator(data):
    mylist = list()
    for row in range(data.shape[0]):
        newlist = list()
        for col in range(data.shape[1]):
            newlist.append(data[row][col])
        mylist.append(newlist)
    return mylist


def getAcc(predictions, labels):
    hit = 0
    for itr in range(labels):
        if predictions[itr] == labels[itr]:
            hit +=1
    print("Accuracy -> " + hit / float(len(labels)))
    pass


def runLibLinear(traindata, trainlabels, testdata, testlabels):
    # Convert data to list of list and labels to list
    traindata = toIterator(traindata)
    testdata = toIterator(testdata)
    trainlabels = trainlabels.tolist()
    testlabels = testlabels.tolist()

    print("Matrix Created")
    # Form feature matrix and then train with L1 and L2 logistic
    train = pyliblinear._liblinear.FeatureMatrix.__new__(traindata, assign_labels=trainlabels)
    test = pyliblinear._liblinear.FeatureMatrix.load(open("test.txt"))
    ridge = pyliblinear._liblinear.Solver.__new__(type=1)    # Ridge -L2 Logistic
    lasso = pyliblinear._liblinear.Solver.__new__(type=6)    # Lasso -L1 Logistic

    print("Matrxi Parsed. Now Training ")
    modelridge = pyliblinear._liblinear.Model.train(train, solver=ridge)

    print("Predicting")
    lassop = pyliblinear.Model.predict(modelridge, test)

    getAcc(list(lassop), testlabels)


if __name__ == '__main__':
    traindata = readData("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/spam_polluted/train_feature.txt")
    testdata = readData("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/spam_polluted/test_feature.txt")
    trainlabels = readLabels("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/spam_polluted/train_label.txt")
    testlabels = readLabels("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/spam_polluted/test_label.txt")

    # writeToFile(traindata, trainlabels, "train.txt")
    # writeToFile(testdata, testlabels, "test.txt")
    runLibLinear(traindata, trainlabels, testdata, testlabels)
    #runLogistic(traindata, trainlabels, testdata, testlabels)

    # runRidge(traindata, trainlabels, testdata, testlabels)
