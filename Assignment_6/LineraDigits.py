import pyliblinear
import numpy as np


def accCalc(predictions, labels):
    hits = 0
    for idx in range(len(predictions)):
        if predictions[idx] == labels[idx]:
            hits += 1
    acc = hits / float(len(labels))
    print("Accuracy -> " + str(acc * 100))


def runLinLinear(traindata, trainlabels, testdata, testlabels):
    # Convert data to list of list and labels to list
    traindata = traindata.tolist()
    testdata = testdata.tolist()
    trainlabels = trainlabels.tolist()
    testlabels = testlabels.tolist()

    print("Matrix Created")
    # Form feature matrix and then train with L1 and L2 logistic
    train = pyliblinear._liblinear.FeatureMatrix.from_iterables(trainlabels, traindata)
    test = pyliblinear._liblinear.FeatureMatrix.from_iterables(testlabels, testdata)
    ridge = pyliblinear._liblinear.Solver.__new__(type=1)    # Ridge -L2 Logistic
    # lasso = pyliblinear._liblinear.Solver.__new__(type=6)    # Lasso -L1 Logistic

    print("Matrxi Parsed. Now Training ")
    modelridge = pyliblinear._liblinear.Model.train(train, solver=ridge)

    print("Predicting")
    predictions = pyliblinear.Model.predict(modelridge, test)

    accCalc(predictions, testlabels)


def loaddata(features, labels):
    data = np.loadtxt(features, dtype=int)
    labels = np.loadtxt(labels, dtype=int)

    return data, labels

if __name__ == '__main__':
    test, testlabels = loaddata("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/HF/htest.txt", "/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/HF/htestlabels.txt")
    train, trainlabels = loaddata("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/HF/strain.txt", "/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/HF/slabels.txt")
    runLinLinear  (train, trainlabels, test, testlabels)