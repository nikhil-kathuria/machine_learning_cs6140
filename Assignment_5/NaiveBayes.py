# from readData import readSpam, extractMatrix

from normalizedata import normalize

# from GDA import computeU, getPrior, accCalc

# from plotROC import rocAndAuc

# from binning import bindata

# from meanbins import bindata

import numpy as np

import math

Epsilon = float(.011)


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


def fixvar(cov):
    mini = np.amin(cov)
    # print(mini)
    for itr in range(len(cov)):
        if cov[itr] == 0:
            cov[itr] = mini
            # print(cov[itr])
    return cov


def pxy(data, var, mean):
    if var == 0:
        print("zero")
    part1 = math.pow((2 * math.pi), .5) * math.pow(var, .5)
    part2 = math.exp(-.5 * (math.pow((data - mean), 2) / float(var)))
    if part1 == 0:
        print("Var " + str(var))
    if part2 == 0:
        print("Var " + str(var) + " Mean " + str(mean) + " Data " + str(data))
    return part2 / part1


def bucketDistribution(data, bincount):
    holder = list()
    for idx in range(data.shape[1]):
        col = data[:, idx]
        holder.append(np.histogram(col, bins=bincount))
    return holder


def bernoulliDistribution(data, meanlist):
    holder = list()
    flen = data.shape[0]
    for idx in range(data.shape[1]):
        col = data[:, idx]
        lessmean = 0
        greatmean = 0
        for elem in col:
            # if (np.isfinite(elem)):
            if elem <= meanlist[idx]:
                lessmean += 1
            else:
                greatmean += 1
        # Calculuate the ratio for less and greater than mean
        lessratio = float(lessmean + 1) / (flen)
        greateratio = float(greatmean + 1) / (flen)
        holder.append((lessratio, greateratio))
    return holder


def partionMatrix(traindata, trainlabels):
    zero = list()
    one = list()
    for idx in range(traindata.shape[0]):
        if trainlabels[idx] == 0:
            zero.append(idx)
        else:
            one.append(idx)

    zeromat = traindata[zero, :]
    onemat = traindata[one, :]
    return (zeromat, onemat)


def predictGaussian(data, U0, U1, var, labels):
    (py0, py1) = getPrior(labels)
    predictions = list()
    vals = list()
    var = fixvar(var)
    # np.savetxt('var.txt', var, fmt='%10.4f')

    for idx in range(data.shape[0]):
        row = data[idx, :]
        prowy0 = math.log(py0)
        prowy1 = math.log(py1)
        # Iterate over each row value
        for itr in range(data.shape[1]):
            pxy0 = pxy(row[itr], var[itr], U0[itr])
            if (pxy0 == 0):
                continue
            prowy0 = prowy0 + math.log(pxy0)

            pxy1 = pxy(row[itr], var[itr], U1[itr])
            if (pxy1 == 0):
                continue
            prowy1 = prowy1 + math.log(pxy1)

        # print("Zero " + str(prowy0) + " One " + str(prowy1))
        # print(str(prowy0) + " " + str(prowy1))
        if (prowy0 >= prowy1):
            predictions.append(0)
        else:
            predictions.append(1)
        if (prowy0 == 0 or prowy1 == 0):
            vals.append(1)
        else:
            vals.append(prowy1 / prowy0)
        # print(" Py0 " + str(prowy0) + " Py1 " + str(prowy1))
    return (vals, predictions)


def predictBernoulli(data, fldy0, fldy1, U, labels):
    (py0, py1) = getPrior(labels)
    predictions = list()
    vals = list()

    for idx in range(data.shape[0]):
        row = data[idx, :]
        prowy0 = py0
        prowy1 = py1
        # Iterate over each row value
        for itr in range(len(row)):
            if (np.isfinite(row[itr])):
                if row[itr] < U[itr]:
                    prowy0 = prowy0 * fldy0[itr][0]
                    prowy1 = prowy1 * fldy1[itr][0]
                else:
                    prowy0 = prowy0 * fldy0[itr][1]
                    prowy1 = prowy1 * fldy1[itr][1]
        # Check for probability of whihc is higher
        if prowy0 >= prowy1:
            predictions.append(0)
        else:
            predictions.append(1)
        vals.append(prowy0 / prowy1)
    return (vals, predictions)


def predictBinning(data, bucket, totlen0, totlen1, labels):
    (py0, py1) = getPrior(labels)
    predictions = list()
    vals = list()

    for idx in range(data.shape[0]):
        row = data[idx, :]
        prowy0 = py0
        prowy1 = py1
        # Iterate over each row value
        for itr in range(len(row)):
            pnum0 = 0
            pnum1 = 0
            val = row[itr]
            tup = bucket[itr]
            # print(tup[0])
            # print(tup[1])
            for rownum in range(len(tup[0])):
                if val >= tup[2][rownum] and val <= tup[2][rownum + 1]:
                    pnum0 = tup[0][rownum]
                    pnum1 = tup[1][rownum]

            # Now calculate probability by laplace smoothing
            prob0 = (1 + pnum0) / float(len(tup[0]) + totlen0)
            prob1 = (1 + pnum1) / float(len(tup[1]) + totlen1)
            prowy0 = prowy0 * prob0
            prowy1 = prowy1 * prob1

            # print(str(prowy0) + " " + str(prowy1))

        # Compare probabilities
        if (prowy0 > prowy1):
            predictions.append(0)
        else:
            predictions.append(1)
        vals.append(prowy0 / prowy1)
    return (vals, predictions)


def runBer(name):
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

        U = np.mean(traindata, axis=0)
        (zeromat, onemat) = partionMatrix(traindata, trainlabels)
        fdistrY0 = bernoulliDistribution(zeromat, U)
        fdistrY1 = bernoulliDistribution(onemat, U)

        (values, predictions) = predictBernoulli(testdata, fdistrY0, fdistrY1, U, trainlabels)

        sumacc += accCalc(predictions, testlabels)
        rocAndAuc(values, predictions, testlabels)
    print("Average Accuracy : " + str(sumacc / len(bucketmap)))


def runGau(name):
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

        (U0, U1) = computeU(traindata, trainlabels)

        var = np.var(traindata, 0, dtype=np.float64)

        (values, predictions) = predictGaussian(testdata, U0, U1, var, trainlabels)

        sumacc += accCalc(predictions, testlabels)

        rocAndAuc(values, predictions, testlabels)
    print("Average Accuracy : " + str(sumacc / len(bucketmap)))


def runBinning(name):
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

        # Normalize test and train data
        # normalize(traindata)
        # normalize(testdata)
        (zeromat, onemat) = partionMatrix(traindata, trainlabels)

        buckets = bindata(traindata, zeromat, onemat, 9, trainlabels)
        # exit()
        '''for itr in range(len(buckets)):
            print(buckets[itr][0])
            print(len(zeromat))
            print(buckets[itr][1])
            print(len(onemat))
            print(buckets[itr][2])
            # print(len(buckets[itr][0]))
            # print(itr)
            pass
        exit()'''
        (values, predictions) = predictBinning(testdata, buckets, len(zeromat), len(onemat), trainlabels)
        sumacc += accCalc(predictions, testlabels)
        rocAndAuc(values, predictions, testlabels)
    print("Average Accuracy : " + str(sumacc / len(bucketmap)))


def main():
    # runBer("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_1/spambase.data.txt")
    runGau("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_1/spambase.data.txt")
    # runBinning("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_1/spambase.data.txt")
    pass


if __name__ == '__main__':
    main()
