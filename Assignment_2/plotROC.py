import matplotlib.pyplot as plt
import numpy as np


def plotROC(fprate, tprate):
    # x = # false_positive_rate
    # y = # true_positive_rate

    # This is the ROC curve
    plt.plot(fprate, tprate)
    plt.show()


def getAUC(fprate, tprate):
    # This is the AUC
    auc = np.trapz(tprate, fprate)
    print("AUC " + str(auc))


def rocAndAuc(prediction, labels, threshold):
    ziplist = zip(labels, prediction)
    zipped = sorted(ziplist, key=lambda t: t[1])
    rna(zipped, threshold)


def rna(zipped, threshold):
    fprate = list()
    tprate = list()

    '''
    pos = 0
    neg = 0
    for row in range(len(zipped)):
        if (zipped[row][0] == 1):
            pos += 1

    neg = len(zipped) - pos
    '''

    for row in range(len(zipped)):
        # Assign variables
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for new in range(len(zipped)):
            if (new > row):
                if (zipped[new][0] == 1):
                    tp += 1
                else:
                    fp += 1
            else:
                if (zipped[new][0] == 0):
                    tn += 1
                else:
                    fn += 1

        tprate.append(float(tp) / (tp + fn))
        fprate.append(float(fp) / (fp + tn))
    # print(tprate)
    # print(fprate)

    # Call the plot and auc
    plotROC(fprate, tprate)
    getAUC(fprate, tprate)


'''
# Another function
def roaauc(prediction, labels, threshold):
    fprate = list()
    tprate = list()

    pos = 0
    neg = 0
    for row in range(labels.shape[0]):
        if (labels[row] == 1):
            pos += 1
    neg = labels.shape[0] - pos

    # Assign variables
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

        tprate.append(float(tp) / pos)
        fprate.append(float(fp) / neg)

    # Call the plot and auc
    plotROC(fprate, tprate)
    getAUC(fprate, tprate)
'''
