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


def rocAndAuc(values, prediction, labels):
    ziplist = zip(values, prediction, labels)
    zipped = sorted(ziplist, key=lambda t: t[0])
    rna(zipped)


def rna(zipped):
    fprate = list()
    tprate = list()

    for row in range(len(zipped)):
        # Assign variables
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for new in range(len(zipped)):
            if (new <= row):
                if (zipped[new][2] == 1):
                    tp += 1
                else:
                    fp += 1
            else:
                if (zipped[new][2] == 0):
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
