import matplotlib.pyplot as plt


def parseGen():
    myfile = open('AdaBoostingResults.txt', 'r')

    Rerr = list()
    Testerr = list()
    Trainerr = list()
    auc = list()

    for line in myfile:
        arr = line.split()
        Rerr.append(float(arr[5]))
        Trainerr.append(float(arr[7]))
        Testerr.append(float(arr[9]))
        auc.append(float(arr[11]))

    myfile.close()

    my = range(len(auc))

    plt.ylabel('ROUND ERROR')
    plt.plot(my, Rerr, linewidth=2.0)
    plt.show()

    plt.ylabel('TRAIN AND TEST ERROR')
    plt.plot(my, Trainerr, linewidth=2.0)
    plt.plot(my, Testerr, linewidth=2.0)
    plt.show()

    plt.ylabel('AUC')
    plt.plot(my, auc, linewidth=2.0)
    plt.show()

    print("done")

if __name__ == '__main__':
    parseGen()
