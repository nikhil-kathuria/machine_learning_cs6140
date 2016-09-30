import numpy as np


def readData(name):
    full = np.loadtxt(name, dtype=np.float64)
    return full


def readLabels(name):
    full = np.loadtxt(name, dtype=np.float64)
    return full
    # return np.array(mylist)


if __name__ == '__main__':
    # main()
    pass
