from readData import readPerceptron

import numpy as np
import copy
import math
from scipy.spatial.distance import euclidean


def gaussian(testv, trainv):
    sigma = .1
    num = euclidean(testv, trainv)
    val = -num / (sigma * sigma)
    return math.exp(val)


def misVal(row, data, mvec, labels, B):
    mysum = 0
    for itr in range(data.shape[0]):
        # kernel = np.dot(data[itr], data[row])
        kernel = gaussian(data[itr], data[row])
        mysum += mvec[itr] * labels[itr] * kernel

    return (mysum + B) * labels[row]


def dualPerceptron(data, labels):
    condition = True
    counter = 0
    mvec = [0] * data.shape[0]
    B = 0

    while(condition):
        counter += 1
        miss = set()
        condition = False
        for row in range(data.shape[0]):
            val = misVal(row, data, mvec, labels, B)
            if val <= 0:
                mvec[row] += 1
                B = B + labels[row]
                condition = True
                miss.add(row)
        # print(miss)
        print("Iteration " + str(counter) + ", Total Mistakes " + str(len(miss)))
        

def performupdate(data, labels):
    condition = True
    counter = 0
    mvec_old = [0] * data.shape[0]
    mvec_new = [0] * data.shape[0]

    while(condition):
        counter += 1
        miss = set()
        condition = False
        for row in range(data.shape[0]):
            val = misVal(row, data, mvec_old, labels)
            # val = sumProductVal(row, data, mvec_old, labels)
            if val <= 0:
                mvec_new[row] += 1
                condition = True
                miss.add(row)
        mvec_old = list(mvec_new)
        print("Iteration " + str(counter) + ", Total Mistakes " + str(len(miss)))


def main():
    #(data, labels) = readPerceptron("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_7/perceptronData.txt")
    # dualPerceptron(data,labels)

    (Sdata, Slabels) = readPerceptron("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_7/spiral.txt")
    dualPerceptron(Sdata, Slabels)

if __name__ == '__main__':
    main()
