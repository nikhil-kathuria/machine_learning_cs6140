import numpy as np
from Regression import sigmoid, sigmoidAll

LR = .02


def sigmoidVector(data):
    for row in range(data.shape[0]):
        data[row][0] = sigmoid(data[row][0])
    return data


def randomweights(row, col):
    mat = np.random.rand(row, col)
    mat = mat - float(.5)
    return mat


def getHidden(data, Wij):
    Hidden = np.ones((8, 3))
    for col in range(data.shape[0]):
        column = data[:, col]
        row = np.ones(Wij.shape[1])
        for col2 in range(Wij.shape[1]):
            row[col2] = column.dot(Wij[:, col2:col2 + 1])
        Hidden[col] = row
    return sigmoidAll(Hidden)


def getFinal(Hidden, Wjk):
    Final = np.ones((8, 8))
    Hidden = np.transpose(Hidden)
    for rowk in range(Final.shape[0]):
        out = np.zeros((8, 1))
        sumwjk = float(0)
        row = Hidden[:, rowk:rowk + 1]
        for rowj in range(Wjk.shape[0]):
            sumwjk = sumwjk + Wjk[rowj][rowk] * row[rowj][0]
            out[rowk][] = sigmoid(sumwjk)
        print(out)
    return sigmoidAll(Final)


def genweights():
    data = np.eye(8, dtype=float)

    L2 = np.zeros((3, 1), dtype=float)
    L3 = np.zeros((8, 1), dtype=float)
    Wij = randomweights(data.shape[0], L2.shape[0])
    Wjk = randomweights(L2.shape[0], data.shape[0])

    B1 = np.ones((Wij.shape[1], 1))
    B2 = np.ones((Wjk.shape[1], 1))
    E2 = np.zeros(L2.shape)
    E3 = np.zeros(L3.shape)

    itr = 0
    while(itr <= 10000):
        for col in range(len(data)):
            # Compute L1, L2 and, L3
            L1 = data[:, col:col + 1]

            for rowj in range(L2.shape[0]):
                sumwij = float(0)
                for rowi in range(Wij.shape[0]):
                    sumwij = sumwij + Wij[rowi][rowj] * L1[rowi][0]
                L2[rowj][0] = sigmoid(sumwij + B1[rowj][0])

            for rowk in range(L3.shape[0]):
                sumwjk = float(0)
                for rowj in range(Wjk.shape[0]):
                    sumwjk = sumwjk + Wjk[rowj][rowk] * L2[rowj][0]
                L3[rowk][0] = sigmoid(sumwjk + B2[rowk][0])

            for rowk in range(L3.shape[0]):
                E3[rowk][0] = L3[rowk][0] * (1 - L3[rowk][0]) * (data[rowk][col] - L3[rowk][0])

            for rowj in range(L2.shape[0]):
                sumwjk = float(0)
                for rowk in range(L3.shape[0]):
                    sumwjk = sumwjk + E3[rowk][0] * Wjk[rowj][rowk]
                E2[rowj][0] = L2[rowj][0] * (1 - L2[rowj][0]) * sumwjk

            for i in range(Wij.shape[0]):
                for j in range(Wij.shape[1]):
                    Wij[i][j] = Wij[i][j] + LR * E2[j][0] * L1[i][0]

            for j in range(Wjk.shape[0]):
                for k in range(Wjk.shape[1]):
                    Wjk[j][k] = Wjk[j][k] + LR * E3[k][0] * L2[j][0]

            for j in range(B1.shape[0]):
                B1[j][0] = B1[j][0] + LR * E2[j][0]

            for k in range(B2.shape[0]):
                B2[k][0] = B2[k][0] + LR * E3[k][0]

        # Increment itr
        itr = itr + 1
    np.set_printoptions(suppress=True)
    Hidden = sigmoidAll(data.dot(Wij))
    Hidden2 = getHidden(data, Wij)
    # print(Hidden)
    # print(Hidden2)
    Final = sigmoidAll(Hidden.dot(Wjk))
    Final2 = getFinal(Hidden2, Wjk)
    print(Final)
    # print(Final2)
    # np.savetxt('Hidden.txt', Hidden2, fmt='%10.2f')
    # np.savetxt('Output.txt', Final2, fmt='%10.2f')

genweights()
