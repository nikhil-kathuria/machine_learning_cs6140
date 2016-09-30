import numpy as np
from Regression import sigmoid, sigmoidAll


LR = float(.2)


def sigmoidVector(data):
    for row in range(data.shape[0]):
        data[row][0] = sigmoid(data[row][0])
    return data


def randomweights(row, col):
    mat = np.random.rand(row, col)
    mat = mat - float(0.5)
    return mat


def updateweights(W, E, O):
    for row in range(W.shape[0]):
        for col in range(W.shape[1]):
            W[row][col] = W[row][col] + LR * E[col] * O[row]
    return W


def genweights():
    data = np.eye(8, dtype=float)

    L2 = np.zeros(3, dtype=float)
    Wij = randomweights(data.shape[0], L2.shape[0])
    Wjk = randomweights(L2.shape[0], data.shape[0])

    B1 = np.ones((Wij.shape[1], 1))
    B2 = np.ones((Wjk.shape[1], 1))

    '''
    print(Wij.shape)
    print(Wjk.shape)
    print(B1.shape)
    print(B2.shape)
    '''

    itr = 0
    while(itr <= 1000):
        for col in range(len(data)):
            # Compute L1, L2 and, L3
            L1 = data[:, col:col + 1]
            # print(L1)
            # print(L1.shape)
            L2 = sigmoidVector(np.transpose(Wij).dot(L1) + B1)
            # print(L2.shape)
            L3 = sigmoidVector(np.transpose(Wjk).dot(L2) + B2)
            # print(L3.shape)

            # Compute E3, E2
            E3 = np.multiply(np.multiply(L3, (1 - L3)), (data[:, col:col + 1] - L3))
            E2 = np.multiply(np.multiply(L2, (1 - L2)), Wjk.dot(E3))

            # print(E3)
            # print(E2)

            # Compute DeltaWij and DeltaWjk
            # print(E2.shape)
            # print(E3.shape)
            Wij = updateweights(Wij, E2, L1)
            Wjk = updateweights(Wjk, E3, L2)

            # DeltaWij = LR * L1.dot(np.transpose(E2))
            # DeltaWjk = LR * L2.dot(np.transpose(E3))

            # Update Wij and Wjk
            # Wij = Wij + DeltaWij
            # Wjk = Wjk + DeltaWjk

            # Calculate DeltaB1 and DeltaB2
            DeltaB1 = LR * E2
            DeltaB2 = LR * E3

            # Update B1 and B2
            B1 = B1 + DeltaB1
            B2 = B2 + DeltaB2

        # exit()

        itr = itr + 1
        # print(Wij)
        # print(Wjk)
    print(B1)
    print(B2)
    np.set_printoptions(suppress=True)
    # print(data)
    # print(Wij)
    Hidden = sigmoidAll(data.dot(Wij))
    # print(Hidden)
    # print(Wjk)
    Output = sigmoidAll(Hidden.dot(Wjk))
    print(Output)
    # np.savetxt('Hidden.txt', Hidden, fmt='%10.2f')
    # np.savetxt('Output.txt', sigmoidAll(np.transpose(Wjk).dot(Hidden)), fmt='%10.2f')

genweights()
