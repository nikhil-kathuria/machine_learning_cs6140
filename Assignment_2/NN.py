import numpy as np
import math

LR = float(3)


def sigmoid(value):
    return 1.0 / (1.0 + math.exp(-value))


def sigmoidVector(data):
    for row in range(data.shape[0]):
        data[row][0] = sigmoid(data[row][0])
    return data


def sigmoidAll(data):
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            data[row][col] = sigmoid(data[row][col])
    return data


def randomweights(row, col):
    mat = np.random.rand(row, col)
    mat = mat - float(0.5)
    return mat


def updateweights(Wmat, Err, lr):
    for row in range(Wmat.shape[0]):
        for col in range(Wmat.shape[1]):
            Wmat[row][col] = Wmat[row][col] + LR * Err[col] * lr[row]


def genweights():
    data = np.eye(8, dtype=float)
    L2 = np.zeros(3, dtype=float)

    Wij = randomweights(data.shape[0], L2.shape[0])
    Wjk = randomweights(L2.shape[0], data.shape[0])

    B2 = np.ones((Wij.shape[1], 1))
    B3 = np.ones((Wjk.shape[1], 1))

    out = np.zeros((8, 8))
    hid = np.zeros((3, 8))

    itr = 0
    while(itr <= 500):
        for col in range(len(data)):
            L1 = data[:, col:col + 1]

            L2 = sigmoidVector(np.transpose(Wij).dot(L1) + B2)
            L3 = sigmoidVector(np.transpose(Wjk).dot(L2) + B3)

            out[:, col:col + 1] = sigmoidVector(np.transpose(Wjk).dot(L2) + B3)
            hid[:, col:col + 1] = sigmoidVector(np.transpose(Wij).dot(L1) + B2)

            # Compute E3, E2
            E3 = np.multiply(np.multiply(L3, (1 - L3)), (L1 - L3))
            E2 = np.multiply(np.multiply(L2, (1 - L2)), Wjk.dot(E3))

            # Update Weights
            updateweights(Wij, E2, L1)
            updateweights(Wjk, E3, L2)

            # Update B1 and B2
            B2 = B2 + LR * E2
            B3 = B3 + LR * E3

        itr = itr + 1
    np.savetxt('Hidden.txt', np.transpose(hid), fmt='%10.2f')
    np.savetxt('Output.txt', out, fmt='%10.2f')


def main():
    genweights()


if __name__ == '__main__':
    main()
