import numpy as np

import math


def load(name):
    full = np.loadtxt(name, dtype=float)
    return full


def init2M(data, p1, p2):
    range1 = range(p1)
    range2 = range(p1, p2)
    range3 = range(p2, len(data))

    mat1 = data[range1, :]
    mat2 = data[range2, :]
    mat3 = data[range3, :]

    U1 = np.mean(mat1, axis=0)
    U2 = np.mean(mat2, axis=0)
    U3 = np.mean(mat3, axis=0)

    C1 = np.cov(mat1.T)
    C2 = np.cov(mat2.T)
    C3 = np.cov(mat3.T)

    Pi1 = len(range1) / float(len(data))
    Pi2 = len(range2) / float(len(data))
    Pi3 = len(range3) / float(len(data))

    M1 = (U1, C1, Pi1)
    M2 = (U2, C2, Pi2)
    M3 = (U3, C3, Pi3)

    return (M1, M2, M3)


def dot(a):
    arr = np.empty((len(a), len(a)))
    for row in range(len(a)):
        for col in range(len(a)):
            arr[row][col] = a[row] * a[col]
    return arr


def pxGivenY(cov, det, inv, diff, nby2):
    par1 = 1 / (math.pow((2 * math.pi), nby2) * math.pow(det, .5))
    part2 = math.exp(-.5 * diff.dot((inv.dot(diff))))
    return par1 * part2


def computeParam(data, Z, U):
    cov = np.zeros((data.shape[1], data.shape[1]))
    Unew = np.zeros(data.shape[1])
    den = float(0)
    Pi = float(0)

    for idx in range(data.shape[0]):
        row = data[idx, :]
        diff = (row - U)

        cov = cov + Z[idx] * dot(diff)

        Unew = Unew + Z[idx] * (row)

        den = den + Z[idx]

        Pi = Pi + Z[idx]

    cov = cov / den
    Unew = Unew / den
    Pi = Pi / len(Z)

    return (Unew, cov, Pi)


def computeE(data, M1, M2, M3):
    nby2 = float(data.shape[1]) / 2
    Z1 = list()
    Z2 = list()
    Z3 = list()
    (U1, C1, P1) = M1
    (U2, C2, P2) = M2
    (U3, C3, P3) = M3

    # print(C1)
    # print(C2)
    I1 = np.linalg.pinv(C1)
    I2 = np.linalg.pinv(C2)
    I3 = np.linalg.pinv(C3)

    D1 = np.linalg.det(C1)
    D2 = np.linalg.det(C2)
    D3 = np.linalg.det(C3)

    for idx in range(data.shape[0]):
        row = data[idx, :]
        diff1 = row - U1
        diff2 = row - U2
        diff3 = row - U3

        pm1 = pxGivenY(C1, D1, I1, diff1, nby2)
        pm2 = pxGivenY(C2, D2, I2, diff2, nby2)
        pm3 = pxGivenY(C3, D3, I3, diff3, nby2)

        z1 = (pm1 * P1) / ((pm2 * P2) + (pm1 * P1) + (pm3 * P3))
        z2 = (pm2 * P2) / ((pm2 * P2) + (pm1 * P1) + (pm3 * P3))
        z3 = (pm3 * P3) / ((pm2 * P2) + (pm1 * P1) + (pm3 * P3))

        Z1.append(z1)
        Z2.append(z2)
        Z3.append(z3)

    return(Z1, Z2, Z3)


def computeM(data, Zim, U1, U2, U3):
    Z1 = Zim[0]
    Z2 = Zim[1]
    Z3 = Zim[2]

    M1 = computeParam(data, Z1, U1)
    M2 = computeParam(data, Z2, U2)
    M3 = computeParam(data, Z3, U3)

    return (M1, M2, M3)


def printM(Model):
    for itr in range(len(Model)):
        print(Model[itr])
    print()


def main():
    mat = load("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_3/3gaussian.txt")
    (M1, M2, M3) = init2M(mat, 2000, 5000)
    counter = 0
    while(counter < 110):
        counter += 1
        Zim = computeE(mat, M1, M2, M3)
        (M1, M2, M3) = computeM(mat, Zim, M1[0], M2[0], M3[0])

    printM(M1)
    printM(M2)
    printM(M3)

if __name__ == '__main__':
    main()
