import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
import numpy
numpy.set_printoptions(threshold=numpy.inf)


def load_mnist(dataset="training", digits=numpy.arange(10), path="."):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels


def printIt(images, labels, num):
    idx = num - 1
    print(images[idx])
    print(sum(sum(images[idx])))
    print(labels[idx])
    print()


if __name__ == '__main__':
    images, labels = load_mnist(dataset="training", path="MINST")
    images /= 128.0

    printIt(images, labels, 4)
    printIt(images, labels, 7)
    printIt(images, labels, 9)
    printIt(images, labels, 15)
    printIt(images, labels, 24)
    printIt(images, labels, 25)
