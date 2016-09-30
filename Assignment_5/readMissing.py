import numpy


def make(full):
    data = full[:, range(full.shape[1] - 1)]

    labels = full[:, full.shape[1] - 1]

    return (data, labels)


def read(name):
    full = numpy.loadtxt(name, delimiter=",", dtype=float)
    return make(full)


if __name__ == '__main__':
    pass
