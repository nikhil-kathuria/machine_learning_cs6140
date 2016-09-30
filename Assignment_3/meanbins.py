import numpy as np


def meanboundary(col, col0, col1, bincount):
    boundary = list()
    boundary.append(min(col))
    boundary.append(np.mean(col1))
    boundary.append(np.mean(col))
    boundary.append(np.mean(col0))
    boundary.append(max(col))

    if (bincount != 4):
        boundary = sorted(boundary)
        new = list()
        for itr in range(1, len(boundary)):
            new.append((boundary[itr -1] + boundary[itr]) / 2)
        boundary = boundary + new
    return sorted(boundary)


def boundaries(minval, maxval, bincount):
    blist = list()
    blist.append(minval)
    rangeval = maxval - minval

    for itr in range(1, bincount):
        blist.append(rangeval * float(itr / bincount))
    blist.append(maxval)

    return blist


def bindata(data, mat0, mat1, bincount, labels):
    holder = list()

    # Iterate over columns of data
    for idx in range(data.shape[1]):
        spamlist = list()
        hamlist = list()

        col = data[:, idx]
        col0 = mat0[:, idx]
        col1 = mat1[:, idx]
        ziplist = zip(col, labels)
        zipped = sorted(ziplist, key=lambda t: t[0])

        # minval = zipped[0][0]
        # maxval = zipped[len(zipped) - 1][0]

        boundary = meanboundary(col, col0, col1, bincount)
        # print(len(boundary))
        spam = 0
        ham = 0
        bin = 1
        for itr in range(len(zipped)):
            if zipped[itr][0] < boundary[bin]:
                if zipped[itr][1] == 0:
                    spam += 1
                else:
                    ham += 1
            elif zipped[itr][0] > boundary[bin]:
                spamlist.append(spam)
                hamlist.append(ham)
                spam = 0
                ham = 0
                if zipped[itr][1] == 0:
                    spam += 1
                else:
                    ham += 1
                #Increase the bin value
                bin += 1
        # Add zero if not last - 1 index is reached.
        '''if (len(spamlist) != bincount - 1):
            for itr in range((bincount - 1) - len(spamlist)):
                spamlist.append(0)
                hamlist.append(0)'''
        # Add the last bin values
        spamlist.append(spam)
        hamlist.append(ham)
        # print(spamlist)
        # print(hamlist)
        # print(boundary)
        holder.append((spamlist, hamlist, boundary))
    return holder


def main():
    pass


if __name__ == '__main__':
    main()
