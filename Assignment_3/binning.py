

def boundaries(minval, maxval, bincount):
    blist = list()
    blist.append(minval)
    rangeval = maxval - minval

    for itr in range(1, bincount):
        blist.append(rangeval * float(itr / bincount))
    blist.append(maxval)

    return blist


def bindata(data, bincount, labels):
    holder = list()

    # Iterate over columns of data
    for idx in range(data.shape[1]):
        spamlist = list()
        hamlist = list()

        col = data[:, idx]
        ziplist = zip(col, labels)
        zipped = sorted(ziplist, key=lambda t: t[0])

        minval = zipped[0][0]
        maxval = zipped[len(zipped) - 1][0]

        boundary = boundaries(minval, maxval, bincount)
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
        if (len(spamlist) != bincount - 1):
            for itr in range((bincount - 1) - len(spamlist)):
                spamlist.append(0)
                hamlist.append(0)
        # Add the last bin values
        spamlist.append(spam)
        hamlist.append(ham)
        holder.append((spamlist, hamlist, boundary))
    return holder


def main():
    pass


if __name__ == '__main__':
    main()
