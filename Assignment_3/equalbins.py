def binning(data, bincount):
    holder = list()
    # Get idex value at multiples we have to break
    if (len(data) % bincount == 0):
        size = int(len(data) / bincount)
    else:
        size = int(len(data) / bincount) +1

    # Iterate over columns of data
    for idx in range(data.shape[1]):
        col = data[:, idx]
        col.sort()
        # np.savetxt('col.txt', col)
        count = list()
        margin = list()
        margin.append(col[0])
        counter = 0

        # Iterate over over sorted colum
        for itr in range(1, len(col)):
            counter += 1
            if (itr % size == 0):
                count.append(counter)
                margin.append(col[itr])
                counter = 0
        # Add the last
        count.append(counter)
        margin.append(col[itr])

        holder.append((count, margin))
        # print(holder)
        # exit()
    return holder


def predictBinning(data, bucket, labels):
    (py0, py1) = getPrior(labels)
    totlen = len(labels)
    predictions = list()

    for idx in range(data.shape[0]):
        row = data[idx, :]
        prowy0 = py0
        prowy1 = py1
        # Iterate over each row value
        for itr in range(len(row)):
            pnum = 0
            val = row[itr]
            tup = buckets[itr]
            # print(tup[0])
            # print(tup[1])
            for rownum in range(len(tup[0])):
                if val >= tup[1][rownum] and val < tup[1][rownum + 1]:
                    pnum = tup[0][rownum]
                    break
                # Now calculate probability by laplace smoothing

            prob = (1 + pnum) / float(4 + totlen)
            prowy0 = prowy0 * prob
            prowy1 = prowy1 * prob

            print(str(prowy0) + " " + str(prowy1))

        # Compare probabilities
        if (prowy0 > prowy1):
            predictions.append(0)
        else:
            predictions.append(1)
    return predictions
