from readData import readHouse, readSpam

def normalize(mat):
    for col in range(mat.shape[1]):
        maxdata = max(mat[:, col])
        mindata = min(mat[:, col])
        if mindata == maxdata:
            mat[:, col] = float(0)
            continue
        # print((mat[:, col] - mindata) / (maxdata - mindata))
        mat[:, col] = (mat[:, col] - mindata) / (maxdata - mindata)


def main():
    (data, labels) = readHouse("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_1/test.txt")
    normalize(data)
    print(data)

if __name__ == '__main__':
    main()
