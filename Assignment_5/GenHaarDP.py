import numpy as np
from random import randint

from ReadMINST import load_mnist


class rectangle(object):

    def __init__(self, left=None, right=None, top=None, bot=None):
        self.left = left
        self.right = right
        self.top = top
        self.bot = bot

    def __repr__(self):
        return str(str(self.left) + " " + str(self.right) + " " + str(self.top) + " " + str(self.bot))


def genRectangles(length, count):
    mylist = list()

    # Compute the half and full length
    half = int(length / 2)
    full = length - 1
    while(len(mylist) < 100):

        # Generate left and right
        left = randint(0, half)
        right = randint(left + 5, full)

        top = randint(0, half)
        bottom = randint(top + 5, full)

        product = (right - left) * (bottom - top)

        if (product > 130 and product < 170):
            mylist.append(rectangle(left, right, top, bottom))
            # print("L " + str(left) + " R " + str(right) + " T " + str(top) + " B " + str(bottom))
    return mylist


def genfeatures(images, rectangles, name):
    features = np.zeros((images.shape[0], 2 * len(rectangles)))

    for icount in range(images.shape[0]):
        img = images[icount]
        final = np.empty_like(img, dtype=int)

        paintBlack(img, final)
    
        # print(img)
        # print(np.sum(img))

        # np.savetxt("final.txt", final, fmt='%1.0f')
        # np.savetxt("seven.txt", img, fmt='%1.0f')

        for irec in range(len(rectangles)):
            rec = rectangles[irec]
            vmid = int((rec.bot - rec.top) / 2) + rec.top
            hmid = int((rec.right - rec.left) / 2) + rec.left
            #print("Hmid " + str(hmid) + " vmid " + str(vmid))

            up = final[vmid][rec.right] - final[rec. top][rec.right] - final[vmid][rec.left] + final[rec.top][rec.left]
            down = final[rec.bot][rec.right] - final[vmid][rec.right] - final[rec.bot][rec.left] + final[vmid][rec.left]

            left = final[rec.bot][hmid] - final[rec.top][hmid] - final[rec.bot][rec.left] + final[rec.top][rec.left]
            right = final[rec.bot][rec.right] - final[rec.top][rec.right] - final[rec.bot][hmid] + final[rec.top][hmid]

            first = left - right
            second = up - down

            
            # print("DP")
            # print("L "+ str(rec.left) + " R " + str(rec.right) + " U " + str(rec.top) + " D " + str(rec.bot))
            # print("L " + str(left) + " R " + str(right) + " U " + str(up) + " D " + str(down) + " first " + str(first) + " second " + str(second))
            '''
            # left = int(np.sum(img[rec.left : vmid, rec.top : rec.bot]))
            # right = int(np.sum(img[vmid : rec.right, rec.top : rec.bot]))

            # up = int(np.sum(img[rec.left : rec.right, rec.top : hmid]))
            # down = int(np.sum(img[rec.left : rec.right, hmid : rec.bot]))

            # print("Numpy")
            # print(" L " + str(left) + " R " + str(right) + " U " + str(up) + " D " + str(down))
            # exit()
            '''

            # Allocate the first and second as features
            features[icount][irec * 2] = first
            features[icount][irec * 2 + 1] = second
        # np.savetxt(name, final, fmt='%1.0f')
        # exit()
        print(icount)
    # Write the final 2d feature matrix as txt
    np.savetxt(name, features, fmt='%1.0f')


def paintBlack(rec, final):
    for row in range(rec.shape[0]):
        for col in range(rec.shape[1]):
            final[row][col] = rec[row][col] + getBlack(final, row, col)


def getBlack(final, row, col):
    if (row == 0 and col == 0):
        return 0
    elif (row == 0):
        return final[row][col - 1]
    elif (col == 0):
        return final[row - 1][col]
    else:
        return final[row - 1][col] + final[row][col - 1] - final[row - 1][col - 1]


if __name__ == '__main__':
    images, labels = load_mnist(dataset="training", path="MINST")
    images /= 128.0
    rectangles = genRectangles(28, 100)
    genfeatures(images, rectangles, "htrain.txt")
    np.savetxt("htrainlabels.txt", labels, fmt='%1.0f')

    images, labels = load_mnist(dataset="testing", path="MINST")
    images /= 128.0
    # rectangles = genRectangles(28, 100)
    genfeatures(images, rectangles, "htest.txt")
    np.savetxt("htestlabels.txt", labels, fmt='%1.0f')
