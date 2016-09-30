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
    final = np.zeros((images.shape[0], 2 * len(rectangles)))
    print(final.shape)
    for itr in range(images.shape[0]):
        img = images[itr]

        # Now iterate over rectangels
        for rec in range(len(rectangles)):
            myrec = rectangles[rec]
            vmid = int((myrec.bot - myrec.top) / 2) + myrec.top
            hmid = int((myrec.right - myrec.left) / 2) + myrec.left

            left = int(np.sum(img[myrec.top : myrec.bot, myrec.left : hmid]))
            right = int(np.sum(img[myrec.top : myrec.bot, hmid : myrec.right]))

            up = int(np.sum(img[myrec.top : hmid, myrec.left : myrec.right]))
            down = int(np.sum(img[myrec.left : myrec.right, hmid : myrec.bot]))

            first = left - right
            second = up - down

            final[itr, rec * 2] = first
            final[itr, rec * 2 + 1] = second
        # np.savetxt(name, final, fmt='%1.0f')
        # exit()
        print(itr)
    # Write the final 2d feature matrix as txt
    np.savetxt(name, final, fmt='%1.0f')


if __name__ == '__main__':
    images, labels = load_mnist(dataset="training", path="MINST")
    images /= 128.0
    rectangles = genRectangles(28, 100)
    genfeatures(images, rectangles, "train.txt")
    np.savetxt("trainlabels.txt", labels, fmt='%1.0f')

    images, labels = load_mnist(dataset="testing", path="MINST")
    images /= 128.0
    # rectangles = genRectangles(28, 100)
    genfeatures(images, rectangles, "test.txt")
    np.savetxt("testlabels.txt", labels, fmt='%1.0f')
