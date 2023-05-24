import matplotlib
import numpy as np
import cv2
import scipy.ndimage
from matplotlib import pyplot as plt

from helper import plotMatches
from matchPics import matchPics
from opts import get_opts

# Q2.1.6


def rotTest(opts):
    # Read the image and convert to grayscale, if necessary
    img = cv2.imread('../data/cv_cover.jpg')
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hist = []
    rotate = []
    for i in range(36):

        # Rotate Image: rotate image itself in increments of 10 degrees
        if i == 0:
            rotate_img = img

        rotate_img = scipy.ndimage.rotate(rotate_img, 10, reshape =False)
        # Compute features, descriptors and Match features
        matches, locs1, locs2 = matchPics(img, rotate_img, opts)
        # Update histogram: stores a histogram of the count of matches for each orientation
        # print(len(matches))
        hist.append(len(matches))
        rotate.append(10*(i+1))

        # plot from three different angles:
        if i == 5 or i == 11 or i == 24:
            plotMatches(img, rotate_img, matches, locs1, locs2)

    # Display histogram
    plt.bar(rotate, hist, width=1.0, color='blue')
    #matplotlib.pyplot.hist(hist, bins=36)
    plt.xlabel("rotation")
    plt.ylabel("number of matches")
    plt.show()


if __name__ == "__main__":

    opts = get_opts()
    rotTest(opts)
