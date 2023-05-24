import numpy as np
import cv2

# Import necessary functions
from matplotlib import pyplot as plt
from opts import get_opts
from matchPics import matchPics
from planarH import computeH_ransac, compositeH


# Q4
def panaromaImage(opts):
    lImg = cv2.imread('../data/left.jpg')
    rImg = cv2.imread('../data/right.jpg')
    # matched the points
    lh, lw,_ = lImg.shape
    rh, rw,_ = rImg.shape

    pana_resize = cv2.copyMakeBorder(rImg, 0, abs(rh-lh),
                                     int(max(lImg.shape[1], rImg.shape[1]) * 1.1) - rw,
                                     0, cv2.BORDER_CONSTANT)
    matches, locs1, locs2 = matchPics(lImg, pana_resize, opts)
    # the coordinate of locs returned from matchPics is (y,x)
    pair1 = locs1[matches[:, 0]]
    pair2 = locs2[matches[:, 1]]

    # compute homography using lImg and rImg
    bestH2to1, inliers = computeH_ransac(pair1, pair2, opts)

    # Copy the left image onto the panorama image
    pImg = compositeH(bestH2to1, lImg, pana_resize)

    plt.imshow(pImg)
    plt.show()


if __name__ == "__main__":

    opts = get_opts()
    panaromaImage(opts)
