import numpy as np
import cv2
import skimage.io 
import skimage.color
from matplotlib import pyplot as plt

from opts import get_opts

# Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac, compositeH


# Q2.2.4

def warpImage(opts):
    # Read 3 images in:
    img1 = cv2.imread('../data/cv_cover.jpg')
    img2 = cv2.imread('../data/cv_desk.png')
    img3 = cv2.imread('../data/hp_cover.jpg')

    # compute a homography automatically using matchPics and computeH_ransac
    # compute H using cv_cover and cv_desk
    matches, locs1, locs2 = matchPics(img1, img2, opts)
    # the coordinate of locs returned from matchPics is (y,x)
    pair1 = locs1[matches[:, 0]]
    pair2 = locs2[matches[:, 1]]

    bestH2to1, inliers = computeH_ransac(pair1, pair2, opts)

    # Use computed H to warp hp_cover to cv_desk
    # h, w = img2.shape[:2]
    #
    # hp_cover_warped = cv2.warpPerspective(img3, bestH2to1, (w, h))
    # cv2.imshow('Warped HP cover', hp_cover_warped)
    # cv2.waitKey(0)

    hp_resize = cv2.resize(img3, (img1.shape[1], img1.shape[0]))
    composite_img = compositeH(bestH2to1, hp_resize, img2)

    plt.imshow(composite_img)
    plt.show()


if __name__ == "__main__":

    opts = get_opts()
    warpImage(opts)


