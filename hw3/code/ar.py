import numpy as np
import cv2

#Import necessary functions
from loadVid import loadVid
from helper import loadVid
from matplotlib import pyplot as plt
from opts import get_opts
from matchPics import matchPics
from planarH import computeH_ransac, compositeH

# Write script for Q3.1
def mymatchImg(opts, coverImg, book_frame, ar_frame):

    # matched the points between the cover of book image and the book image that is in video
    matches, locs1, locs2 = matchPics(coverImg, book_frame, opts)

    # the coordinate of locs returned from matchPics is (y,x)
    pair1 = locs1[matches[:, 0]]
    pair2 = locs2[matches[:, 1]]

    # compute homography using lImg and rImg
    bestH2to1, inliers = computeH_ransac(pair1, pair2, opts)

    # remove the top and bottom black space
    ar_frame = ar_frame[44:315, :, :]

    # crop the width of ar_frame
    croped_width = int(coverImg.shape[1] / coverImg.shape[0] * ar_frame.shape[0])
    disregard_width = int((ar_frame.shape[1] - croped_width)/2)
    ar_frame = ar_frame[:, 0+disregard_width:disregard_width+croped_width, :]

    # resize the ar_frame to be the same as cover
    ar_resize = cv2.resize(ar_frame, (coverImg.shape[1], coverImg.shape[0]))

    composite_img = compositeH(bestH2to1, ar_resize, book_frame)

    # plt.imshow(composite_img)
    # plt.show()

    return composite_img


def ar(opts):
    ar_vid = loadVid('../data/ar_source.mov')
    book_vid = loadVid('../data/book.mov')
    coverImg = cv2.imread('../data/cv_cover.jpg')

    f_book, h_book, w_book, _ = book_vid.shape
    f_ar, _, _, _ = ar_vid.shape

    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('../result/ar.avi', fourcc, fps, (w_book, h_book))

    for i in range(f_ar):
        if i <= 27 or i >= 157:
            print(i)
            b_frame = book_vid[i]
            ar_frame = ar_vid[i]
            composite_img = mymatchImg(opts, coverImg, b_frame, ar_frame)
            out.write(composite_img)

    cv2.destroyAllWindows()
    out.release()


if __name__ == "__main__":

    opts = get_opts()
    ar(opts)

