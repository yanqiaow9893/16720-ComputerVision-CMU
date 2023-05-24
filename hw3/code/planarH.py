import numpy as np
import cv2

def computeH(x1, x2):
    # x1 and x2 are NX2 matrices containing the coordinates (x,y) of point pairs
    # Q2.2.1

    # Construct matrix A
    N = x1.shape[0]
    A = np.zeros((2*N, 9))
    i = 0
    n = 0
    while i < 2*N:
        x1i = x1[n][0]
        y1i = x1[n][1]
        x2i = x2[n][0]
        y2i = x2[n][1]

        A[i] = np.array([-x2i, -y2i, -1, 0, 0, 0, x2i*x1i, y2i*x1i, x1i])
        A[i+1] = np.array([0, 0, 0, -x2i, -y2i, -1, x2i*y1i, y2i*y1i, y1i])

        i += 2
        n += 1

    # Compute the homography between two sets of points
    U, Sig, Vh = np.linalg.svd(A)
    # get the index of the smallest eigenvalue
    h = Vh[-1, :]

    H2to1 = h.reshape((3,3))

    return H2to1


def computeH_norm(x1, x2):
    # Q2.2.2

    # Compute the centroid of the points
    # calculate the average of 4 coordinates
    x1x_avg = np.mean(x1[:, 0])
    x1y_avg = np.mean(x1[:, 1])
    x2x_avg = np.mean(x2[:, 0])
    x2y_avg = np.mean(x2[:, 1])

    # Shift the origin of the points to the centroid
    s1 = np.zeros((x1.shape[0]))
    s2 = np.zeros((x2.shape[0]))
    for i in range(x1.shape[0]):
        s1[i] = np.sqrt((x1[i, 0] - x1x_avg)**2 + (x1[i, 1] - x1y_avg)**2)
        s2[i] = np.sqrt((x2[i, 0] - x2x_avg) ** 2 + (x2[i, 1] - x2y_avg) ** 2)

    # Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    scale1 = np.sqrt(2) / np.max(s1)
    scale2 = np.sqrt(2) / np.max(s2)

    # Similarity transform 1
    T1 = np.array([[scale1, 0, -scale1 * x1x_avg], [0, scale1, -scale1 * x1y_avg], [0, 0, 1]])
    x1_homo = np.hstack((x1, np.ones((x1.shape[0], 1))))
    x1_homo = T1 @ x1_homo.T
    x1_norm = x1_homo.T[:, 0:2]

    # Similarity transform 2
    T2 = np.array([[scale2, 0, -scale2 * x2x_avg], [0, scale2, -scale2 * x2y_avg], [0, 0, 1]])
    x2_homo = np.hstack((x2, np.ones((x2.shape[0], 1))))
    x2_homo = T2 @ x2_homo.T
    x2_norm = x2_homo.T[:, 0:2]

    # Compute homography
    H_norm = computeH(x1_norm, x2_norm)

    # Denormalization
    H2to1 = np.linalg.inv(T1) @ H_norm @ T2

    return H2to1


def computeH_ransac(locs1, locs2, opts):
    # Q2.2.3
    # Compute the best fitting homography given a list of matching points
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier

    # locs1, locs2 are Nx2 matrices containing the matched points
    num_pts = locs1.shape[0] #2

    # Run RANSAC for max_iters iteration
    # bestH2to1 is homography H with most inliers found for during RANSAC
    bestH2to1 = None

    best_num_inliers = 0

    # inliers is a vector of length N with a 1 at those matches, 0 elsewhere
    inliers = np.zeros(num_pts)

    x1_homo = np.hstack((locs1, np.ones((locs1.shape[0], 1))))
    x2_homo = np.hstack((locs2, np.ones((locs2.shape[0], 1))))

    for i in range(max_iters):
        # choose 4 random points
        indices = np.random.choice(num_pts, 4, replace=False)
        x1_rand = locs1[indices]
        x2_rand = locs2[indices]
        # print("x1: ", x1_rand)
        # print("x2: ", x2_rand)

        # compute homography H using these points
        H2to1 = computeH_norm(x1_rand, x2_rand)

        # Apply H to all points in locs2
        proj_locs2 = H2to1 @ x2_homo.T
        proj_locs2 = proj_locs2 / proj_locs2[2, :]

        # compute distances between projected points and locs1
        distances = np.sqrt(np.sum((x1_homo.T - proj_locs2) ** 2, axis=0))

        # Count inliers
        num_inliers = np.sum(distances < inlier_tol)

        if num_inliers > best_num_inliers:
            bestH2to1 = H2to1
            best_num_inliers = num_inliers
            inliers = distances < inlier_tol

    return bestH2to1, inliers


def compositeH(H2to1, template, img):

    # Create a composite image after warping the template image on top
    # of the image using the homography

    # Note that the homography we compute is from the image to the template;
    # x_template = H2to1*x_photo
    # For warping the template to the image, we need to invert it.

    # Create mask of same size as template
    mask = np.ones(template.shape)

    # Warp mask by appropriate homography
    H2to1 = np.linalg.inv(H2to1)
    warp_mask = cv2.warpPerspective(cv2.transpose(mask), H2to1, (img.shape[0], img.shape[1]))
    warp_mask = cv2.transpose(warp_mask)

    # Warp template by appropriate homography
    warp_template = cv2.warpPerspective(cv2.transpose(template), H2to1, (img.shape[0], img.shape[1]))
    warp_template = cv2.transpose(warp_template)

    # Use mask to combine the warped template and the image
    index = np.nonzero(warp_mask)
    img[index] = warp_template[index]

    # make the color back to rgb
    composite_img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2RGB)

    return composite_img


