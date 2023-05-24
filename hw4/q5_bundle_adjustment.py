import math

import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous
from q2_1_eightpoint import eightpoint
from q2_2_sevenpoint import sevenpoint
from q3_2_triangulate import findM2, triangulate

import scipy

# Insert your package here


# Helper functions for this assignment. DO NOT MODIFY!!!
"""
Helper functions.

Written by Chen Kong, 2018.
Modified by Zhengyi (Zen) Luo, 2021
"""
def plot_3D_dual(P_before, P_after):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Blue: before; red: after")
    ax.scatter(P_before[:,0], P_before[:,1], P_before[:,2], c = 'blue')
    ax.scatter(P_after[:,0], P_after[:,1], P_after[:,2], c='red')
    while True:
        x, y = plt.ginput(1, mouse_stop=2)[0]
        plt.draw()


'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
            nIters, Number of iterations of the Ransac
            tol, tolerence for inliers
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers

    Hints:
    (1) You can use the calc_epi_error from q1 with threshold to calcualte inliers. Tune the threshold based on 
        the results/expected number of inliners. You can also define your own metric. 
    (2) Use the seven point alogrithm to estimate the fundamental matrix as done in q1
    (3) Choose the resulting F that has the most number of inliers
    (4) You can increase the nIters to bigger/smaller values
 
'''
def ransacF(pts1, pts2, M, nIters=1000, tol=1):
    N = pts1.shape[0]

    pts1_homo = np.vstack((pts1.T, np.ones((1, N))))
    pts2_homo = np.vstack((pts2.T, np.ones((1, N))))

    # inliers is a vector of length N with a 1 at those matches, 0 elsewhere
    inliers = np.zeros(N)
    best_num_inliers = 0

    # Use the seven point alogrithm to estimate the fundamental matrix as done in q1
    for i in range(nIters):
        rand_idx = np.random.choice(pts1.shape[0], 7, replace=False)
        Farray = sevenpoint(pts1[rand_idx, :], pts2[rand_idx, :], M)

        # Choose the resulting F that has the most number of inliers
        for f in Farray:
            # Calculate epiploar line for pts1
            epi_line = f @ pts1_homo

            # calculate the distance from each point in pts2 to the corresponding epipolar line
            distance = abs(np.sum(epi_line * pts2_homo, axis=0)) / np.sqrt(epi_line[0,:] ** 2 + epi_line[1,:] ** 2)

            # Count inliers
            num_inliers = np.sum(distance < tol)

            if num_inliers > best_num_inliers:
                best_num_inliers = num_inliers
                inliers = distance < tol
                F = f

    return F, inliers


'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    theta = np.linalg.norm(r)
    if theta == 0:
        return np.eye(3)

    u = r / theta
    uut = np.outer(u, u.T)
    ux = np.array([[0, -u[2], u[1]],
                   [u[2], 0, -u[0]],
                   [-u[1], u[0], 0]])
    # R = I cos(theta) + (1-cos(theta))uu.T + ux * sin(theta)
    R = np.eye(3) * np.cos(theta) + (1 - np.cos(theta)) * uut + ux * np.sin(theta)

    return R


'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    A = (R - R.T) / 2

    a32 = A[2, 1]
    a13 = A[0, 2]
    a21 = A[1, 0]

    rh = np.vstack((a32, a13, a21))
    s = np.linalg.norm(rh)
    r11 = R[0, 0]
    r22 = R[1, 1]
    r33 = R[2, 2]

    c = (r11 + r22 + r33 - 1) / 2

    if s == 0. and c == 1.:
        # vec_r = 0
        r = np.zeros((3, 1))

    elif s == 0. and c == -1.:
        # let v = a nonzero column of R+I
        v_tmp = R + np.eye(3)
        for i in range(3):
            if np.sum(v_tmp[:, i]) != 0:
                v = v_tmp[:, i]
                break

        u = v / np.linalg.norm(v)
        func = u * np.pi
        if np.linalg.norm(func) == np.pi and ((func[0, 0] == 0 and func[1, 0] == 0 and func[2, 0] < 0)
                                              or (func[0, 0] == 0 and func[1, 0] < 0)
                                              or (func[0, 0] < 0)):
            r = -func
        else:
            r = func

    else:
        u = rh / s
        theta = np.arctan2(float(s), float(c))
        r = u * theta

    return r.flatten()


'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    P = x[:-6]
    r2 = x[-6: -3]
    t2 = x[-3:]

    N = math.floor(P.shape[0]/3)

    P = np.reshape(P, (N, 3))
    P_homo = np.vstack((P.T, np.ones((1, P.shape[0]))))

    # r2 (in the Rodrigues vector form)
    R2 = rodrigues(r2)

    # r2 and t2 are associated with the projection matrix M2: 3x4
    t2 = np.reshape(t2, (3, 1))
    M2 = np.hstack((R2, t2))

    C1 = K1 @ M1
    C2 = K2 @ M2

    x1_homo = C1 @ P_homo
    x2_homo = C2 @ P_homo

    p1_hat = (x1_homo[:2, :] / x1_homo[2, :]).T
    p2_hat = (x2_homo[:2, :] / x2_homo[2, :]).T

    # Residuals are the difference between original image projections and estimated projections.
    residuals = np.concatenate([(p1 - p1_hat).reshape([-1]), (p2 - p2_hat).reshape([-1])])

    return residuals


'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
            o1, the starting objective function value with the initial input
            o2, the ending objective function value after bundle adjustment

    Hints:
    (1) Use the scipy.optimize.minimize function to minimize the objective function, rodriguesResidual. 
        You can try different (method='..') in scipy.optimize.minimize for best results. 
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    
    obj_start = obj_end = 0

    # extract rotation and translation from M2_init
    R2_init = M2_init[:, :3]
    t2_init = M2_init[:, 3]

    # use invR to transform the rotation
    r2_init = invRodrigues(R2_init).reshape([-1])

    def fun(x): return (rodriguesResidual(K1, M1, p1, K2, p2, x))

    p = P_init.flatten()
    r2 = r2_init.flatten()
    t2 = t2_init.flatten()

    # concatenate x with translation and 3D points
    x0 = []
    x0 = np.append(x0, p)
    x0 = np.append(x0, r2)
    x0 = np.append(x0, t2)

    # optimzie for the best extrinsic matrix and 3D points
    x_opt, _ = scipy.optimize.leastsq(fun, x0)

    # decompose back to rotation, translation and 3D point
    P2 = x_opt[:-6]
    N = math.floor(P2.shape[0] / 3)
    P = np.reshape(P2, (N, 3))

    r2 = x_opt[-6:-3]
    t2 = x_opt[-3:]
    t2 = np.reshape(t2, (3, 1))

    R2 = rodrigues(r2)
    M2 = np.hstack((R2, t2))

    return M2, P, obj_start, obj_end



if __name__ == "__main__":
              
    # np.random.seed(1) #Added for testing, can be commented out
    #
    # some_corresp_noisy = np.load('data/some_corresp_noisy.npz') # Loading correspondences
    # intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    # K1, K2 = intrinsics['K1'], intrinsics['K2']
    # noisy_pts1, noisy_pts2 = some_corresp_noisy['pts1'], some_corresp_noisy['pts2']
    # im1 = plt.imread('data/im1.png')
    # im2 = plt.imread('data/im2.png')

    # F, inliers = ransacF(noisy_pts1, noisy_pts2, M=np.max([*im1.shape, *im2.shape]))
    #uncomment for q5.1
    #
    # M = np.max([*im1.shape, *im2.shape])
    # F = eightpoint(noisy_pts1, noisy_pts2, M=M)
    # displayEpipolarF(im1, im2, F)
    #
    # # Simple Tests to verify your implementation:
    # pts1_homogenous, pts2_homogenous = toHomogenous(noisy_pts1), toHomogenous(noisy_pts2)
    #
    # assert(F.shape == (3, 3))
    # assert(F[2, 2] == 1)
    # assert(np.linalg.matrix_rank(F) == 2)

    # Simple Tests to verify your implementation: Q 5.2
    # from scipy.spatial.transform import Rotation as sRot
    # rotVec = sRot.random()
    # r = rotVec.as_rotvec()
    # mat = rodrigues(r) # mat = R
    # r1 = invRodrigues(mat)
    # tmp2 = rotVec.as_matrix()
    # # check if tmp2 = mat & r = r1
    #
    # assert(np.linalg.norm(rotVec.as_rotvec() - invRodrigues(mat)) < 1e-3)
    # assert(np.linalg.norm(rotVec.as_matrix() - mat) < 1e-3)
    #

    # Visualization:
    np.random.seed(1)
    correspondence = np.load('data/some_corresp_noisy.npz') # Loading noisy correspondences
    intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')
    M=np.max([*im1.shape, *im2.shape])

    # TODO: YOUR CODE HERE
    '''
    Call the ransacF function to find the fundamental matrix
    Call the findM2 function to find the extrinsics of the second camera
    Call the bundleAdjustment function to optimize the extrinsics and 3D points
    Plot the 3D points before and after bundle adjustment using the plot_3D_dual function
    '''
    M1 = np.array([[1.0, 0, 0, 0],
                   [0, 1.0, 0, 0],
                   [0, 0, 1.0, 0]])
    # Call the ransacF function to find the fundamental matrix
    F, inliers = ransacF(pts1, pts2, M)

    # Call the findM2 function to find the extrinsics of the second camera
    p1, p2 = correspondence['pts1'][inliers, :], correspondence['pts2'][inliers, :]
    M2_init, C2, P_init = findM2(F, p1, p2, intrinsics)
    #  Call the bundleAdjustment function to optimize the extrinsics and 3D points
    M2, P, _, _ = bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init)

    # new 3d point
    P_new = np.hstack((P, np.ones((P.shape[0], 1))))
    err_proj = 0
    for i in range(p1.shape[0]):
        proj1 = K1 @ M1 @ (P_new[i, :]).T
        proj2 = K2 @ M2 @ (P_new[i, :]).T
        proj1 = np.transpose(proj1[:2] / proj1[-1])
        proj2 = np.transpose(proj2[:2] / proj2[-1])

        # Reprojection error between initial M2 and w with optimized matrices
        err_proj = err_proj + np.sum((proj1 - p1[i])**2 + (proj2 - p2[i])**2)

    print("The error is", err_proj)


    # plot 3D points before and after bundle adjustment
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(P_init[:, 0], P_init[:, 1], P_init[:, 2], c='r', marker='.')
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='b', marker='.')

    plt.show()


