import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous, refineF, _singularize

# Insert your package here



'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix

    HINTS:
    (1) Normalize the input pts1 and pts2 using the matrix T.
    (2) Setup the eight point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Use the function `_singularize` (provided) to enforce the singularity condition. 
    (5) Use the function `refineF` (provided) to refine the computed fundamental matrix. 
        (Remember to use the normalized points instead of the original points)
    (6) Unscale the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # scale the data by dividing each coordinate by M:
    pts1 = pts1 / M
    pts2 = pts2 / M
    x1 = pts1[:, 0]
    y1 = pts1[:, 1]
    x2 = pts2[:, 0]
    y2 = pts2[:, 1]

    # solve for least square solution
    A = np.vstack((x1*x2, x2*y1, x2, x1*y2, y1*y2, y2, x1, y1, np.ones(x1.shape))).T

    U, S, Vh = np.linalg.svd(A)
    f = Vh[-1, :]
    f = np.reshape(f, (3,3))

    # refine F
    F = refineF(f, pts1, pts2)

    # set the last singular val to be 0
    F = _singularize(F)

    # normalize using matrix T:
    T = np.array([[1.0 / M, 0, 0], [0, 1.0 / M, 0], [0, 0, 1]])

    # scale the data such that F_unnormalized = T.T @ F @T
    F = T.T @ F @ T

    F = F / F[2, 2]

    return F


if __name__ == "__main__":
        
    correspondence = np.load('data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')

    M = np.max([*im1.shape, *im2.shape])
    F = eightpoint(pts1, pts2, M=M)

    # Q2.1
    print(F)
    # displayEpipolarF(im1, im2, F)
    np.savez('q2_1.npz', F=F, M=M)
    print("saved")

    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)


    assert(F.shape == (3, 3))
    assert(F[2, 2] == 1)
    assert(np.linalg.matrix_rank(F) == 2)
    assert(np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1)