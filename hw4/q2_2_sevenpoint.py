import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous, _singularize

# Insert your package here
from helper import refineF

'''
Q2.2: Seven Point Algorithm for calculating the fundamental matrix
    Input:  pts1, 7x2 Matrix containing the corresponding points from image1
            pts2, 7x2 Matrix containing the corresponding points from image2
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated 3x3 fundamental matrixes.
    
    HINTS:
    (1) Normalize the input pts1 and pts2 scale paramter M.
    (2) Setup the seven point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Pick the last two colum vector of vT.T (the two null space solution f1 and f2)
    (5) Use the singularity constraint to solve for the cubic polynomial equation of  F = a*f1 + (1-a)*f2 that leads to 
        det(F) = 0. Solving this polynomial will give you one or three real solutions of the fundamental matrix. 
        Use np.polynomial.polynomial.polyroots to solve for the roots
    (6) Unscale the fundamental matrixes and return as Farray
'''
def sevenpoint(pts1, pts2, M):

    Farray = []
    # ----- TODO -----
    # YOUR CODE HERE
    # scale the data by dividing each coordinate by M:
    pts1 = pts1 / M
    pts2 = pts2 / M
    x1 = pts1[:, 0]
    y1 = pts1[:, 1]
    x2 = pts2[:, 0]
    y2 = pts2[:, 1]

    # normalize using matrix T:
    T = np.array([[1.0 / M, 0, 0], [0, 1.0 / M, 0], [0, 0, 1]])

    A = np.vstack((x1 * x2, x2 * y1, x2, x1 * y2, y1 * y2, y2, x1, y1, np.ones(x1.shape))).T

    # solving for nullspace of A to get two F
    U, S, Vh = np.linalg.svd(A)
    f1 = Vh[-1, :]
    f2 = Vh[-2, :]
    f1 = np.reshape(f1, (3, 3))
    f2 = np.reshape(f2, (3, 3))

    # find F that meets the singularity constraint: det(a*F1 + (1-a)*F2) = 0
    # get the coefficient of the polynomial
    Ka = np.array([[1, 0.1, 0.1**2, 0.1**3], [1, 0.3, 0.3**2, 0.3**3],
                   [1, 0.6, 0.6**2, 0.6**3], [1, 0.9, 0.9**2, 0.9**3]])
    B = np.array([np.linalg.det(0.1 * f1 + (1 - 0.1) * f2), np.linalg.det(0.3 * f1 + (1 - 0.3) * f2),
                  np.linalg.det(0.6 * f1 + (1 - 0.6) * f2), np.linalg.det(0.9 * f1 + (1 - 0.9) * f2)]).T
    # k = [k0, k1, k2, k3].T, where k3a^3+k2a^2+k1a+k0
    k = np.linalg.inv(Ka) @ B
    k = k[::-1]
    alpha = np.roots(k)

    Farray = [a*f1+(1-a)*f2 for a in k]
    # refind F
    Farray = [refineF(F, pts1, pts2) for F in Farray]

    # scale the data such that F_unnormalized = T.T @ F @T
    Farray =[T.T @ F @ T for F in Farray]
    Farray = [F / F[2, 2] for F in Farray]

    return Farray


if __name__ == "__main__":
        
    correspondence = np.load('data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')

    # indices = np.arange(pts1.shape[0])
    # indices = np.random.choice(indices, 7, False)
    indices = np.array([82, 19, 56, 84, 54, 24, 18])

    M = np.max([*im1.shape, *im2.shape])

    Farray = sevenpoint(pts1[indices, :], pts2[indices, :], M)

    print(Farray)

    F = Farray[0]

    np.savez('q2_2.npz', F, M)

    # fundamental matrix must have rank 2!
    # assert(np.linalg.matrix_rank(F) == 2)

    # displayEpipolarF(im1, im2, F)

    # Simple Tests to verify your implementation:
    # Test out the seven-point algorithm by randomly sampling 7 points and finding the best solution. 
    np.random.seed(1) #Added for testing, can be commented out
    
    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    max_iter = 500
    pts1_homo = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_homo = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

    ress = []
    F_res = []
    choices = []
    M=np.max([*im1.shape, *im2.shape])
    for i in range(max_iter):
        choice = np.random.choice(range(pts1.shape[0]), 7)
        pts1_choice = pts1[choice, :]
        pts2_choice = pts2[choice, :]
        Fs = sevenpoint(pts1_choice, pts2_choice, M)
        for F in Fs:
            choices.append(choice)
            res = calc_epi_error(pts1_homo,pts2_homo, F)
            F_res.append(F)
            ress.append(np.mean(res))
            
    min_idx = np.argmin(np.abs(np.array(ress)))
    F = F_res[min_idx]
    print("Error:", ress[min_idx])


    assert(F.shape == (3, 3))
    assert(F[2, 2] == 1)
    assert(np.linalg.matrix_rank(F) == 2)
    assert(np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1)