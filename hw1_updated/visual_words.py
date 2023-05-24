import os
import multiprocessing
from os.path import join, isfile

import numpy as np
import scipy.ndimage
import skimage.color
from PIL import Image
import sklearn.cluster
from sklearn.cluster import KMeans

from opts import get_opts
import imageio


def extract_filter_responses(opts, img):
    """
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3) or (H,W,4) with range [0, 1]
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    """

    # get the filter scale [1,2,4]
    filter_scales = opts.filter_scales

    # create variable needs to return
    filter_responses = []

    # check number of input image channels and convert it to 3-channel
    if img.shape[2] < 3:
        img = np.dstack([img] * 3)
    # higher channel images to 3 channels, disregard the last channel
    if img.shape[2] > 3:
        img = img[:, :, :3]

    # convert the image into lab color space
    lab_image = skimage.color.rgb2lab(img)

    # scale -> filter -> channel
    for sig in filter_scales:
        # Gaussian filter
        for i in range(0, 3):
            # filter each channel separately
            filter_responses.append(scipy.ndimage.gaussian_filter(lab_image[:, :, i], sigma = sig))

        # Laplacian Gaussian filter:
        for i in range(0, 3):
            filter_responses.append(scipy.ndimage.gaussian_laplace(lab_image[:, :, i], sigma = sig))

        # Gaussian Derivative in x direction
        for i in range(0, 3):
            filter_responses.append(scipy.ndimage.gaussian_filter(lab_image[:, :, i], sigma = sig, order = [1, 0]))

        # Gaussian Derivative in y direction
        for i in range(0, 3):
            filter_responses.append(scipy.ndimage.gaussian_filter(lab_image[:, :, i], sigma = sig, order = [0, 1]))

    # stack all the filters each with 3 channels, the total should be 36: 3*4*3
    filter_responses = np.dstack(filter_responses)

    return filter_responses


def compute_dictionary_one_image(args):
    """
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    """

    # Here args is a collection of arguments passed into the function.
    opts = get_opts()
    i, alpha, path = args

    # Inside compute dictionary one image(), you should read an image,
    img = imageio.imread('../data/' + path)
    img = img.astype('float') / 255

    # extract the responses
    filter_responses = extract_filter_responses(opts, img)

    # Random sampling of responses:
    h = filter_responses.shape[0]
    w = filter_responses.shape[1]
    c = filter_responses.shape[2]
    random_sampling = np.reshape(filter_responses, (h * w, c))

    # generate alpha random pixel in the image
    i = np.random.randint(h * w, size=alpha)
    random_sampling = random_sampling[i, :]

    # print('complete one_img')
    return random_sampling

    # Saving the sampled responses to a temporary file
    np.save('%s%d' % (sample_response_path, i), np.asarray(random_sampling))



def compute_dictionary(opts, n_worker=8):
    """
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel

    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    """

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K

    train_files = open(join(data_dir, "train_files.txt")).read().splitlines()#train files.txt

    train_data = np.asarray(train_files)  # ['aq/sun_as' 'aq/sun_atk',..]

    T = train_data.shape[0]  # size = 1177
    # T = 400
    alpha = opts.alpha

    pool = multiprocessing.Pool(processes=n_worker)
    responses = []
    for i in range(0, T):
        args = [(i, alpha, train_data[i])] # args:[(0,25,'aqua/sun_as.jpg')]
        # print()
        responses.append(pool.apply_async(compute_dictionary_one_image, args))

    features = []
    for res in responses:
        features.append(res.get())

    # collect a matrix filter_responses over all images that is alpha*T x 3F
    filter_responses = features[0]
    for i in range(1, len(features)):
        filter_responses = np.concatenate((filter_responses, features[i]), axis=0)

    # run k-means clustering
    kmeans = sklearn.cluster.KMeans(n_clusters=K).fit(filter_responses)
    dictionary = kmeans.cluster_centers_
    # print(dictionary.shape)  # 10, 36

    # example code snippet to save the dictionary
    np.save(join(out_dir, 'dictionary.npy'), dictionary)

    # print('success!')
    return dictionary



def get_visual_words(opts, img, dictionary):
    """
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)

    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    """

    # each pixel in wordmap is assigned the closest visual word of the filter response at the respective pixel in img.
    filter_responses = extract_filter_responses(opts, img)
    h = filter_responses.shape[0]
    w = filter_responses.shape[1]

    wordmap = np.zeros((h,w))
    for i in range(0, h):
        for j in range(0, w):
            pixel = filter_responses[i,j,:]
            # use standard Euclidean distance
            dist = scipy.spatial.distance.cdist(np.array([pixel]), dictionary, 'euclidean')
            # find the closest between
            [best_match] = np.where(dist == np.min(dist))[1]
            wordmap[i, j] = best_match

    # print(wordmap.shape)
    return wordmap



