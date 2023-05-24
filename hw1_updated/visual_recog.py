import os
import math
import multiprocessing
from os.path import join
from copy import copy

import imageio
import numpy as np
from PIL import Image

import visual_words
import matplotlib.pyplot as plt
from opts import get_opts


def get_feature_from_wordmap(opts, wordmap):
    """
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    """

    K = opts.K

    # histogram need to be computed over the flattened array.
    flat_wordmap = wordmap.flatten()
    hist, bin_edges = np.histogram(flat_wordmap, bins=K, density=True)
    hist = hist / np.sum(hist)
    # print("success")

    return hist


def get_feature_from_wordmap_SPM(opts, wordmap):
    """
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape K*(4^(L+1) - 1) / 3
    """

    K = opts.K
    L = opts.L

    # calculate the weight for each layer
    # layer 0 and 1 is 2^(-L)
    weight = []
    for i in range(0, L + 1):
        if i == 0 or i == 1:
            weight.append(np.exp2(-L))
        else:
            weight.append(np.exp2(i - L - 1))

    hist_all = []
    # split layers to cells (2^l x 2^l)
    for i in range(0, L + 1):
        idx_num = np.exp2(i)
        row_div = np.array_split(wordmap, idx_num, axis=0)
        for r in row_div:
            small_cell = np.array_split(r, idx_num, axis=1)
            # print("ok in spm wordmap")
            for idx in small_cell:
                hist_cell, bin_edges = np.histogram(idx, bins=K)
                # concatenate all list
                hist_all = np.append(hist_all, weight[i] * hist_cell)

    # normalize by all total feature
    hist_all = hist_all / np.sum(hist_all)
    # print(hist_all.shape)

    return hist_all

def get_image_feature(opts, img_path, dictionary):
    """
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K)
    """
    # load an image
    img = imageio.imread('../data/' + img_path)
    img = img.astype('float') / 255

    # extract word map from the image
    wordmap = visual_words.get_visual_words(opts, img, dictionary)

    # compute the SPM
    feature = get_feature_from_wordmap_SPM(opts, wordmap)

    # return computed feature
    return feature


def build_recognition_system(opts, n_worker=8):
    """
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    """

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, "train_files.txt")).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, "train_labels.txt"), np.int32)
    dictionary = np.load(join(out_dir, "dictionary.npy"))

    # training data as an array
    train_data = np.asarray(train_files)  # ['aq/sun_as' 'aq/sun_atk',..]
    N = train_data.shape[0]

    K = opts.K

    pool = multiprocessing.Pool(processes=n_worker)

    response = []
    for i in range(0, N):
        args = [opts, train_data[i], dictionary]  # args:[()]
        # print()
        response.append(pool.apply_async(get_image_feature, args))

    feat_tmp = []
    for res in response:
        feat_tmp.append(res.get())

    # feature size M:
    M = int(K * (4**(SPM_layer_num+1)-1) / 3)

    # reshape features to be N*M
    features = np.reshape(feat_tmp,(N, M))
    # example code snippet to save the learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
        features=features,
        labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )


def distance_to_set(word_hist, histograms):
    """
    Compute distance between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    """
    # discard last channel for laundromat/sun afrrjykuhhlwiwun.jpg HOW??

    # similarity intersection is the sum mini value of each corresponding bins
    intersect = np.minimum(word_hist, histograms)
    sim = np.sum(intersect, axis=1)
    return sim


def get_test_feature(args):
    """
        Compute the test features for the test set.

        [input]
        * args: the args are user-defined

        [output]
        * test_feature: numpy.ndarray of shape (K)
        """
    img_path, dictionary, test_opts = args
    test_feature = get_image_feature(test_opts, img_path, dictionary)
    return test_feature


def evaluate_recognition_system(opts, n_worker=8):
    """
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    """

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, "trained_system.npz"))
    dictionary = trained_system["dictionary"]

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    # test_opts.K = dictionary.shape[0]
    # test_opts.L = trained_system["SPM_layer_num"]
    # hyperparameter
    test_opts.K = 30
    test_opts.L = 3

    test_files = open(join(data_dir, "test_files.txt")).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, "test_labels.txt"), np.int32)

    # load trained image and their labels and features
    trained_label = trained_system["labels"]
    trained_feature = trained_system["features"]

    # load test image and their labels and features
    test_data = np.asarray(test_files)
    test_label = np.asarray(test_labels)

    # obtain feature for test images
    pool = multiprocessing.Pool(processes=n_worker)

    response = []
    for i in range(0, len(test_data)):
        args = [(test_data[i], dictionary, test_opts)]
        response.append(pool.apply_async(get_test_feature, args))

    test_feature = []
    for res in response:
        test_feature.append(res.get())

    # create a confusion matrix size should be 8x8
    size_C = len(np.unique(test_label)) # 8
    conf = np.zeros((size_C, size_C))

    for i, t_feature in enumerate(test_feature):
        # compute the predicted label of each one that is distance between img from test and train set
        dist = distance_to_set(t_feature, trained_feature)
        idx = np.argmax(dist)
        conf[test_label[i], trained_label[idx]] += 1

    accuracy = np.diag(conf).sum() / conf.sum()

    return conf, accuracy
