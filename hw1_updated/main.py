from os.path import join

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import util
import visual_words
import visual_recog
from opts import get_opts


def main():
    opts = get_opts()

    # Q1.1
    # img_path = join(opts.data_dir, 'aquarium/sun_aztvjgubyrgvirup.jpg')#kitchen/sun_aasmevtpkslccptd
    # img = Image.open(img_path) #dtype = uint8 0-255
    # img = np.array(img).astype(np.float32) / 255 #dtype = float32, 0-1
    # filter_responses = visual_words.extract_filter_responses(opts, img)
    # util.display_filter_responses(opts, filter_responses)

    # Q1.2
    # n_cpu = util.get_num_CPU()
    # visual_words.compute_dictionary(opts, n_worker=n_cpu)

    # Q1.3
    # img_path = join(opts.data_dir, 'laundromat/sun_aalvewxltowiudlw.jpg') # kitchen/sun_aasmevtpkslccptd, aquarium/sun_atyotlbmwehdotaw
    # img = Image.open(img_path)
    # img = np.array(img).astype(np.float32)/255
    # dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
    # wordmap = visual_words.get_visual_words(opts, img, dictionary)
    # util.visualize_wordmap(wordmap)

    # Q2.1 Test
    # hist = visual_recog.get_feature_from_wordmap(opts, wordmap)

    # Q2.2 Test
    # hist = visual_recog.get_feature_from_wordmap_SPM(opts, wordmap)


    # Q2.1-2.4
    # n_cpu = util.get_num_CPU()
    # visual_recog.build_recognition_system(opts, n_worker=n_cpu)

    # Q2.5
    # n_cpu = util.get_num_CPU()
    # conf, accuracy = visual_recog.evaluate_recognition_system(opts, n_worker=n_cpu)
    #
    # print(conf)
    # print(accuracy)
    # np.savetxt(join(opts.out_dir, 'confmat.csv'), conf, fmt='%d', delimiter=',')
    # np.savetxt(join(opts.out_dir, 'accuracy.txt'), [accuracy], fmt='%g')

    # # Q2.6
    # img_path1 = join(opts.data_dir, 'kitchen/sun_afmsnwrincaowafk.jpg')
    # img1 = Image.open(img_path1)
    # img1 = np.array(img1).astype(np.float32) / 255
    # dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
    # wordmap1 = visual_words.get_visual_words(opts, img1, dictionary)
    # util.visualize_wordmap(wordmap1)
    #
    # img_path2 = join(opts.data_dir, 'laundromat/sun_aiyluzcowlbwxmdb.jpg')
    # img2 = Image.open(img_path2)
    # img2 = np.array(img2).astype(np.float32) / 255
    # dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
    # wordmap2 = visual_words.get_visual_words(opts, img2, dictionary)
    # util.visualize_wordmap(wordmap2)

    # Q3.1
    n_cpu = util.get_num_CPU()
    # create a new dictionary
    visual_words.compute_dictionary(opts, n_worker=n_cpu)
    print("dictionary created")
    # create a new trained file
    visual_recog.build_recognition_system(opts, n_worker=n_cpu)
    print("trained file created")
    # evaluate accuracy
    conf, accuracy = visual_recog.evaluate_recognition_system(opts, n_worker=n_cpu)

    print(conf)
    print(accuracy)
    # np.savetxt(join(opts.out_dir, 'confmat.csv'), conf, fmt='%d', delimiter=',')
    # np.savetxt(join(opts.out_dir, 'accuracy.txt'), [accuracy], fmt='%g')


if __name__ == '__main__':
    main()
