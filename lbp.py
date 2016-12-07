#!/usr/bin/python2.7
'''Local Binary Pattern for texture Classification.

Here we will use LBP (Local Binary Pattern) to classify textures. LBP looks
at points surrounding a central point and tests whether the surrounding points
are grater than or less thant the central point (i.e. gives a binary result).

Author: Cor Zuurmond <jczuurmond@gmail.com>

Src: The code is based on the LBP example of scikit-image
    http://scikit-image.org/docs/dev/auto_examples/plot_local_binary_pattern.html
'''


import csv
import glob
import os

import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb
import pickle


METHOD = 'uniform'
plt.rcParams['font.size'] = 9


def kullback_leibler_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))


def avg_lbp_hist(
        images, n_points, radius, method, range=(None, None), bins=None):
    '''Creates a histogram of the average lbp for a set of images.

    Args:
        images (lst of str or lst of images): A list containing paths
            to images or the images.
        n_points (int): See skimage.feature.local_binary_pattern
        radius (int): See skimage.feature.local_binary_pattern.
        method (str): See skimage.feature.local_binary_pattern.
        range ((float, float), optional): Max and min values of the histogram.
            Defaults to (0, n_points+1).
        bins (integer): Number of bins of the histogram. Defaults to
            n_poin  ts + 1.

    Returns:
        numpy.histogram: A histogram containing the average distribution.
    '''
    if range == (None, None):
        range = (0, n_points+1)
    if bins == None:
        bins = n_points+1
    avg_hist = np.zeros(bins)
    for im in images:
        if isinstance(im, str):
            im = skimage.io.imread(im)
        im = skimage.color.rgb2gray(im)
        lbp = local_binary_pattern(im, n_points, radius, method)
        hist, bins = np.histogram(lbp, normed=True, range=range, bins=bins)
        avg_hist = np.add(avg_hist, hist)
    return avg_hist/len(images)


def train(trainset, radius, n_points, method, redo=False):
    '''Trains the data using LBP.

    Args:
        trainset (list of tuples with (str, list of str)). Mapping of the
            labels with their images.
        radius (int) : Radius used for LBP.
        n_points (int) : Number of points used for LBP
        method (str, optional): See skimage.feature.local_binary_pattern.
        redo (bool, optional): Redoes the training if True, else uses saved
            history. Defaults to False.

    Returns:
        dict {str: np.array}: Mapping of the labels with their average LBP
            matrix.
    '''
    refs = {}
    for label, images in trainset:
        print 'Training for %s ...' % label,
        file_name = '%s_ref_lbp.npy' % label
        if not redo and os.path.isfile(file_name):
            print 'using saved data.'
            lbp = np.load(file_name)
            refs[label] = lbp
            continue
        if os.path.isfile(file_name):
            os.remove(file_name)
        lbp = avg_lbp_hist(images, n_points, radius, method)
        np.save(file_name, lbp)
        refs[label] = lbp
        print 'done.'
    return refs


def validate(validationset, refs, truth, radius, n_points, method,
             dist_func=kullback_leibler_divergence, n_bins=None):
    '''Validates the validation set using

    Args:
        validationset (list of str): List with path to files to be validated.
        refs (dict): str: np.array mapping, of the labels and their representing
            lbp matrix.
        truth (dict): Mapping of actual label per filename.
        radius (int) : Radius used for LBP.
        n_points (int) : Number of points used for LBP
        method (str, optional): See skimage.feature.local_binary_pattern.
        dist_func(func, optional): Function to calculate distance between two
            vectors. Defaults to kull_back_leibler_divergence.
        n_bins (int, optional): Number of bins of the histogram. Defaults to
            n_points + 1.

    Returns:
        ((float, int), dict, list of tuples): ((accuracy, total images count),
            mapping of (accuracy, image count) per label, list of (actual
            label, predicted label).
    '''
    if n_bins is None:
        n_bins = n_points + 1
    matches, matches_per_label, wanted_got = 0., {}, []
    for image in validationset:
        best_score = 10
        best_name = None
        im = skimage.io.imread(image)
        im = skimage.color.rgb2gray(im)
        lbp = local_binary_pattern(im, n_points, radius, METHOD)
        hist, _ = np.histogram(lbp, normed=True, bins=n_bins, range=(0, n_bins))
        for name, ref_hist in refs.items():
            score = kullback_leibler_divergence(hist, ref_hist)
            if score < best_score:
                best_score = score
                best_name = name
        label = truth[image]
        match = int(label == best_name)
        matches += match
        if not label in matches_per_label:
            matches_per_label[label] = [match, 1]
        else:
            matches_per_label[label][0] += match
            matches_per_label[label][1] += 1.
        wanted_got.append((truth[image], best_name))
    for k, v in matches_per_label.iteritems():
        matches_per_label[k] = (v[0]/v[1], v[1])
    tot_count = len(validationset)
    return ((matches/tot_count, tot_count), matches_per_label, wanted_got)


if __name__ == '__main__':
    radius = 3
    n_points = 8*radius
    #redo = True
    redo = False

    trainset_path = os.path.join('dataset', 'trainset')
    trainset = []
    for p in glob.glob(os.path.join(trainset_path, '*')):
        label = os.path.basename(p)
        trainset.append((label, glob.glob(os.path.join(p, '*'))))

    refs = train(trainset, radius, n_points, METHOD, redo=redo)

    validationset_path = os.path.join('dataset', 'validationset')
    validationset = glob.glob(os.path.join(validationset_path, '*', '*'))

    val_set_truth = 'valset-overview.csv'
    truth = {}
    with open(val_set_truth, 'r') as csvfile:
        r = csv.reader(csvfile, delimiter=',')
        for row in r:
            truth[row[0]] = row[1]

    validate_result_path = 'lbp_validate_result.p'
    print 'validating ...',
    if not redo and os.path.isfile(validate_result_path):
        print 'using saved data.'
        validate_result = pickle.load(open(validate_result_path, 'rb'))
    else:
        validate_result = validate(
            validationset, refs, truth, radius, n_points, METHOD)
        pickle.dump(validate_result, open(validate_result_path, 'wb'))
        print 'done.'

    ((accuracy, tot_count), matches_per_label, wanted_got) = validate_result
    print 'got an accuracy of %s for %s images' % (accuracy, tot_count)

    labels, accs, counts = [], [], []
    for label, (acc, count) in matches_per_label.iteritems():
        print 'got an accuracy of %s for %s images for label %s' % \
                (acc, count, label)
        labels.append(label)
        accs.append(acc)
        counts.append(count)

    ind = np.arange(len(labels))
    width = 0.5

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, accs, width, color='r')
    #rects2 = ax.bar(ind+width, counts, width, color='y')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy per label')
    ax.set_xticks(ind+0.5*width)
    ax.set_xticklabels(labels)

    #ax.legend((rects1[0], rects2[0]), ('Men', 'Women'))
    plt.show()

