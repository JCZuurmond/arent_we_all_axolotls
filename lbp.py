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
import itertools
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import skimage
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb
import pandas as pd
import pickle


METHOD = 'uniform'
plt.rcParams['font.size'] = 9


def kullback_leibler_divergence(p, q):
    '''See scikit image LBP example'''
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


def predict(images, refs, radius, n_points, method, def_score=10,
            def_name=None, dist_func=kullback_leibler_divergence,
            n_bins=None):
    '''Predict the validation set using

    Args:
        validationset (list of str): List with path to files to be validated.
        refs (dict): {str: np.array mapping}, of the labels and their representing
            lbp matrix.
        radius (int) : Radius used for LBP.
        n_points (int) : Number of points used for LBP
        method (str): See skimage.feature.local_binary_pattern.
        def_score (int, optional): The default distance to start comparing the
            histograms with. Defaults to 10.
        def_name (str, optional): The default name when comparing the
            histograms. Defaults to None.
        dist_func(func, optional): Function to calculate distance between two
            vectors. Defaults to kull_back_leibler_divergence.
        n_bins (int, optional): Number of bins of the histogram. Defaults to
            n_points + 1.

    Returns:
        lst of tuple with two str: mapping of wanted label and predicted label.
            (wanted_label, predicted_label)
    '''
    if n_bins is None:
        n_bins = n_points + 1
    predictions = []
    for image in images:
        best_score, best_name = def_score, def_name
        im = skimage.color.rgb2gray(skimage.io.imread(image))
        lbp = local_binary_pattern(im, n_points, radius, method)
        hist, _ = np.histogram(lbp, normed=True, bins=n_bins, range=(0, n_bins))
        for name, ref_hist in refs.items():
            score = dist_func(hist, ref_hist)
            if score < best_score:
                best_score = score
                best_name = name
        predictions.append((image, best_name))
    return predictions


def plot_confusion_matrix(
    cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


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
    #redo = True
    if not redo and os.path.isfile(validate_result_path):
        print 'using saved data.'
        validate_result = pickle.load(open(validate_result_path, 'rb'))
    else:
        validate_result = predict(
            validationset, refs, radius, n_points, METHOD)
        pickle.dump(validate_result, open(validate_result_path, 'wb'))
        print 'done.'

    validate_result = [(truth[image], pred) for (image, pred) in validate_result]
    wanted_predicted = np.zeros(len(validate_result), dtype=[('wanted', 'a10'), ('predicted', 'a10')])
    wanted_predicted[:] = validate_result
    df = pd.DataFrame(wanted_predicted)
    df['match'] = df.apply(lambda row: int(row['wanted'] == row['predicted']), axis=1)
    print df

    total_acc = sum(df.match)/float(len(df.match))
    print 'got an accuracy of %s, predicted %s out of %s images correctly' \
            % (total_acc, sum(df.match), len(df.match))

    acc_grouped_df = pd.DataFrame(columns=['group', 'acc'])
    for i, (key, grp) in enumerate(df.groupby(['wanted'])):
        print key, grp
        acc = sum(grp['match'])/float(len(grp['match']))
        acc_grouped_df.loc[i] = [key, acc]
    print acc_grouped_df
    ax = acc_grouped_df.plot(x='group', kind='bar', legend=False, rot=45)
    ax.set_ylabel('accuracy')
    plt.axhline(total_acc, color='red')
    plt.text(0, total_acc, 'overall accuracy', color='red', fontsize=12)
#    plt.show()

    plt.figure()
    class_names = np.unique(df['wanted'].values)
    cnf_matrix = confusion_matrix(df['wanted'].values, df['predicted'].values)
    plot_confusion_matrix(cnf_matrix, classes=class_names, #normalize=True,
                          title='Confusion matrix')
    plt.show()

