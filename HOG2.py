# Formal implementation of HOG
# The seperate scripts from HogImplementation.ipynb tidied up
# By Sebastiaan Hoekstra and Alwin Lijdsman

import os
import matplotlib.pyplot as plt
import numpy as np
import itertools
from skimage.feature import hog
from skimage import data, color, exposure
from skimage import io
#from sklearn.svm import SVC
from sklearn import svm
from sklearn import preprocessing
from skimage.transform import resize
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV


import random


''' Parameters which are adjustable '''
# Scaling pictures
pic = 450
pictureHeight = pic
pictureWidth = pic

# Labda SVM
labda = float(10)

# Kernel
krnl = 'rbf' #'linear'

# Cell size hog
cell = 5 # square
cellHeight = cell
cellWidth = cell

# block size
blk = 1
blockHeight = blk
blockWidth = blk

# orientations
orientations = 5

''' main function '''
def main():
    resultList = []
    # resultDict = {"Result":["pic", "cell", "blk", "orientations", "labda"]}

    # Put this in a loop where the 5 parameters are changed randomly
    #for i in range (0, 2):
    #print "busy with iteration", i
    #pic = random.randrange(100,600, 100) # Random between 100 and 1000, with 100 as a step
    #cell = random.randrange(4, 32, 2) # Random cell increases between 4x4 and 32x32 with steps of 2
    #blk = random.randrange(1, 10, 1)
    #orientations = random.randrange(4, 10, 1)
    #labda = random.randrange(1, 100, 1)

    score = performPipeline(pic, cell, blk, orientations, labda)
    result = {score:[pic, cell, blk, orientations, labda]}
    resultList.append(result)
    print "Percentage of succesful identifications on validation set is", score, "%"
    highest = np.argmax(resultList)
    print "result:[pic, cell, blk, orientations, labda]"
    print "Update, higest =", highest

    # After the loop, the best score:
    print "All outputs =", resultList
    bestHighest = np.argmax(resultList)
    print "Best of 100 is", bestHighest

''' Full pipeline '''
def performPipeline(pic, cell, blk, orientations, labda):
    classes, photoMatrix, labels = hogFeatureExtraction(pic, cell, blk, orientations)
    model = svmClassifier(photoMatrix, labels, labda)
    predictions, gt = predict(model, classes, pic, cell, blk, orientations)
    result = evaluate(predictions, gt)
    return result

''' Primary functions'''
def hogFeatureExtraction(pic, cell, blk, orientations):
    # Set parameters
    pictureHeight = pic
    pictureWidth = pic
    cellHeight = cell
    cellWidth = cell
    blockHeight = blk
    blockWidth = blk
    orientations = orientations

    classes = retrieveClasses()
    photoMatrix = []
    labels = []

    labelCount = 0
    for each in classes:
        photoList = retrievePhotosFromSet("trainset", str(each))
        for photo in photoList:
            scaledPhoto = scalePhoto(photo, pictureHeight, pictureWidth)
            features = hogDescription(scaledPhoto, orientations, (cellHeight, cellWidth), (blockHeight,blockWidth))
            photoMatrix.append(features)
            labels.append(labelCount)
        labelCount += 1

    # Store the data with it's representative class information - 1 matrix and 1 array of labels
    assert (len(photoMatrix) == len(labels))
    return classes, photoMatrix, labels

# Classifier kernel='rbf' or 'linear'
# Classifier kernel='rbf' or 'linear'
def svmClassifier(X, y, labda):
    #parameters = {'kernel': ['rbf'], 'C': [10],'gamma': [0.0001]}
    parameters = {'kernel': ['linear', 'rbf'], 'C': [1, 10, 100, 1000],'gamma': [0.01, 0.001, 0.0001]}            
    svr = svm.SVC()
    clf = GridSearchCV(svr, parameters)
    clf.fit(X, y)
    classifier = clf.best_estimator_
    if print_metrics:
        print()
        print('Parameters:', clf.best_params_)
    return classifier

# Prediction
def predict(clf, classes, pic, cell, blk, orientations):
    pictureHeight = pic
    pictureWidth = pic
    cellHeight = cell
    cellWidth = cell
    blockHeight = blk
    blockWidth = blk
    orientations = orientations

    # loop over validation items
    gt = []
    outputList = []
    labelCount = 0
    for each in classes:
        photoList = retrievePhotosFromSet("validationset", str(each))
        for photo in photoList:
            scaledPhoto = scalePhoto(photo, pictureHeight, pictureWidth)
            features = hogDescription(scaledPhoto, orientations, (cellHeight, cellWidth), (blockHeight,blockWidth))
            output = clf.predict(features)[0]
            outputList.append(output)
            gt.append(labelCount)
        labelCount += 1
    return outputList, gt
    
def plot_confusion_matrix(
        cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    if normalize:
        plt.clim(0, 1)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

# Evaluate
def evaluate(outputList, gt):
    outputTest = zip(gt, outputList)
    succesCount = 0
    for i in range (0, len(outputTest)):
        if outputTest[i][0] == outputTest[i][1]:
            succesCount += 1

    result = float(succesCount) / len(outputTest) *100
 
    print outputList
    print ''
    print gt
   
    class_names = np.unique(gt)
    cnf_matrix = confusion_matrix(gt, outputList)
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                              title='Confusion matrix')
                              
                              
    return result

''' Secondary functions'''
# Retrieve items from target set
def retrievePhotosFromSet(targetSet, targetClass):
    photoList = []
    counter = 0
    for item in os.listdir('.'):
        if item == "dataset":
            path = item
            for photoSet in os.listdir(path):
                if photoSet == targetSet:
                    path += "/"+str(photoSet)
                    for classType in os.listdir(path):
                        if classType == targetClass:
                            path = path + "/"+str(classType)
                            for photo in os.listdir(path):
                                image = color.rgb2gray(io.imread(path+"/"+str(photo)))
                                photoList.append(image)
    #print len(photoList)
    return photoList

# Retrieves all classes
def retrieveClasses():
    targetSet = "trainset"
    classes = []
    for item in os.listdir('.'):
        if item == "dataset":
            path = item
            for photoSet in os.listdir(path):
                if photoSet == targetSet:
                    path += "/"+str(photoSet)
                    for classType in os.listdir(path):
                        classes.append(classType)
    classes = classes[1:] # Cut of the DS. folder that is invisible
    return classes

# Make class dict
def createClassDict(classes):
    classNumberDict = {}
    for number, each in enumerate(classes):
        classNumberDict[number] = each
    return classNumberDict

# Scale photos
def scalePhoto(image, width, heigth):
    new_image = resize(image, (width,heigth))
    return new_image

# HOG process images
def hogDescription(image, orientations, pixels_per_cell, cells_per_block):
    fd = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell ,cells_per_block=cells_per_block)
    return fd


''' Helper functions '''
# remove Sklearn warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


main()
