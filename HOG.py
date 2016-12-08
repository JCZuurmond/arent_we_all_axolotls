# Formal implementation of HOG
# The seperate scripts from HogImplementation.ipynb tidied up
# By Alwin Lijdsman

# TODO Make it so that you can turn of the random iterations and can also easilty to 1 full implementation
# with hand coded parameters
# 2 don't reload full image set each time? - errors no

import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
from skimage import data, color, exposure
from skimage import io
from sklearn.svm import SVC
from sklearn import preprocessing
from skimage.transform import resize

import random


''' Parameters which are adjustable '''
# # Scaling pictures
# pic = 600
# pictureHeight = pic
# pictureWidth = pic
#
# # Labda SVM
# labda = float(10)
#
# # Cell size hog
# cell = 16 # square
# cellHeight = cell
# cellWidth = cell
#
# # block size
# blk = 1
# blockHeight = blk
# blockWidth = blk
#
# # orientations
# orientations = 9

''' main function '''
def main():
    resultList = []
    # resultDict = {"Result":["pic", "cell", "blk", "orientations", "labda"]}

    # Put this in a loop where the 5 parameters are changed randomly
    for i in range (0, 2):
        print "busy with iteration", i
        pic = random.randrange(100,600, 100) # Random between 100 and 1000, with 100 as a step
        cell = random.randrange(4, 32, 2) # Random cell increases between 4x4 and 32x32 with steps of 2
        blk = random.randrange(1, 10, 1)
        orientations = random.randrange(4, 10, 1)
        labda = random.randrange(1, 100, 1)

        score = performPipeline(pic, cell, blk, orientations, labda)
        result = {score:[pic, cell, blk, orientations, labda]}
        resultList.append(result)
        print "Percentage of succesful identifications on validation set is", score, "%"
        highest = resultList[np.argmax(test)]
        print "result:[pic, cell, blk, orientations, labda]"
        print "Update, higest =", highest

    # After the loop, the best score:
    print "All outputs =", resultList
    bestHighest = resultList[np.argmax(test)]
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

# Classifier
def svmClassifier(X, y, labda):
    clf = SVC(C=labda, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)
    clf.fit(X, y)
    return clf

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

# Evaluate
def evaluate(outputList, gt):
    outputTest = zip(gt, outputList)
    succesCount = 0
    for i in range (0, len(outputTest)):
        if outputTest[i][0] == outputTest[i][1]:
            succesCount += 1

    result = float(succesCount) / len(outputTest) *100
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
