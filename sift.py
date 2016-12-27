def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import sys
sys.path.append("pySift")
from pySift import sift, matching
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import euclidean_distances

trainimages_sift = np.load('sift_all_image.npy')
valimages_sift = np.load('sift_valpoints_image.npy')
dictionary = np.load('sift_all_concet.npy')

nr_clusters = 80

# Maak clusters 
traincluster = KMeans(n_clusters=nr_clusters, max_iter=25, n_jobs=-1).fit(dictionary)

from sklearn.cluster import KMeans
#Calculate the histogram representations for the train images
train_feat = np.zeros((len(trainimages_sift),nr_clusters))

for i in xrange(len(trainimages_sift)):#len(trainimages)
    sift_features = trainimages_sift[i]  #uit array halen
    dis = traincluster.predict(sift_features)
    train_feat[i] = np.histogram(dis, bins=range(nr_clusters+1))[0]

# Normalize histogram
from sklearn import preprocessing

train_feat_normalized = preprocessing.normalize(train_feat, norm='l2')

#Training ground truth labels
train_labels = np.array([int(line.strip().split(" ")[1]) for line in open("trainset-overview.txt", "r")])

#Validation ground truth labels
val_labels = np.array([int(line.rstrip().split(' ')[1]) for line in open('valset-overview.txt','r')])

#Calculate the histogram representations for the validation images
val_feat = np.zeros((len(valimages_sift), nr_clusters))

for i in xrange(len(valimages_sift)):
    sift_features = valimages_sift[i]
    dis = traincluster.predict(sift_features)
    val_feat[i] = np.histogram(dis, bins=range(nr_clusters+1))[0]
    
val_feat_normalized = preprocessing.normalize(val_feat, norm='l2')

#train classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

pipeline = Pipeline([
#     ('tree', tree.DecisionTreeClassifier()),
#     ('randomForest', RandomForestClassifier()),
#     ('gnb', GaussianNB()),
        ('svm', svm.SVC(kernel='linear')),
#     ('knn', KNeighborsClassifier(n_neighbors=10)),
])

pipeline.fit(train_feat_normalized, train_labels) 

#Predict the classes of the images in the validation set using the classifier
y_predict = pipeline.predict(val_feat_normalized)
print y_predict
acc = pipeline.score(val_feat_normalized,val_labels)
print acc
