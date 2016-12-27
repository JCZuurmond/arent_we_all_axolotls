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

# Evaluate
def evaluate:
	
	testimages_sift = np.load('sift_testpoints_images.npy')
	testimages = [line.strip().split(' ')[0] for line in open('testset-overview-final.txt','r')]
	#Calculate the histogram representation for the test images.
	test_feat = np.zeros((len(testimages_sift), nr_clusters))

	for i in xrange(len(testimages_sift)):
	    sift_features = testimages_sift[i]
	    dis = traincluster.predict(sift_features)
	    test_feat[i] = np.histogram(dis, bins=range(nr_clusters+1))[0]
	    
	test_feat_normalized = preprocessing.normalize(test_feat, norm='l2')

	#Use your classifier to make class predictions:
	test_predictions = pipeline.predict(test_feat_normalized)

	#Fill in the 'test_predictions' variable with your predictions.

	#We save your predictions to file
	test_p_file = open('all_80clust_lin.csv','w')
	test_p_file.write('ImageName,Prediction\n')
	for i,image in enumerate(testimages):
	    test_p_file.write(image+','+str(int(test_predictions[i]))+'\n')
	test_p_file.close()

# create confusion matrix
def create_confusion_matrix:
	class_names = ["Blue whale", "Chihuahua", "Chimpanzee", "fox", "gorilla", "Killer whale", "Seal", "Tiger", "Wolf", "Zebra"]

	from sklearn.metrics import confusion_matrix
	import itertools
	import matplotlib.pyplot as plt
	%matplotlib qt 

	# Compute confusion matrix
	cnf_matrix = confusion_matrix(val_labels, y_predict)
	np.set_printoptions(precision=2)

	# Plot non-normalized confusion matrix
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=class_names)

	# Plot normalized confusion matrix
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True)

	plt.show()

def plot_confusion_matrix:
        cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
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
