from sklearn import metrics
from sklearn import preprocessing
from sklearn import svm
import numpy as np
import sys

def readData(filename):

	with open(filename) as F:
		ip_data = [lines.rstrip() for lines in F]

	ip_data = np.array([map(float,ip_data[i].split(",")) for i in range(len(ip_data))])
	M,N = np.shape(ip_data)
	feats = np.array([ip_data[i][0:N-1] for i in range(M)])
	labels = np.array([ip_data[i][N-1] for i in range(M)])
	return (feats,labels)


print "Reading data"
# Read inputs
train_feats, train_labels = readData(sys.argv[1])
valid_feats, valid_labels = readData(sys.argv[2])
test_feats, test_labels = readData(sys.argv[3])

#Normalize features
#train_feats = preprocessing.scale(train_feats)
#valid_feats = preprocessing.scale(valid_feats)
#test_feats = preprocessing.scale(test_feats)

# Znorm
train_mean = np.mean(train_feats,axis=0)
train_std = np.std(train_feats,axis=0)
train_feats = (train_feats - train_mean)/(train_std)
valid_feats = (valid_feats - train_mean)/(train_std)
test_feats = (test_feats - train_mean)/(train_std)

train_labels = np.round(train_labels)
valid_labels = np.round(valid_labels)
test_labels = np.round(test_labels)

print "train support vectors"
#clf1 = svm.LinearSVC(dual=True, max_iter=1000, C=1)

clf5 = svm.LinearSVC(dual=True, max_iter=1000, C=5)
clf10 = svm.LinearSVC(dual=True, max_iter=1000, C=10)
clf50 = svm.LinearSVC(dual=True, max_iter=1000, C=50)
clf100 = svm.LinearSVC(dual=True, max_iter=1000, C=100)

#clf1.fit(train_feats, train_labels)

clf5.fit(train_feats, train_labels)
clf10.fit(train_feats, train_labels)
clf50.fit(train_feats, train_labels)
clf100.fit(train_feats, train_labels)

print "test"
#ypredicted1 = clf1.predict(test_feats)

ypredicted5 = clf5.predict(test_feats)
ypredicted10 = clf10.predict(test_feats)
ypredicted50 = clf50.predict(test_feats)
ypredicted100 = clf100.predict(test_feats)



#writefile = open('SVM_test_and_predicted.txt','w')
#for i in range(len(test_labels)):
#	writefile.write(str(test_labels[i])+" "+str(ypredicted[i])+'\n')

#writefile.close()
#print(metrics.confusion_matrix(test_labels,ypredicted))
#print(metrics.accuracy_score(test_labels,ypredicted1))

print(metrics.accuracy_score(test_labels,ypredicted5))
print(metrics.accuracy_score(test_labels,ypredicted10))
print(metrics.accuracy_score(test_labels,ypredicted50))
print(metrics.accuracy_score(test_labels,ypredicted100))
