from sklearn import metrics
from sklearn import preprocessing
from sklearn import svm
import numpy as np

print "Loading Data"
with open('features1000_48.csv','r') as F1:
	flines = [lines.rstrip() for lines in F1]

N = len(flines)
per_split = 0.80


flines_parsed = []
for i in range(len(flines)):
	flines_parsed.extend([map(float, str.split(flines[i],','))])

Z = np.array(flines_parsed)
N = Z.shape[0]
M = Z.shape[1]
X = [Z[i][0:M-1] for i in range(len(Z))]
y = [Z[i][M-1] for i in range(len(Z))]


t1 = int(N/2*per_split)
h = int(N/2)
xtrain = np.append(X[0:t1],X[h:h+t1],axis=0)
ytrain = np.append(y[0:t1],y[h:h+t1],axis=0)

xtest = np.append(X[t1:h],X[h+t1:N],axis=0)
ytest = np.append(y[t1:h],y[h+t1:N],axis=0)

xtrain = preprocessing.scale(xtrain)
xtest = preprocessing.scale(xtest)


print "train support vectors"
clf = svm.SVC(gamma = 0.0001, kernel='rbf',C=1000, max_iter = 10000)
clf.fit(xtrain,ytrain)

print "test"
ypredicted = clf.predict(xtest)
print(metrics.confusion_matrix(ytest,ypredicted))
