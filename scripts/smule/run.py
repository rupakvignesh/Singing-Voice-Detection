"""
Wrapper script to run VocalDetection
Takes 5 arguments:
training_files -- file containing list paths to train data
bootstrap_files -- file containing list of paths to bootstrap data
bootstrap_targets -- file containing list of paths to bootstrap targets
test_files -- file containing list of paths to test data
test_targets -- file containing list of paths to test targets 
"""

import numpy as np
import librosa
import sys
from sklearn.mixture import GaussianMixture
from VocalDetection import *
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score


# Read Arguments
training_files = sys.argv[1]    # path list to train files
bootstrap_files = sys.argv[2]   # path list to bootstrap files
bootstrap_targets = sys.argv[3] # path list to bootstrap target lab files
test_files = sys.argv[4]        # path list to test files
test_targets = sys.argv[5]      # path list to test target lab files

# Read input files
with open(training_files) as F1:
    train_list = [lines.rstrip() for lines in F1]

with open(bootstrap_files) as F2:
    bootstrap_list = [lines.rstrip() for lines in F2]

with open(bootstrap_targets) as F3:
    target_list = [lines.rstrip() for lines in F3]

with open(test_files) as F4:
    test_list = [lines.rstrip() for lines in F4]

with open(test_targets) as F5:
    test_target_list = [lines.rstrip() for lines in F5]

# Create two GMMs (novocal clf and vocal clf ) using bootstrap data
(novocal_clf, vocal_clf) = bootstrap_model(bootstrap_list,target_list)

# Combine the GMMs and Initialize a model for EM estimation of train data parameters
(clf, n_components_novocals, n_components_vocals) = train(train_list, novocal_clf, vocal_clf)

#print the train model
print clf

# Test the model
y_pred_total = np.array([])
y_targets_total = np.array([])

for i in range(len(test_list)):
    #Test file by file and return output predictions
    print "Testing "+test_list[i]
    y_pred,sr,hop_length = test(test_list[i],clf,n_components_novocals, n_components_vocals)

    # Read targets
    with open(test_target_list[i]) as TL:
        y_targets = np.array(map(float,[lines.rstrip() for lines in TL]))

    # Append 0's to have same length as output
    y_targets = np.append(y_targets,np.zeros(len(y_pred)-len(y_targets)))
    y_pred_total = np.append(y_pred_total, y_pred)
    y_targets_total = np.append(y_targets_total, y_targets)
    print confusion_matrix(y_targets,y_pred)

#Print the results
print "Accuracy: ", accuracy_score(y_targets_total, y_pred_total)
print "Precision: ", precision_score(y_targets_total, y_pred_total)
print "Recall: ",recall_score(y_targets_total, y_pred_total)
print "F1 Score: ", f1_score(y_targets_total, y_pred_total)
