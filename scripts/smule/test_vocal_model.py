# Generates predictions for the test files

import numpy as np
import librosa
import sys
from sklearn.mixture import GaussianMixture
from feature_extraction import feature_extractor
from sklearn import preprocessing

fs = 16000          # Resample to fs Hertz
frame_length = 512  # window size
hop_length = 256    # frame shift
n_mfcc = 13         # 13 mfcc coefficients
n_components_vocals = 8     # number of vocal mixtures in GMM
n_components_novocals = 8   # number of non vocal mixtures in GMM
covariace_type = 'diag'

def test(filename, clf, n_components_novocals, n_components_vocals):
    y,sr = librosa.load(filename,sr=44100)
    y = librosa.core.resample(y,sr,fs)
    features = feature_extractor(y,fs,frame_length,hop_length,n_mfcc)
    features = features.T
    #features = preprocessing.scale(features)
    y_predicted = np.array(clf.predict(features))

    # Post process to have 0 and 1 as class labels
    for i in range(0,n_components_novocals):
        y_predicted[y_predicted==i] = 0
    for i in range(0,n_components_vocals):
        y_predicted[y_predicted==n_components_novocals+i] = 1

    return y_predicted
