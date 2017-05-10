###################
# VocalDetection  #
###################

import numpy as np
import librosa
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing
from scipy.signal import butter, lfilter
from scipy import ndimage

# Global variables
fs = 16000          # Resample incoming audio to fs Hertz
frame_length = 512  # window size
hop_length = 256    # frame shift
n_mfcc = 13         # number of mfcc coefficients
n_features = 47             # feature dimension
n_components_vocals = 8     # number of vocal mixtures in GMM
n_components_novocals = 8   # number of non vocal mixtures in GMM
covariance_type = 'diag'    # type of GMM covariance
max_iter = 300              # number of iterations of EM
low = 300                   # low cut off of bandpass_filter
high = 5000                 # high cut off of bandpass_filter
n_smooth_targets = 1       # number of zeros/ones interspersed in the tagets to be smoothed



"""
function:: feature_extractor
Extracts mfccs, their velocity and acceleration coefficients, spectral_centroid
and spectral_contrast.
"""
def feature_extractor(y, sr, frame_length, hop_length, n_mfcc):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,n_fft=frame_length,hop_length=hop_length)
    mfccs_del = librosa.feature.delta(mfccs)
    mfccs_del_del = librosa.feature.delta(mfccs_del, order=2)
    spec_centroid = librosa.feature.spectral_centroid(y=y,sr=sr,n_fft=frame_length,hop_length=hop_length)
    spec_contrast = librosa.feature.spectral_contrast(y=y,sr=sr,n_fft=frame_length,hop_length=hop_length)
    features = np.concatenate((mfccs,mfccs_del,mfccs_del_del, spec_centroid, spec_contrast, spec_rolloff),axis=0)
    return features


"""
function:: bandpass_filter
Implements a butterworth bandpass_filter. Takes in signal y and outputs filtered
signal bandpassed with cut off frequencies low and high (in Hertz)
"""
def bandpass_filter(y,sr,low,high,order):
    low = float(low)/(sr/2)                         # compute low and high digital frequencies
    high = float(high)/(sr/2)
    b,a = butter(order, [low, high], btype='band')  # compute filter coefficients
    y = lfilter(b,a,y)                              # filter input
    return y


"""
function:: train
Initializes a single GMM with first set of mixtures from novocal class and next
set of mixtures from vocal class. Iteratively estimates GMM parameters using EM
algorithm using the features from train_list files.
"""
def train(train_list, novocal_clf, vocal_clf):
    print "Extracting Train Features"
    train_features = np.empty((n_features,0))
    for i in range(len(train_list)):
        print train_list[i]
        y,sr = librosa.load(train_list[i],sr=fs)
        y = librosa.effects.hpss(y)[0]          #Perform HPSS
        y = bandpass_filter(y,sr,low,high,2)    #Bandpass
        train_features = np.concatenate((train_features, feature_extractor(y,fs,frame_length,hop_length,n_mfcc)),axis=1)

    # Transpose feature matrix
    train_features = train_features.T
    # Build a Tri-gaussian model from the extracted features
    clf = GaussianMixture(n_components=n_components_vocals+n_components_novocals,covariance_type=covariance_type,max_iter=max_iter)
    # Initialize model with bootstrap models
    clf.means_init = np.concatenate((novocal_clf.means_,vocal_clf.means_))
    # Expectation-Maximization
    print "EM Estimations of parameters"
    clf.fit(train_features)
    return (clf, n_components_novocals, n_components_vocals)


"""
function:: bootstrap_model
Builds two GMMs, one for vocal class another for novocal class
"""
def bootstrap_model(bootstrap_list,target_list):
    # Build models with bootstrap data
    bootstrap_features = np.empty((n_features,0))
    targets = np.empty((0,0))
    print 'Extracting Bootstrap Features'
    for i in range(len(bootstrap_list)):
        print bootstrap_list[i]
        y,sr = librosa.load(bootstrap_list[i],sr=fs)
        y = librosa.effects.hpss(y)[0]              #Perform HPSS
        y = bandpass_filter(y,sr,low,high,2)        #Bandpass
        temp_bootstrap_features = feature_extractor(y,fs,frame_length,hop_length,n_mfcc)
        bootstrap_features = np.concatenate((bootstrap_features, temp_bootstrap_features),axis=1)
        with open(target_list[i]) as TL:
            temp_targets = np.array(map(float,[lines.rstrip() for lines in TL]))
        # Append zeros to targets to match number of frames
        temp_targets = np.append(temp_targets,np.zeros(np.shape(temp_bootstrap_features)[1]-len(temp_targets)))
        targets = np.append(targets,temp_targets)

    # Transpose feature matrix
    bootstrap_features = bootstrap_features.T

    # Initialize models and do Expectation-Maximization
    print 'Training bootstrap models'
    novocal_clf = GaussianMixture(n_components=n_components_novocals,covariance_type=covariance_type,max_iter=max_iter)
    vocal_clf = GaussianMixture(n_components=n_components_vocals,covariance_type=covariance_type,max_iter=max_iter)
    novocal_clf.fit(bootstrap_features[targets==0])
    vocal_clf.fit(bootstrap_features[targets==1])

    return (novocal_clf, vocal_clf)


"""
function:: test
Takes input file, generates features and tests each instance against the mixtures in clf
Also smooths the output so that the sudden changes of 0's and 1's are smoothed.
"""
def test(filename, clf, n_components_novocals, n_components_vocals):
    y,sr = librosa.load(filename,sr=fs)
    y = librosa.effects.hpss(y)[0]          #Perform HPSS
    y = bandpass_filter(y,sr,low,high,2)    #Bandpass
    features = feature_extractor(y,fs,frame_length,hop_length,n_mfcc)
    features = features.T
    y_predicted = np.array(clf.predict(features))   #GMM estimator predicts mixtures

    # Post process mixture labels to have 0 and 1 as class labels
    for i in range(0,n_components_novocals):
        y_predicted[y_predicted==i] = 0
    for i in range(0,n_components_vocals):
        y_predicted[y_predicted==n_components_novocals+i] = 1

    # Smooth the predictions, change interspersed 0's and 1's using Binary Opening/Closing
    y_predicted = ndimage.binary_closing(y_predicted,structure=np.ones((n_smooth_targets+1))).astype(int)
    y_predicted = ndimage.binary_opening(y_predicted,structure=np.ones((n_smooth_targets+1))).astype(int)

    return (y_predicted,sr,hop_length)
