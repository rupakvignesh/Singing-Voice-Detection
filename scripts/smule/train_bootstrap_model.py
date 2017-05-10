import numpy as np
import librosa
from sklearn.mixture import GaussianMixture
from feature_extraction import feature_extractor
from sklearn import preprocessing
import soundfile as sf

# Global variables
fs = 16000          # Resample to fs Hertz
frame_length = 512  # window size
hop_length = 256    # frame shift
n_mfcc = 13         # 13 mfcc coefficients
n_components_vocals = 8     # number of vocal mixtures in GMM
n_components_novocals = 8   # number of non vocal mixtures in GMM
n_features = 41             # feature dimension
covariance_type = 'full'

def bootstrap_model(bootstrap_list,target_list):
    # Build models with bootstrap data
    bootstrap_features = np.empty((n_features,0))
    targets = np.empty((0,0))
    print 'Extracting Bootstrap Features'
    for i in range(len(bootstrap_list)):
        print bootstrap_list[i]
        y,sr = sf.read(bootstrap_list[i])
        y = np.mean(y,axis=1)
        y = librosa.core.resample(y,sr,fs)
        temp_bootstrap_features = feature_extractor(y,fs,frame_length,hop_length,n_mfcc)
        bootstrap_features = np.concatenate((bootstrap_features, temp_bootstrap_features),axis=1)
        with open(target_list[i]) as TL:
            temp_targets = np.array(map(float,[lines.rstrip() for lines in TL]))
        temp_targets = np.append(temp_targets,np.zeros(np.shape(temp_bootstrap_features)[1]-len(temp_targets)))
        targets = np.append(targets,temp_targets)

    #Normalize features
    bootstrap_features = bootstrap_features.T
    #bootstrap_features = preprocessing.scale(bootstrap_features.T)

    # Initialize models and do Expectation-Maximization

    print 'Training bootstrap models'
    novocal_clf = GaussianMixture(n_components=n_components_novocals,covariance_type=covariance_type)
    vocal_clf = GaussianMixture(n_components=n_components_vocals,covariance_type=covariance_type)
    novocal_clf.fit(bootstrap_features[targets==0])
    vocal_clf.fit(bootstrap_features[targets==1])

    return (novocal_clf, vocal_clf)
