# Extracts Spectral features from audio files
# Features include MFCCs, their acceleration
# and velocity coefficients, Spectral Centroid and Zero-crossing rate
import numpy as np
import librosa

def feature_extractor(y, sr, frame_length, hop_length, n_mfcc):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,n_fft=frame_length,hop_length=hop_length)
    mfccs_del = librosa.feature.delta(mfccs)
    mfccs_del_del = librosa.feature.delta(mfccs_del, order=2)
    spec_centroid = librosa.feature.spectral_centroid(y,sr,n_fft=frame_length,hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)
    features = np.concatenate((mfccs,mfccs_del,mfccs_del_del,spec_centroid,zcr),axis=0)
    return features
