import numpy as np
import librosa

def getMFCCs(data):
    mfccs = librosa.feature.mfcc(y=data, sr=4000, n_mfcc=25)
    # get mean of each mfcc, also other statistics
    means = mfccs.mean(axis=1)
    return means
