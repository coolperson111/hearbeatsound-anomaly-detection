import numpy as np
import librosa

def getMels(data):
    mels = librosa.feature.melspectrogram(y=data, sr=4000)
    # get mean of each mfcc, also other statistics
    means = mels.mean(axis=1)
    return means
