import librosa
import numpy as np


def getSpectralCentroid(data):
    sc = librosa.feature.spectral_centroid(y=data, sr=4000)[0]
    return (
        # np.min(sc),
        # np.max(sc),
        # np.sum(sc),
        np.mean(sc),
        np.std(sc),
        # np.median(sc),
    )
