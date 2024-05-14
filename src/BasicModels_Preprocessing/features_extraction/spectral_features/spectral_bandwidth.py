import numpy as np
import librosa

def getSpectralBandwidth(data):
    sb = librosa.feature.spectral_bandwidth(y=data, sr=4000)[0]
    return (
        # np.min(sb),
        # np.max(sb),
        # np.sum(sb),
        np.mean(sb),
        np.std(sb),
        # np.median(sb)
    )
