import numpy as np
import librosa

def getSpectralRolloff(data):
    sr = librosa.feature.spectral_rolloff(y=data, sr=4000)[0]
    return (
        # np.min(sr),
        # np.max(sr),
        # np.sum(sr),
        np.mean(sr),
        np.std(sr),
        # np.median(sr)
    )
