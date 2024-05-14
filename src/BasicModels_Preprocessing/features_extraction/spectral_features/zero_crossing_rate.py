import numpy as np
import librosa

def getZCR(data):
    zcr = librosa.feature.zero_crossing_rate(y=data, sr=4000)[0]
    return (
        np.min(zcr),
        np.max(zcr),
        np.sum(zcr),
        np.mean(zcr),
        np.std(zcr),
        np.median(zcr)
    )
