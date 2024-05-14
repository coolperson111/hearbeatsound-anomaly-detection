import numpy as np


def getTimeFeatures(audio_data):
    mean = np.mean(audio_data)
    std = np.std(audio_data)
    return (
        np.min(audio_data),
        np.max(audio_data),
        mean,
        std,
        np.mean((audio_data - mean) ** 3) / (std**3),  # skewness
        np.mean((audio_data - mean) ** 4) / (std**4),  # kurtosis
        np.median(audio_data),
        np.percentile(audio_data, 75) - np.percentile(audio_data, 25),  # iqr
    )
