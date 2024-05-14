"""
This program is to plot some things for the midsem presentation.. the plots in this program should include:
        spectral normal vs abnormal sounds
        spectral murmur vs not
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import librosa
import librosa.display

sys.path.append(os.path.abspath("./src/utils"))
from extractPath import ExtractPath

pn_path = "./datasets/physionet/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data.csv"
pn_df = pd.read_csv(pn_path)

# plot the spectral for normal, abnormal, murmur present, absent, unknown
# normal
normal = pn_df[pn_df["Outcome"] == "Normal"].iloc[0]
normal = pn_df[pn_df["Outcome"] == "Normal"].sample(1)
normal_locs = normal["Recording locations:"].values[0].split("+")
loc = normal_locs[0]
normal_path = ExtractPath(normal, loc)
print("Normal path: ", normal_path)
normal_sound, sr = librosa.load(normal_path, sr=None)

# abnormal
abnormal = pn_df[pn_df["Outcome"] == "Abnormal"].sample(1)
abnormal_locs = abnormal["Recording locations:"].values[0].split("+")
loc = abnormal_locs[np.random.randint(0, len(abnormal_locs))]
abnormal_path = ExtractPath(abnormal, loc)
print("Abnormal path: ", abnormal_path)
abnormal_sound, sr = librosa.load(abnormal_path, sr=None)

# plot the normal sound spectral
