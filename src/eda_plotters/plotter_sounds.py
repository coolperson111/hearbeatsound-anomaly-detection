"""
This program is to plot some things for the midsem presentation.. the plots in this program should include:
        normal vs abnormal sounds
        murmur vs not
"""

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath("./src/utils"))

from extractPath import ExtractPath

pn_path = "./datasets/physionet/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data.csv"
pn_df = pd.read_csv(pn_path)
# sound_path = "./datasets/physionet/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data/"

# plot the sounds for normal, abnormal, murmur present, absent, unknown
# normal
normal = pn_df[pn_df["Outcome"] == "Normal"]
normal = normal.sample(1)

normal_locs = normal["Recording locations:"].values[0].split("+")
loc = normal_locs[np.random.randint(0, len(normal_locs))]
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

# plot the normal sound 
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(normal_sound)
plt.title("Normal sound")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.subplot(2, 1, 2)
plt.plot(abnormal_sound)
plt.title("Abnormal sound")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()


# murmur present vs absent
# murmur present
murmur = pn_df[pn_df["Murmur"] == "Present"].sample(1)
murmur_locs = murmur["Recording locations:"].values[0].split("+")
loc = murmur_locs[np.random.randint(0, len(murmur_locs))]
murmur_path = ExtractPath(murmur, loc)
print("Murmur path: ", murmur_path)
murmur_sound, sr = librosa.load(murmur_path, sr=None)

# murmur absent
no_murmur = pn_df[pn_df["Murmur"] == "Absent"].sample(1)
no_murmur_locs = no_murmur["Recording locations:"].values[0].split("+")
loc = no_murmur_locs[np.random.randint(0, len(no_murmur_locs))]
no_murmur_path = ExtractPath(no_murmur, loc)
print("No murmur path: ", no_murmur_path)
no_murmur_sound, sr = librosa.load(no_murmur_path, sr=None)

# plot the murmur sounds
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(murmur_sound)
plt.title("Murmur sound")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.subplot(2, 1, 2)
plt.plot(no_murmur_sound)
plt.title("No murmur sound")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()
