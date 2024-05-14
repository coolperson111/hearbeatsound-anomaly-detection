"""
This file is used to preprocess the Plaksha Heart Dataset,
and store all the necessary mfccs in a single h5 file.
"""

import os

import h5py
import librosa
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from tqdm import tqdm

path = "./datasets/Plaksha_Heart_Dataset"


def get_all_files():
    filenames = os.listdir(path)
    outcomes = []  # Normal - 0, Abnormal - 1
    murmurs = []
    for i in filenames:
        if("Abnormal" in i):
            outcomes.append(1)
        else:
            outcomes.append(0)
        murmurs.append("Unknown")

    return filenames, outcomes, murmurs


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    y = filtfilt(b, a, data)
    return y


def resize_audio(filename, target_duration, sample_r=None):
    y, sr = librosa.load(path + "/" + filename, sr=sample_r)
    # apply butterworth bandpass filter
    y = butter_bandpass_filter(y, 25, 450, sr, order=2)
    # z normalize
    # y = (y - np.mean(y)) / np.std(y)
    duration = librosa.get_duration(y=y, sr=sr)

    # divide into 10s segments - as many as possible
    # if the leftover part is less than 5s, discard it
    # if the leftover part is more than 5s, pad it with zeros till 10s and keep
    resized_audios = []
    for i in range(int(duration // target_duration)):
        start = i * target_duration
        end = (i + 1) * target_duration
        resized_audio = y[int(start * sr) : int(end * sr)]
        resized_audios.append(resized_audio)

    # check if there is any leftover part
    leftover = duration % target_duration
    if leftover >= target_duration / 2:
        start = int((duration // target_duration) * target_duration)
        end = int(duration)
        resized_audio = y[int(start * sr) : int(end * sr)]
        # pad with zeros
        zeros = np.zeros(int(target_duration * sr - len(resized_audio)))
        resized_audio = np.concatenate((resized_audio, zeros))
        resized_audios.append(resized_audio)

    return resized_audios



def save_mfccs_to_h5(h5path, filenames, outcomes, murmurs, mfccs):
    print("Saving mfccs to ", h5path, "...")
    with h5py.File(h5path, "w") as f:
        for i in tqdm(range(len(filenames))):
            filename = filenames[i]
            outcome = outcomes[i]
            murmur = murmurs[i]
            mfccs_temp = mfccs[i]
            # split filename to get last patt after /
            # filename = filename.split("/")[-1]

            for i in range(len(mfccs_temp)):
                mfcc = mfccs_temp[i]

                f.create_dataset(f"{filename}/{i+1}/mfcc", data=mfcc)
                f.create_dataset(f"{filename}/{i+1}/outcome", data=outcome)
                f.create_dataset(f"{filename}/{i+1}/murmur", data=murmur)


def main():
    filenames, outcomes, murmurs = get_all_files()

    print("Number of files: ", len(filenames))
    print("Number of outcomes: ", len(outcomes))
    print("Number of murmurs: ", len(murmurs))
    print("\nFiles\nfilename\toutcome\tmurmur")
    for i in range(14):
        print(f"{filenames[i]}\t{outcomes[i]}\t{murmurs[i]}")
    print("\n")

    target_duration = 10
    sampling_rate = 2000

    final_audios = []
    mfccs = []
    new_filenames = []
    new_outcomes = []
    new_murmurs = []

    print("Resizing audios (and getting their mfccs)...")
    for i in tqdm(range(len(filenames))):
        resized_audios = resize_audio(
            filenames[i], target_duration, sample_r=sampling_rate
        )
        final_audios.append(resized_audios)
        mfccs_temp = []
        for resized_audio in resized_audios:
            mfcc = librosa.feature.mfcc(y=resized_audio, sr=sampling_rate, n_mfcc=25)
            mfccs_temp.append(mfcc)
        mfccs.append(mfccs_temp)


    print("len(final_audios): ", len(final_audios))
    print("len(mfccs): ", len(mfccs))
    print("len(final_audios[0]): ", len(final_audios[0]))

    h5path = "data/spectograms/mfccs_plaksha_butter_10s_2000hz.h5"
    save_mfccs_to_h5(h5path, filenames, outcomes, murmurs, mfccs)


if __name__ == "__main__":
    main()
