"""
This file is used to preprocess the Physionet Dataset,
and store all the necessary mfccs in a single h5 file.
"""

import os

import h5py
import librosa
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from tqdm import tqdm

path = "datasets/physionet/the-circor-digiscope-phonocardiogram-dataset-1.0.3"
df = pd.read_csv(path + "/training_data.csv")


def convertDupLocs(input_list):
    counts = {}
    result = []
    for item in input_list:
        if item not in counts:
            counts[item] = 1
        else:
            counts[item] += 1

    for item in input_list:
        if counts[item] == 0:
            continue
        if counts[item] > 1:
            for i in range(1, counts[item] + 1):
                result.append(item + "_" + str(i))
                counts[item] = 0
        else:
            result.append(item)
    return result


def get_all_files():
    filenames = []  # list of all files in dataset
    outcomes = []  # Normal - 0, Abnormal - 1
    murmurs = []

    for patient_id in df["Patient ID"]:  # loop to fill the list
        # print("Patient ID: ", patient_id)
        patient = df[df["Patient ID"] == patient_id]
        locs = patient["Recording locations:"].values[0].split("+")
        locs = convertDupLocs(locs)  # find duplicate recording locs
        for loc in locs:
            filenames.append(
                path
                + "/training_data/"
                + str(patient["Patient ID"].values[0])
                + "_"
                + loc
                + ".wav"
            )
            if patient["Outcome"].values[0] == "Normal":
                outcomes.append(0)
            else:
                outcomes.append(1)
            murmurs.append(patient["Murmur"].values[0])

    return filenames, outcomes, murmurs


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    y = filtfilt(b, a, data)
    return y


def resize_audio(filename, target_duration, sample_r=None):
    y, sr = librosa.load(filename, sr=sample_r)
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
            mfcc = mfccs[i]
            # split filename to get last patt after /
            # filename = filename.split("/")[-1]
            f.create_dataset(f"{filename}/mfcc", data=mfcc)
            f.create_dataset(f"{filename}/outcome", data=outcome)
            f.create_dataset(f"{filename}/murmur", data=murmur)


def main():
    filenames, outcomes, murmurs = get_all_files()

    print("Number of files: ", len(filenames))
    print("Number of outcomes: ", len(outcomes))
    print("Number of murmurs: ", len(murmurs))
    print("\nFirst 5 files\nfilename\toutcome\tmurmur")
    for i in range(5):
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

        c = 1
        for resized_audio in resized_audios:
            final_audios.append(resized_audio)
            new_filenames.append(filenames[i].split("/")[-1] + "_s" + str(c))
            new_outcomes.append(outcomes[i])
            new_murmurs.append(murmurs[i])
            c += 1
            mfcc = librosa.feature.mfcc(y=resized_audio, sr=sampling_rate, n_mfcc=25)
            mfccs.append(mfcc)

    print("\nLength of final_audios: ", len(final_audios))
    print("Shape of mfccs: ", np.array(mfccs).shape)
    print("\n")

    h5path = "mfccs_10s_butter_2000hz.h5"
    save_mfccs_to_h5(h5path, new_filenames, new_outcomes, new_murmurs, mfccs)


if __name__ == "__main__":
    main()
