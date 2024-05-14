"""
Aggregator function to extract all features
and store in a csv
"""

import os
import sys
import time

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath("./src/utils"))
sys.path.append(os.path.abspath("./src/features_extraction"))
from extractPath import ExtractPath, convertDupLocs
from time_features import getTimeFeatures
from spectral_features.spectral_centroid import getSpectralCentroid
from spectral_features.spectral_bandwidth import getSpectralBandwidth
from spectral_features.spectral_rolloff import getSpectralRolloff
from spectral_features.mfcc import getMFCCs
from spectral_features.mels import getMels

pn_path = "./datasets/physionet/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data.csv"
num_files = 3163

def main():
    pn_df = pd.read_csv(pn_path)

    # create a new df to store the features
    new_df = pd.DataFrame(
        columns=[
            "Patient ID",
            "filename",
            "Recording location",
            # "Sound",
            "SR",
            "Duration",
            "Murmur",
            "Outcome",
            "min",
            "max",
            "mean",
            "std",
            "skewness",
            "kurtosis",
            "median",
            "iqr",
            "sc_mean", "sc_std",
            "sb_mean", "sb_std",
            "sr_mean", "sr_std",
            "mels1", "mels2", "mels3", "mels4", "mels5", "mels6", "mels7", "mels8", "mels9", "mels10", "mels11", "mels12", "mels13", "mels14", "mels15", "mels16", "mels17", "mels18", "mels19", "mels20", "mels21", "mels22", "mels23", "mels24", "mels25", "mels26", "mels27", "mels28", "mels29", "mels30", "mels31", "mels32", "mels33", "mels34", "mels35", "mels36", "mels37", "mels38", "mels39", "mels40", "mels41", "mels42", "mels43", "mels44", "mels45", "mels46", "mels47", "mels48", "mels49", "mels50", "mels51", "mels52", "mels53", "mels54", "mels55", "mels56", "mels57", "mels58", "mels59", "mels60", "mels61", "mels62", "mels63", "mels64", "mels65", "mels66", "mels67", "mels68", "mels69", "mels70", "mels71", "mels72", "mels73", "mels74", "mels75", "mels76", "mels77", "mels78", "mels79", "mels80", "mels81", "mels82", "mels83", "mels84", "mels85", "mels86", "mels87", "mels88", "mels89", "mels90", "mels91", "mels92", "mels93", "mels94", "mels95", "mels96", "mels97", "mels98", "mels99", "mels100", "mels101", "mels102", "mels103", "mels104", "mels105", "mels106", "mels107", "mels108", "mels109", "mels110", "mels111", "mels112", "mels113", "mels114", "mels115", "mels116", "mels117", "mels118", "mels119", "mels120", "mels121", "mels122", "mels123", "mels124", "mels125", "mels126", "mels127", "mels128",
            "mfcc1", "mfcc2", "mfcc3", "mfcc4", "mfcc5", "mfcc6", "mfcc7", "mfcc8", "mfcc9", "mfcc10", "mfcc11", "mfcc12", "mfcc13", "mfcc14", "mfcc15", "mfcc16", "mfcc17", "mfcc18", "mfcc19", "mfcc20", "mfcc21", "mfcc22", "mfcc23", "mfcc24", "mfcc25",
        ]
    )

    i = 0
    start_time = time.time()
    # iterate through patient id
    for patient_id in pn_df["Patient ID"]:
        # print("Patient ID: ", patient_id)
        patient = pn_df[pn_df["Patient ID"] == patient_id]
        locs = patient["Recording locations:"].values[0].split("+")
        locs = convertDupLocs(locs)  # find duplicate recording locs
        for loc in locs:
            # extract path

            path = ExtractPath(patient, loc)
            sound, sr = librosa.load(path, sr=None)
            duration = len(sound) / sr
            time_feats = getTimeFeatures(sound)
            sc = getSpectralCentroid(sound)
            sb = getSpectralBandwidth(sound)
            sr = getSpectralRolloff(sound)
            mels = getMels(sound)
            mfcc = getMFCCs(sound)


            features = (
                    patient_id,
                    path[path.rfind("/") + 1 :],
                    loc,
                    sr,
                    duration,
                    patient["Murmur"].values[0],
                    patient["Outcome"].values[0],
                    )
            # append time features
            features = features + time_feats + sc + sb + sr + tuple(mels) + tuple(mfcc)

            # print("features len: ", len(features))
            new_df.loc[len(new_df)] = features 
            
            i = i+1

            progress = i/num_files * 100

            elapsed_time = time.time() - start_time
            remaining_time = (elapsed_time / (i)) * (num_files - i - 2)

            bar_length = 20
            filled = int(progress / (100 / bar_length))
            bar = '█' * filled + '-' * (bar_length - filled)

            # print(f"\rPatient ID: {patient_id}  Training Progress: [{'#' * int(progress / 5)}{'-' * (20 - int(progress / 5))}] {progress:.1f}%", end="", flush=True)
            sys.stdout.write(f"\rPatient ID: {patient_id} | Training Progress: |{bar}| {progress:.1f}% | Elapsed: {elapsed_time:.2f}s | Remaining: {remaining_time/60:.2f}mins")
            sys.stdout.flush()

        """
        if i>100:
            break
        """
        

    # print("len of df cols: ", len(new_df.columns))
    # print("new_df: ", new_df)
    
    # save to csv
    new_df.to_csv("data/full_data.csv", index=False)


if __name__ == "__main__":
    main()
