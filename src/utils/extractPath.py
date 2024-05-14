"""
This file is used to extract the path of the sound files for the Physionet dataset.
"""

# import numpy as np
# import pandas as pd


def ExtractPath(df, loc):
    sound_path = "./datasets/physionet/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data/"
    path = sound_path + str(df["Patient ID"].values[0]) + "_" + loc + ".wav"
    return path


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
