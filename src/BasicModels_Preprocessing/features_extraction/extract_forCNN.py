import librosa
import numpy as np
import matplotlib.pyplot as plt

pn_path = "./datasets/physionet/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data/13918_AV.wav"
# Load the audio file
audio_file = pn_path  # Replace 'your_audio_file.wav' with the path to your audio file
y, sr = librosa.load(audio_file, sr=None)  # Load audio without resampling

# Compute MFCCs
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=25)  # Compute MFCCs with 13 coefficients

# Display the shape of the MFCCs matrix
print("MFCCs shape:", mfccs.shape)

# Optionally, visualize the MFCCs
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()
