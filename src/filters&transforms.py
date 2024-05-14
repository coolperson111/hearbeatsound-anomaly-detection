"""
THis script was used to try various filters and trasformation 
measures and compare their results
"""

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pywt
import sounddevice as sd
from scipy.signal import butter, filtfilt
from scipy.fft import fft


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def dwt_denoise(data, wavelet="db6", level=1):
    coeffs = pywt.wavedec(data, wavelet, mode="per", level=level)
    threshold = np.sqrt(2 * np.log(len(data))) * np.median(
        np.abs(coeffs[-level]) / 0.6745
    )
    coeffs[1:] = (pywt.threshold(i, value=threshold, mode="soft") for i in coeffs[1:])
    reconstructed_signal = pywt.waverec(coeffs, wavelet, mode="per")
    return reconstructed_signal


def svd_denoise(data, window_size, rank):
    shape = (data.size - window_size + 1, window_size)
    strides = (data.strides[0], data.strides[0])
    X = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    U, s, V = np.linalg.svd(X, full_matrices=False)
    s[rank:] = 0
    denoised_data = np.dot(U, np.dot(np.diag(s), V))
    return np.mean(denoised_data, axis=1)

def calculate_snr(clean_signal, noisy_signal):
    signal_power = np.sum(clean_signal ** 2)
    noise_power = np.sum((clean_signal - noisy_signal) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def calculate_rmse(clean_signal, denoised_signal):
    return np.sqrt(np.mean((clean_signal - denoised_signal) ** 2))

def plot_spectrum(original_signal, denoised_signal, fs):
    orig_fft = fft(original_signal)
    denoised_fft = fft(denoised_signal)
    
    freq = np.linspace(0, fs, len(original_signal))
    
    plt.figure(figsize=(12, 6))
    plt.plot(freq, np.abs(orig_fft), label='Original Signal', alpha=0.5)
    plt.plot(freq, np.abs(denoised_fft), label='Denoised Signal', alpha=0.7)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title('Spectrum Comparison')
    plt.xlim([0, fs / 2])  # Only plot up to Nyquist frequency
    plt.show()

def play_sound(data, filtered_data, fs):
    print("Playing original data...")
    sd.play(data, fs)
    sd.wait()
    print("Playing filtered data...")
    sd.play(filtered_data, fs)
    sd.wait()  

def main():
    lowcut = 25.0  # Low cutoff frequency
    highcut = 250.0  # High cutoff frequency

    window_size = 50  # Example window size
    rank = 10  # Example rank

    path = "./datasets/physionet/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data/9979_AV.wav"
    data, fs = librosa.load(path, sr=None)

    filtered_data = butter_bandpass_filter(data, lowcut, highcut, fs, order=5)
    svd_denoised_data = svd_denoise(filtered_data, window_size, rank)
    dwt_denoised_data = dwt_denoise(filtered_data, wavelet="db6", level=1)
    print("Original data length: ", len(data))
    print("Filtered data length: ", len(filtered_data))
    print("DWT denoised data length: ", len(dwt_denoised_data))
    print("SVD denoised data length: ", len(svd_denoised_data))

    """
    play_sound(data, dwt_denoised_data, fs)
    play_sound(data, svd_denoised_data, fs)
    """

    # Plot the original and filtered signals
    plt.figure(1)
    plt.subplot(311)
    plt.plot(data)
    plt.title("Original Signal")
    plt.subplot(312)
    plt.plot(dwt_denoised_data)
    plt.title("DWT denoised Signal")
    plt.subplot(313)
    plt.plot(svd_denoised_data)
    plt.title("SVD denoised Signal")
    plt.show()

    snr_dwt = calculate_snr(data, dwt_denoised_data)
    # snr_svd = calculate_snr(data, svd_denoised_data)


    rmse_dwt = calculate_rmse(data, dwt_denoised_data)
    # rmse_svd = calculate_rmse(data, svd_denoised_data)

    print("SNR DWT: ", snr_dwt)
    # print("SNR SVD: ", snr_svd)
    print("RMSE DWT: ", rmse_dwt)
    # print("RMSE SVD: ", rmse_svd)


if __name__ == "__main__":
    main()
