"""
This file is used to generate frequency domain snapshots of the antenna array.
Including the following steps:
0. Apply time delay to the signal based on the antenna array configuration.
1. Slice the time domain signal into multiple segments. Each segment is a snapshot.
2. Perform FFT on each segment.
3. Find the center frequency of the signal and the N-largest frequency components near the center frequency.
4. Generate the frequency domain snapshot based on the N-largest frequency components.
"""
from typing import Tuple, Any

import numpy as np
import scipy
import matplotlib.pyplot as plt
from numpy import ndarray, dtype
from __init__ import args_data_generator
args = args_data_generator()


def time_delay(signal, time_delay: float, frequency_sampling=args.frequency_sampling) -> np.ndarray:
    """
    Apply time delay to the signal
    :param signal: the signal to be delayed
    :param time_delay: the time delay
    :param frequency_sampling: the frequency of the signal
    :return: the delayed signal
    """
    length_signal = len(signal)
    time_sequence = np.arange(0, length_signal, 1) / frequency_sampling
    time_delay_index = int(time_delay * frequency_sampling)
    signal_delayed = np.zeros(length_signal)
    if time_delay_index >= 0:
        signal_delayed[:length_signal - time_delay_index] = signal[time_delay_index:]
        signal_delayed[length_signal - time_delay_index:] = signal[:time_delay_index]
    else:
        signal_delayed[-time_delay_index:] = signal[:time_delay_index]
        signal_delayed[:-time_delay_index] = signal[time_delay_index:]
    return signal_delayed


def delay_time_calculator(antenna_distance: float, theta: float, speed_of_sound=args.speed_of_sound) -> float:
    """
    Calculate the time delay based on the antenna array configuration
    :param antenna_distance: the distance between two adjacent antennas
    :param theta: the angle of the signal
    :param speed_of_sound: the speed of sound
    :return: the time delay
    """
    return antenna_distance * np.sin(theta) / speed_of_sound

def signal_slicer(signal: np.ndarray, snapshot_length=args.snapshot_length, overlap=args.snapshot_length//2) -> np.ndarray:
    """
    Slice the signal into multiple segments
    :param signal: the signal to be sliced
    :param snapshot_length: the length of each segment
    :param overlap: the overlap between two adjacent segments
    :return: the sliced signal
    """
    length_signal = len(signal)
    num_snapshots = (length_signal - snapshot_length) // (snapshot_length - overlap) + 1
    signal_sliced = np.zeros((num_snapshots, snapshot_length))
    for i in range(num_snapshots):
        signal_sliced[i] = signal[i * (snapshot_length - overlap): i * (snapshot_length - overlap) + snapshot_length]
    return signal_sliced


def fft_transform(signal: np.ndarray, frequency_sampling=args.frequency_sampling, max_fre = 10000):
    """
    Perform FFT on the signal
    :param signal: the signal to be transformed
    :param frequency_sampling: the frequency of the signal
    :return: the frequency domain signal
    """
    frequency = np.fft.fftfreq(len(signal), 1 / frequency_sampling)
    spectrum = np.fft.fft(signal)
    spectrum[0] = 0
    frequency_out = [freq for freq in frequency if max_fre > freq > 0]
    frequency_out = np.array(frequency_out).reshape(-1)
    spectrum_out = np.abs(spectrum[:len(frequency_out)])
    return frequency_out, spectrum_out


def find_n_largest_frequency_components(frequency_fft, signal_fft: np.ndarray, frequency_center=args.frequency_center, frequency_band=1000, n=args.search_numbers) -> \
        tuple[Any, ndarray[Any, Any]]:
    """
    Find the N-largest frequency components near the center frequency
    :param signal_fft: the frequency domain signal
    :param frequency_center: the center frequency of the signal
    :param n: the number of frequency components
    :return: the N-largest frequency components
    """
    signal_fft_ = signal_fft[np.where((frequency_fft > frequency_center - frequency_band) & (frequency_fft < frequency_center + frequency_band))]
    frequency_fft_ = frequency_fft[np.where((frequency_fft > frequency_center - frequency_band) & (frequency_fft < frequency_center + frequency_band))]
    frequency_index = scipy.signal.find_peaks(signal_fft_, distance=args.frequency_fault//10)[0]
    frequency_index_ = frequency_index[np.argsort(signal_fft_[frequency_index])[-n:]]
    frequency_index_ = frequency_index_[::-1]
    return frequency_fft_[frequency_index_], signal_fft_[frequency_index_]




if __name__ == '__main__':
    from TimeDomainGenerator import fault_generator
    from __init__ import args_data_generator
    args = args_data_generator()
    signal = fault_generator(args.frequency_center, args.frequency_fault, args.damping_ratio, args.slipping_factor,
                             args.time, args.frequency_sampling, args.SNR)
    time_dalays = [delay_time_calculator(args.antenna_distance * i, 30) for i in range(args.anrenna_num)]
    signals_delayed = [time_delay(signal, time_dalay) for time_dalay in time_dalays]
    signals_delayed = np.array(signals_delayed).reshape(args.anrenna_num, -1)

    # plot the time domain signal
    for i in range(args.anrenna_num):
        signal_sliced = [signal_slicer(signals_delayed[i]) for i in range(args.anrenna_num)]
    signal_sliced = np.array(signal_sliced).reshape(args.anrenna_num, -1, args.snapshot_length)

    # signal_sliced = signal_slicer(signal, 8192, 4096)
    virtual_index = 12000
    sample_index = 0
    plt.figure()
    plt.plot(signal_sliced[0, sample_index, :virtual_index], label='Antenna 1')
    plt.plot(signal_sliced[1, sample_index, :virtual_index], label='Antenna 2')
    plt.plot(signal_sliced[2, sample_index, :virtual_index], label='Antenna 3')
    plt.plot(signal_sliced[3, sample_index, :virtual_index], label='Antenna 4')
    plt.plot(signal_sliced[4, sample_index, :virtual_index], label='Antenna 5')
    plt.plot(signal_sliced[5, sample_index, :virtual_index], label='Antenna 6')
    plt.plot(signal_sliced[6, sample_index, :virtual_index], label='Antenna 7')
    plt.plot(signal_sliced[7, sample_index, :virtual_index], label='Antenna 8')
    plt.legend()
    plt.show()

    signal_sample = signal_sliced[0, 0]
    # plot the frequency domain signal
    frequency_fft, signals_fft = fft_transform(signal_sample)
    plt.figure()
    plt.plot(frequency_fft, signals_fft, label='Antenna 1')
    plt.title('Frequency Domain Signal Sample')
    plt.show()

    # find the N-largest frequency components

    n_largest_frequency, n_largest_amp = find_n_largest_frequency_components(frequency_fft, signals_fft)
    print(f"The N-largest frequency components are {n_largest_frequency}")

