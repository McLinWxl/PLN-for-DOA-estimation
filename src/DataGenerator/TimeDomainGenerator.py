"""
This file is used to generate fault signal in time domain, reference to the following paper:
@article{buzzoni2020tool,
  title={A tool for validating and benchmarking signal processing techniques applied to machine diagnosis},
  author={Buzzoni, Marco and D’Elia, Gianluca and Cocconcelli, Marco},
  journal={Mechanical Systems and Signal Processing},
  volume={139},
  pages={106618},
  year={2020},
  publisher={Elsevier}
}
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy


def fault_generator(frequency_nature: float, frequency_fault: float, damping_ratio: float, slipping_factor: float,
                    time: float, frequency_sampling: float, SNR: float):
    """
    Generate a fault signal
    :param slipping_factor:
    :param frequency_nature: aka. center frequency
    :param frequency_fault: aka. offset frequency
    :param damping_ratio:
    :param time:
    :param frequency_sampling:
    :return:
    """
    omega_n = 2 * np.pi * frequency_nature
    omega_d = omega_n * np.sqrt(1 - damping_ratio ** 2)

    # Generate impulse response function
    length_filter = frequency_sampling // 5
    time_filter = np.arange(0, length_filter, 1) / frequency_sampling
    response_function = (
            (omega_n ** 2 * damping_ratio ** 2) * np.exp(-damping_ratio * omega_n * time_filter) * np.sin(
        omega_d * time_filter))
    - (omega_d ** 2) * np.exp(-damping_ratio * omega_n * time_filter) * np.sin(omega_d * time_filter)
    - (2 * omega_n * omega_d * damping_ratio) * np.exp(-damping_ratio * omega_n * time_filter) * np.cos(
        omega_d * time_filter)
    # response_function = np.exp(-damping_ratio * omega_n * time_filter) * np.sin(omega_d * time_filter)

    # Generate periodic modulation
    impulsive_signal = np.zeros(int(time * frequency_sampling))
    start = 0
    while start < time * frequency_sampling:
        impulsive_signal[start] = 1
        start += int((frequency_sampling / frequency_fault) * (1 + np.random.normal() * slipping_factor))
    # plt.plot(impulsive_signal[0:1000])
    # plt.show()
    # fftfilt
    signal_fault = np.convolve(impulsive_signal / frequency_sampling, response_function, mode='full')
    signal_fault = min_max_normalize(signal_fault[0:int(time * frequency_sampling)])
    signal_generated = add_noise(signal_fault, SNR)
    return (signal_generated)


def add_noise(signal, SNR):
    '''
    Add noise to signal
    :param signal:
    :param SNR:
    :return:
    '''
    signal_power = np.sum(signal ** 2) / len(signal)
    noise_power = signal_power / (10 ** (SNR / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise


def min_max_normalize(signal):
    """
    Normalize signal to [-1, 1]
    :param signal:
    :return:
    """
    return 2 * (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) - 1


def frequency_spectrum(signal, frequency_sampling):
    """
    Generate frequency spectrum of a signal
    :param signal:
    :param frequency_sampling:
    :return:
    """
    length_signal = len(signal)
    frequency = np.fft.fftfreq(length_signal, 1 / frequency_sampling)
    spectrum = np.fft.fft(signal)
    spectrum[0] = 0
    return np.abs(frequency), np.abs(spectrum)


if '__main__' == __name__:
    from __init__ import args_data_generator

    args = args_data_generator()
    frequency_center = args.frequency_center
    frequency_fault = args.frequency_fault
    damping_ratio = args.damping_ratio
    time = args.time
    frequency_sampling = args.frequency_sampling
    slipping_factor = args.slipping_factor
    SNR = args.SNR

    signal_fault_ = fault_generator(frequency_center, frequency_fault, damping_ratio, slipping_factor, time,
                                    frequency_sampling, SNR)
    plt.plot(signal_fault_[0:1000])
    plt.show()
    frequency, spectrum = frequency_spectrum(signal_fault_, frequency_sampling)
    plt.scatter(frequency_center, 1000, marker='+', color='red', s=50)
    plt.plot(frequency[0:10000], spectrum[0:10000])
    plt.show()
