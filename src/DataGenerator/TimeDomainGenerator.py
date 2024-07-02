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
from __init__ import args_data_generator

args = args_data_generator()


def fault_generator():
    """
    Generate fault signal in time domain
    """
    print("#" * 30)
    print("Generating impulsive signal...")
    print(f"Frequency center (Hz): {args.frequency_center}")
    print(f"Frequency fault (Hz): {args.frequency_fault}")
    print(f"Damping ratio: {args.damping_ratio}")
    print(f"Slipping factor: {args.slipping_factor}")
    print(f"Frequency sampling (Hz): {args.frequency_sampling}")
    print(f"Time (Seconds): {args.time}")
    print("#" * 30)

    omega_n = 2 * np.pi * args.frequency_center
    damping_ratio = args.damping_ratio
    slipping_factor = args.slipping_factor
    frequency_sampling = args.frequency_sampling
    time = args.time
    frequency_fault = args.frequency_fault
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
        impulsive_signal[start] = np.random.uniform(0.95, 1.05)
        start += int((frequency_sampling / frequency_fault) * (1 + np.random.normal() * slipping_factor))
    # plt.plot(impulsive_signal[0:1000])
    # plt.show()
    # fftfilt
    signal_fault = np.convolve(impulsive_signal / frequency_sampling, response_function, mode='full')
    signal_fault = min_max_normalize((signal_fault[0:int(time * frequency_sampling)]))
    print("Impulsive signal generated.")
    return signal_fault


def min_max_normalize(signal):
    """
    Normalize signal to [-1, 1]
    :param signal:
    :return:
    """
    return 2 * (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) - 1


def frequency_spectrum(signal):
    """
    Generate frequency spectrum of a signal
    :param signal:
    :param frequency_sampling:
    :return:
    """
    frequency_sampling = args.frequency_sampling
    length_signal = len(signal)
    frequency = np.fft.fftfreq(length_signal, 1 / frequency_sampling)
    spectrum = np.fft.fft(signal)
    spectrum[0] = 0
    return np.abs(frequency), np.abs(spectrum)


if '__main__' == __name__:
    from __init__ import args_data_generator

    args = args_data_generator()

    signal_fault_ = fault_generator()
    plt.plot(signal_fault_[0:1000])
    plt.show()
    frequency, spectrum = frequency_spectrum(signal_fault_)
    plt.scatter(args.frequency_center, 1000, marker='+', color='red', s=50)
    plt.plot(frequency, spectrum)
    plt.show()
