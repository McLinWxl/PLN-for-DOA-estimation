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

# args = args_data_generator()
def cal_fre(signal, fs):
    signal = signal.reshape(-1)
    fft_result = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(len(fft_result), 1 / fs)
    fft_magnitude = np.abs(fft_result) / len(signal)
    single_sided_magnitude = fft_magnitude[:len(fft_magnitude) // 2]
    single_sided_freq = fft_freq[:len(fft_freq) // 2]
    return single_sided_freq, single_sided_magnitude

def fault_generator(args):
    """
    Generate fault signal in time domain
    """
    # print("#" * 30)
    print("Generating impulsive signal...")
    # print(f"Frequency center (Hz): {args.frequency_center}")
    # print(f"Frequency fault (Hz): {args.frequency_fault}")
    # print(f"Damping ratio: {args.damping_ratio}")
    # print(f"Slipping factor: {args.slipping_factor}")
    # print(f"Frequency sampling (Hz): {args.frequency_sampling}")
    # print(f"Time (Seconds): {args.time}")
    # print("#" * 30)
    time = args.time
    frequency_sampling = args.frequency_sampling * args.amp_sample
    if args.fault_type == 0:
        # omega_n = 2 * np.pi * args.frequency_center
        damping_ratio = args.damping_ratio
        frequency_fault = args.frequency_fault
        # omega_d = omega_n * np.sqrt(1 - damping_ratio ** 2)

        t = np.linspace(0, time, int(frequency_sampling * time) + 100, endpoint=False)
        alpha = 2 * np.pi * args.frequency_center * damping_ratio / np.sqrt(1 - damping_ratio ** 2)
        N = int(time * frequency_fault)
        fault_signal = np.zeros_like(t)
        for n in range(N):
            start_time = n / frequency_fault
            impulse = np.exp(-alpha * (t - start_time)) * np.sin(2 * np.pi * args.frequency_center * (t - start_time))
            impulse[t < start_time] = 0
            fault_signal += impulse

        # Generate impulse response function
        # length_filter = frequency_sampling // 5
        # time_filter = np.arange(0, length_filter, 1) / frequency_sampling
        # response_function = (
        #         (omega_n ** 2 * damping_ratio ** 2) * np.exp(-damping_ratio * omega_n * time_filter) * np.sin(
        #     omega_d * time_filter))
        # - (omega_d ** 2) * np.exp(-damping_ratio * omega_n * time_filter) * np.sin(omega_d * time_filter)
        # - (2 * omega_n * omega_d * damping_ratio) * np.exp(-damping_ratio * omega_n * time_filter) * np.cos(
        #     omega_d * time_filter)
        # # response_function = np.exp(-damping_ratio * omega_n * time_filter) * np.sin(omega_d * time_filter)
        #
        # # Generate periodic modulation
        # impulsive_signal = np.zeros(int(time * frequency_sampling))
        # start = int((frequency_sampling / frequency_fault) / 2)
        # while start < time * frequency_sampling:
        #     impulsive_signal[start] = np.random.uniform(0.95, 1.05)
        #     start += int((frequency_sampling / frequency_fault) * (1 + np.random.normal() * slipping_factor))
        # # plt.plot(impulsive_signal[0:1000])
        # # plt.show()
        # # fftfilt
        # signal_fault = np.convolve(impulsive_signal / frequency_sampling, response_function, mode='full')


    # print("Impulsive signal generated.")
    elif args.fault_type == 1:
        duration = time
        fs = frequency_sampling
        num_p = 4
        A_init = 0.1
        A = 0.4
        f_fault = args.frequency_fault
        f_rotation = 17.9
        f_mesh = args.frequency_center
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        fault_signal_order_1 = (1 - A_init * np.cos(2 * np.pi * num_p * f_rotation * t)) * (
                    1 + A * np.cos(2 * np.pi * f_fault * t)) * np.cos(
            2 * np.pi * f_mesh * t + 4 * + A * np.sin(2 * np.pi * f_fault * t))
        fault_signal_order_2 = (1 - A_init * np.cos(2 * np.pi * num_p * f_rotation * t)) * (
                    1 + A * np.cos(2 * np.pi * f_fault * t)) * np.cos(
            4 * np.pi * f_mesh * t + 4 * + A * np.sin(2 * np.pi * f_fault * t))
        fault_signal_order_3 = (1 - A_init * np.cos(2 * np.pi * num_p * f_rotation * t)) * (
                    1 + A * np.cos(2 * np.pi * f_fault * t)) * np.cos(
            6 * np.pi * f_mesh * t + 4 * + A * np.sin(2 * np.pi * f_fault * t))
        fault_signal = fault_signal_order_1 + 0.5 * fault_signal_order_2 + 0.25 * fault_signal_order_3



    fault_signal = min_max_normalize((fault_signal[0:int(time * frequency_sampling)]))

    # x_fre, y_fre = cal_fre(fault_signal, 51200*20)
    # plt.plot(x_fre, y_fre)
    # plt.xlim(0, 10000)
    # plt.show()
    # ...

    return fault_signal, args.frequency_center, args.frequency_fault


def min_max_normalize(signal):
    """
    Normalize signal to [-1, 1]
    :param signal:
    :return:
    """
    return 2 * (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) - 1


def frequency_spectrum(signal, args):
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
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.figure(dpi=500)
    signal_fault_ = fault_generator(args)
    ## add noise and hormanic
    harmonic_f = 1736.3

    SNR = 0
    power_fault = np.sum(signal_fault_**2)
    power_noise = power_fault / (10**(SNR/10))
    noise = np.random.randn(len(signal_fault_))
    power_noise = np.sum(noise**2)
    noise = noise * np.sqrt(power_fault/power_noise)
    signal_fault_ = signal_fault_ + noise
    signal_fault_ = signal_fault_ / np.max(np.abs(signal_fault_))
    signal_fault_ = signal_fault_[::args.amp_sample][:8192]
    x_axis = np.linspace(0, 1500 / args.frequency_sampling, 1500)
    plt.plot(x_axis, signal_fault_[0:1500], label='Impulsive signal')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.title(f'Impulsive signal at: \n '
    #           f'1. Center frequency: {args.frequency_center} Hz, \n '
    #           f'2. Fault frequency: {args.frequency_fault} Hz, \n ')
    # plt.legend(fontsize=8)
    plt.savefig('../../Test/Figures/ImpulsiveSignal.pdf')
    plt.show()

    # plt.figure(dpi=800)
    frequency, spectrum = frequency_spectrum(signal_fault_, args)
    plt.plot(frequency, spectrum, label='Frequency spectrum')
    plt.axvline(x=args.frequency_center, color='red', linestyle='--', label='Center frequency')
    # plt.scatter(args.frequency_center, 1000, marker='+', color='red', s=50)
    # plt.xlim(0, 12000)
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Amplitude')
    # plt.title(f'Frequency spectrum of impulsive signal at: \n '
    #           f'1. Center frequency: {args.frequency_center} Hz, \n '
    #           f'2. Fault frequency: {args.frequency_fault} Hz, \n ')
    # plt.legend(fontsize=8)
    plt.savefig('../../Test/Figures/FrequencySpectrum.pdf')
    plt.show()
