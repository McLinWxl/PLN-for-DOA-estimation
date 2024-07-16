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
from rich.progress import track
import numpy as np
import scipy
import matplotlib.pyplot as plt
from numpy import ndarray, dtype
from __init__ import args_data_generator

# args = args_data_generator()
from TimeDomainGenerator import min_max_normalize


def add_noise(signal:np.ndarray, SNR):
    """
    Add noise to signal
    :param signal:
    :param SNR:
    :return:
    """
    signal_power = np.sum(signal ** 2) / signal.size
    noise_power = signal_power / (10 ** (SNR / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise




def time_delay(signal, time_delay: float, args) -> np.ndarray:
    """
    Apply time delay to the signal
    :param signal: the signal to be delayed
    :param time_delay: the time delay
    :param frequency_sampling: the frequency of the signal
    :return: the delayed signal
    """
    frequency_sampling = args.frequency_sampling
    length_signal = len(signal)
    # time_sequence = np.arange(0, length_signal, 1) / frequency_sampling
    time_delay_index = int(time_delay * frequency_sampling * args.amp_sample)
    signal_delayed = np.zeros(length_signal)
    if time_delay_index >= 0:
        signal_delayed[:length_signal - time_delay_index] = signal[time_delay_index:]
        signal_delayed[length_signal - time_delay_index:] = signal[:time_delay_index]
    else:
        signal_delayed[-time_delay_index:] = signal[:time_delay_index]
        signal_delayed[:-time_delay_index] = signal[time_delay_index:]
    return signal_delayed


def delay_time_calculator(antenna_distance: float, theta: float, args) -> float:
    """
    Calculate the time delay based on the antenna array configuration
    :param antenna_distance: the distance between two adjacent antennas
    :param theta: the angle of the signal
    :param speed_of_sound: the speed of sound
    :return: the time delay
    """
    speed_of_sound = args.speed_of_sound
    theta_ = np.deg2rad(theta)
    return antenna_distance * np.sin(theta_) / speed_of_sound


def signal_slicer(signal: np.ndarray, args) -> np.ndarray:
    """
    Slice the signal into multiple segments
    :param signal: the signal to be sliced
    :param snapshot_length: the length of each segment
    :param overlap: the overlap between two adjacent segments
    :return: the sliced signal
    """
    snapshot_length = args.snapshot_length
    overlap = args.snapshot_length // 2
    length_signal = len(signal)
    num_snapshots = (length_signal - snapshot_length) // (snapshot_length - overlap) + 1
    signal_sliced = np.zeros((num_snapshots, snapshot_length))
    for i in range(num_snapshots):
        signal_sliced[i] = signal[i * (snapshot_length - overlap): i * (snapshot_length - overlap) + snapshot_length]
    return signal_sliced


def fft_transform(signal: np.ndarray, args, max_fre=15000):
    """
    Perform FFT on the signal
    :param signal: the signal to be transformed
    :param frequency_sampling: the frequency of the signal
    :return: the frequency domain signal
    """
    frequency_sampling = args.frequency_sampling
    frequency = np.fft.fftfreq(len(signal), 1 / frequency_sampling)
    spectrum = np.fft.fft(signal)
    spectrum[0] = 0
    frequency_out = [freq for freq in frequency if max_fre > freq > 0]
    frequency_out = np.array(frequency_out).reshape(-1)
    spectrum_out = (spectrum[:len(frequency_out)])
    return frequency_out, spectrum_out


def find_largest_frequency_components(frequency_fft, signal_fft: np.ndarray, frequency_center, frequency_band, args) -> \
        tuple[Any, ndarray[Any, Any]]:
    """
    Find the N-largest frequency components near the center frequency
    :param frequency_band:
    :param frequency_fft: the frequency x-axis
    :param signal_fft: the frequency domain signal
    :param frequency_center: the center frequency of the signal
    :return: the N-largest frequency components, and their indexes
    """
    n = 1
    frequency_band = frequency_band / 2
    signal_fft_ = signal_fft[np.where(
        (frequency_fft > frequency_center - frequency_band) & (frequency_fft < frequency_center + frequency_band))]
    frequency_fft_ = frequency_fft[np.where(
        (frequency_fft > frequency_center - frequency_band) & (frequency_fft < frequency_center + frequency_band))]
    frequency_index = scipy.signal.find_peaks(np.abs(signal_fft_))[0]
    frequency_index__ = frequency_index[np.argsort(np.abs(signal_fft_[frequency_index]))[-n:]]
    frequency_index_ = frequency_index__[::-1]

    plt.plot(frequency_fft, np.abs(signal_fft))
    plt.show()
    plt.plot(frequency_fft_, np.abs(signal_fft_))
    plt.show()

    if signal_fft_[frequency_index_].shape[0] == 0:
        if signal_fft_[0] > signal_fft_[-1]:
            frequency_index_ = [0]
        else:
            frequency_index_ = [len(signal_fft_) - 1]
    ...
    # frequency_index_ = [12]
    return frequency_fft_[frequency_index_], signal_fft_[frequency_index_]


def snapshot_exactor(signal, args):
    """
    main function:

    :return: The concatenated frequency domain snapshots and the target frequency components
    """
    # print("Generating frequency domain snapshots...")
    print(f"Frequency Center: {args.frequency_center}")
    # print(f"Theta range: {args.theta_min} ~ {args.theta_max}")
    # print(f"SNR range: {args.SNR_min} ~ {args.SNR_max}")
    # print(f"Number of samples: {args.samples}")
    # print(f"Number of snapshots: {args.num_snapshots}")
    # # print(f"Available samples
    # print(f"Antenna number: {args.antenna_num}")
    # print("#" * 30)
    theta_min = args.theta_min
    theta_max = args.theta_max
    SNR_source_min = args.SNR_source_min
    SNR_source_max = args.SNR_source_max
    SNR_env_min = args.SNR_env_min
    SNR_env_max = args.SNR_env_max
    num_samples = args.samples
    num_snapshots = args.num_snapshots
    narrow_band = int(args.frequency_center / 15)
    data_samples = np.zeros((num_samples, args.search_numbers, args.antenna_num, num_snapshots)) + 1j * np.zeros(
        (num_samples, args.search_numbers, args.antenna_num, num_snapshots))
    data_frequency = np.zeros((args.search_numbers, 1))
    for i in range(args.search_numbers):
        data_frequency[i] = args.frequency_center + (i - args.search_numbers // 2) * narrow_band
    label_theta = np.zeros((num_samples, 1))
    label_SNR = np.zeros((num_samples, 2))
    signal_ori = signal
    for sample in range(num_samples):
        # generate the SNR
        signal = signal_ori[int(args.frequency_sampling * args.amp_sample * 1 * np.random.uniform(0, 0.8)):]
        SNR_source = np.random.uniform(SNR_source_min, SNR_source_max)

        signal_noised = add_noise(signal, SNR_source)
        signal_noised = min_max_normalize(signal_noised)
        theta = np.random.randint(theta_min, theta_max)
        time_dalays = [delay_time_calculator(args.antenna_distance * i, theta, args) for i in range(args.antenna_num)]
        signals_delayed = [time_delay(signal_noised, time_dalay, args) for time_dalay in time_dalays]

        signals_delayed = np.array(signals_delayed).reshape(args.antenna_num, -1)

        signals_delayed_downsample = np.zeros((args.antenna_num, len(signals_delayed[0]) // args.amp_sample - 1))
        for i in range(args.antenna_num):
            signals_delayed_downsample[i] = signals_delayed[i][::args.amp_sample][:signals_delayed_downsample.shape[1]]
        signals_delayed = signals_delayed_downsample


        SNR_env = np.random.uniform(SNR_env_min, SNR_env_max)

        signals_delayed = np.array(signals_delayed).reshape(args.antenna_num, -1)
        signals_delayed_noised = add_noise(signals_delayed, SNR_env)

        # normalize the signal to [-1, 1]
        signals_delayed = (signals_delayed_noised - np.min(signals_delayed_noised)) / (np.max(signals_delayed_noised) - np.min(signals_delayed_noised))

        # slice the original signal into multiple segments
        for antenna in range(args.antenna_num):
            signal_sliced = [signal_slicer(signals_delayed[i], args) for i in range(args.antenna_num)]
        signal_sliced = np.array(signal_sliced).reshape(args.antenna_num, -1, args.snapshot_length)
        signal_sliced = signal_sliced.transpose((1, 0, 2))
        # assert signal_sliced.shape[0] >= num_snapshots, "Collected time is not enough to generate enough snapshots."
        signal_sliced = signal_sliced[:num_snapshots]

        # apply FFT to each segment
        snapshot_sample = np.zeros((args.search_numbers, args.antenna_num, num_snapshots)) + 1j * np.zeros(
            (args.search_numbers, args.antenna_num, num_snapshots))
        for antenna in range(args.antenna_num):
            for snapshot in range(num_snapshots):
                frequency_fft, signals_fft = fft_transform(signal_sliced[snapshot, antenna], args)
                # plt.plot(frequency_fft, np.abs(signals_fft))
                # plt.show()

                for narrow in range(args.search_numbers):
                    frequency_narrow_center = args.frequency_center + (narrow - args.search_numbers // 2) * narrow_band
                    # Find the narrowed frequency range and its amplitude
                    # TODO: Find the N-largest frequency components near the center frequency makes a fake result at zero degree.
                    # n_largest_frequency, n_largest_amp = find_largest_frequency_components(frequency_fft, signals_fft,
                    #                                                                        frequency_narrow_center,
                    #                                                                        narrow_band,
                    #                                                                        args)

                    # snapshot_sample[narrow, antenna, snapshot] = n_largest_amp[0]
                    frequency_components = signals_fft[np.where((frequency_fft > frequency_narrow_center - narrow_band / 2) & (frequency_fft < frequency_narrow_center + narrow_band / 2))]
                    snapshot_sample[narrow, antenna, snapshot] = frequency_components[int(frequency_components.size / 2)]
        data_samples[sample] = snapshot_sample
        label_theta[sample] = theta
        label_SNR[sample] = [SNR_env, SNR_source]
        # label_SNR[sample] = [SNR_source]

        paras = {
            'frequency_center': args.frequency_center,
            'frequency_fault': args.frequency_fault,
            'num_bands': args.search_numbers,
            'antenna_distance': args.antenna_distance,
        }
    return data_samples, label_theta, label_SNR, paras


if __name__ == '__main__':
    from TimeDomainGenerator import fault_generator
    from __init__ import args_data_generator

    args = args_data_generator()
    signal = fault_generator()
    data_samples_, data_frequency_, label_theta_, label_SNR_ = snapshot_exactor(signal, args)
    ...

    # time_dalays = [delay_time_calculator(args.antenna_distance * i, 30) for i in range(args.anrenna_num)]
    # signals_delayed = [time_delay(signal, time_dalay) for time_dalay in time_dalays]
    # signals_delayed = np.array(signals_delayed).reshape(args.anrenna_num, -1)
    #
    # # plot the time domain signal
    # for i in range(args.anrenna_num):
    #     signal_sliced = [signal_slicer(signals_delayed[i]) for i in range(args.anrenna_num)]
    # signal_sliced = np.array(signal_sliced).reshape(args.anrenna_num, -1, args.snapshot_length)
    #
    # # signal_sliced = signal_slicer(signal, 8192, 4096)
    # virtual_index = 12000
    # sample_index = 0
    # plt.figure()
    # plt.plot(signal_sliced[0, sample_index, :virtual_index], label='Antenna 1')
    # plt.plot(signal_sliced[1, sample_index, :virtual_index], label='Antenna 2')
    # plt.plot(signal_sliced[2, sample_index, :virtual_index], label='Antenna 3')
    # plt.plot(signal_sliced[3, sample_index, :virtual_index], label='Antenna 4')
    # plt.plot(signal_sliced[4, sample_index, :virtual_index], label='Antenna 5')
    # plt.plot(signal_sliced[5, sample_index, :virtual_index], label='Antenna 6')
    # plt.plot(signal_sliced[6, sample_index, :virtual_index], label='Antenna 7')
    # plt.plot(signal_sliced[7, sample_index, :virtual_index], label='Antenna 8')
    # plt.legend()
    # plt.show()
    #
    # signal_sample = signal_sliced[0, 0]
    # # plot the frequency domain signal
    # frequency_fft, signals_fft = fft_transform(signal_sample)
    # plt.figure()
    # plt.plot(frequency_fft, np.abs(signals_fft), label='Antenna 1')
    # plt.title('Frequency Domain Signal Sample')
    # plt.show()
    #
    # # find the N-largest frequency components
    #
    # n_largest_frequency, n_largest_amp = find_n_largest_frequency_components(frequency_fft, signals_fft)
    # print(f"The N-largest frequency components are {n_largest_frequency} \nwith the amplitude {n_largest_amp}")
