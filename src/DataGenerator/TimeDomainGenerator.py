import matplotlib.pyplot as plt
import numpy as np
import scipy


def fault_generator(frequency_nature: float, frequency_fault: float, damping_ratio: float, time: float, frequency_sampling: float):
    '''
    Generate a fault signal
    :param frequency_nature: aka. center frequency
    :param frequency_fault: aka. offset frequency
    :param damping_ratio:
    :param time:
    :param frequency_sampling:
    :return:
    '''
    omega_n = 2 * np.pi * frequency_nature
    omega_d = omega_n * np.sqrt(1 - damping_ratio ** 2)

    # Generate impulse response function
    length_filter = 512
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
    for i in range(0, len(impulsive_signal), int(frequency_sampling / frequency_fault)):
        impulsive_signal[i] = 1

    # fftfilt
    signal_fault = np.convolve(impulsive_signal / frequency_sampling, response_function, mode='full')
    return signal_fault[0:int(time * frequency_sampling)]


def frequency_spectrum(signal, frequency_sampling):
    '''
    Generate frequency spectrum of a signal
    :param signal:
    :param frequency_sampling:
    :return:
    '''
    length_signal = len(signal)
    frequency = np.fft.fftfreq(length_signal, 1 / frequency_sampling)
    spectrum = np.fft.fft(signal)
    return np.abs(frequency), np.abs(spectrum)


if '__main__' == __name__:
    signal_fault_ = fault_generator(4000, 500, 0.1, 10, 25600)
    plt.plot(signal_fault_[0:1000])
    plt.show()
    frequency, spectrum = frequency_spectrum(signal_fault_, 25600)
    plt.plot(frequency, spectrum)
    plt.show()


