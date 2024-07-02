import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data

from FrequencyDomainExtractor import snapshot_exactor
from TimeDomainGenerator import fault_generator
from __init__ import args_data_generator

args = args_data_generator()


class DatasetGeneration_sample(torch.utils.data.Dataset):
    def __init__(self):
        signal = fault_generator()
        data_samples_, label_theta_, label_SNR_, paras = snapshot_exactor(signal)

        # paras = {
        #     'frequency_center': args.frequency_center,
        #     'frequency_fault': args.frequency_fault,
        #     'num_bands': args.search_numbers,
        # }

        narrow_band = paras["frequency_fault"]
        data_frequency = np.zeros((args.search_numbers, 1))

        for i in range(args.search_numbers):
            data_frequency[i] = args.frequency_center + (i - args.search_numbers // 2) * narrow_band

        self.data_samples = torch.from_numpy(data_samples_)
        self.data_frequency = torch.from_numpy(data_frequency)
        self.label_theta = torch.from_numpy(label_theta_)
        self.label_SNR = torch.from_numpy(label_SNR_)

    def __getitem__(self, index):
        return self.data_samples[index], self.data_frequency, self.label_theta[index], self.label_SNR[index]

    def __len__(self):
        return len(self.data_samples)

# class Dataset_Train(torch.utils.data.Dataset):
#     def __init__(self):
#         data_samples_ = torch.zeros((args.samples_repeat*args.
#         for _ in args.samples_repeat:



if '__main__' == __name__:

    dataset = DatasetGeneration_sample()
    print(f"Dataset length: {len(dataset)}")
    # Save dataset
    torch.save(dataset, '../../data/data2train.pt')
    # Load dataset

    dataset_ld = torch.load('../../data/data2train.pt')
    print(f"Dataset length: {len(dataset_ld)}")

    # read a sample
    data_samples, data_frequency, label_theta, label_SNR = dataset_ld[1]

    covariance_matrix_samples = torch.matmul(data_samples, data_samples.conj().transpose(1, 2)) / args.num_snapshots
    plt.imshow(np.abs(covariance_matrix_samples[0].numpy()))
    plt.show()

    import scipy.linalg
    # Apply wideband MUSIC algorithm
    def MUSIC(R, num_sources, num_sensors, frequency):
        """
        :param CovarianceMatrix
        :return:
        """
        CovarianceMatrix = np.array(R)
        frequency = np.array(frequency)[0]
        w, V = np.linalg.eig(CovarianceMatrix)
        w_index_order = np.argsort(w)
        V_noise = V[:, w_index_order[0:-num_sources]]
        noise_subspace = np.matmul(V_noise, np.matrix.getH(V_noise))
        doa_search = np.linspace(-60, 60, 121)
        p_music = np.zeros((len(doa_search), 1))

        for doa_index in range(len(doa_search)):
            # a = np.exp(
            #     1j * np.pi * np.arange(num_snesnors)[:, np.newaxis] * np.sin(np.deg2rad(doa_search[doa_index])))
            a = np.exp(1j * 2 * np.pi * frequency * 0.034 * np.arange(num_sensors)[:, np.newaxis] * np.sin(np.deg2rad(doa_search[doa_index])) / 340)
            p_music[doa_index] = np.abs(1 / np.matmul(np.matmul(np.matrix.getH(a), noise_subspace), a).reshape(-1)[0])
        p_music = p_music / np.max(p_music)
        p_music = 10 * np.log10(p_music)
        p_norm = p_music - np.min(p_music)
        # norm to [0, 1]
        p_norm = (p_norm - np.min(p_norm)) / (np.max(p_norm) - np.min(p_norm))
        return p_norm

    p_all = np.zeros((9, 121))
    for i in range(p_all.shape[0]):
        p_all[i] = MUSIC(covariance_matrix_samples[i], 1, 8, data_frequency[i]).reshape(-1)
    p_ave = np.mean(p_all, axis=0)
    thete = np.linspace(-60, 60, 121)

    plt.style.use(['science', 'ieee', 'grid'])
    plt.figure(dpi=800)
    for i in range(p_all.shape[0]):
        plt.plot(thete, p_all[i], alpha=0.3, linewidth=0.7, linestyle='-')
    plt.plot(thete, p_ave, alpha=1, linewidth=1.5, color='k', linestyle='-', label='Estimated spectrum')
    plt.axvline(x=label_theta[0], color='r', linestyle='--', label='Ground truth')
    plt.xlabel("DOA (°)")
    plt.ylabel("Amplitude")
    plt.title("MUSIC Algorithm")
    plt.legend(fontsize=8)
    plt.savefig('../../Test/Figures/MUSIC.pdf')
    plt.show()

