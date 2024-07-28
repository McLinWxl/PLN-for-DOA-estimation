import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data

from FrequencyDomainExtractor import snapshot_exactor
from TimeDomainGenerator import fault_generator
from __init__ import args_data_generator
from rich.progress import track



class DatasetGeneration_sample(torch.utils.data.Dataset):
    def __init__(self, args_):
        signal = fault_generator(args_)
        data_samples_, label_theta_, label_SNR_, paras = snapshot_exactor(signal, args_)

        # paras = {
        #     'frequency_center': args.frequency_center,
        #     'frequency_fault': args.frequency_fault,
        #     'num_bands': args.search_numbers,
        # }

        # a_frequency[i] = args.frequency_center + (i - args.search_numbers // 2) * narrow_band

        self.data_samples = torch.from_numpy(data_samples_)
        # self.data_frequency = torch.from_numpy(data_frequency)
        self.label_theta = torch.from_numpy(label_theta_)
        self.label_SNR = torch.from_numpy(label_SNR_)
        # self.paras = paras

        self.frequency_center = np.zeros_like(label_theta_) + paras['frequency_center']
        self.frequency_fault = np.zeros_like(label_theta_) + paras['frequency_fault']
        self.antenna_distance = np.zeros_like(label_theta_) + paras['antenna_distance']

    def __getitem__(self, index):
        return (self.data_samples[index],
                self.frequency_center[index],
                self.frequency_fault[index],
                self.antenna_distance[index],
                self.label_theta[index],
                self.label_SNR[index])

    def __len__(self):
        return len(self.data_samples)

def dataset_train(args):
    # center_sets = [8500, 7000, 5666, 5000, 4250, 4000, 3400, 3000, 2833]
    # fault_sets = [75, 150, 300, 600]
    # spacing_sets = [0.02, 0.03, 0.04, 0.05, 0.06]
    center_sets = [14000, 4000, 1750]
    fault_sets = [50, 100, 200, 450, 800]
    spacing_sets = [0.01, 0.035, 0.08]
    # center_sets = [10000, 9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000]
    # fault_sets = [75, 150, 225, 300, 450, 600, 800]
    # spacing_sets = [0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    # center_sets = [5000, 4000, 3000]
    # fault_sets = [50, 150, 300]
    # spacing_sets = [0.03]
    # center_sets = [5666]
    # fault_sets = [150]
    # spacing_sets = [0.03]
    dataset_all = DatasetGeneration_sample(args)
    for center in track(center_sets, description="Generating dataset"):
        for fault in fault_sets:
            for spacing in spacing_sets:
                if np.abs(center * spacing - 170) >= 25:
                    continue
                print(center, spacing, fault, center * spacing - 170)

                args.frequency_center = center
                args.frequency_fault = fault
                args.antenna_distance = spacing
                for _ in range(args.samples_repeat):
                    dataset = DatasetGeneration_sample(args)
                    dataset_all = dataset_all + dataset
    return dataset_all


if '__main__' == __name__:
    # import scipy.linalg as la
    args = args_data_generator()

    #
    dataset = dataset_train(args)
    print(f"Dataset length: {len(dataset)}")
    # Save dataset
    torch.save(dataset, '../../data/data2test.pt')
    # Load dataset

    dataset_ld = torch.load('../../data/data2test.pt')
    print(f"Dataset length: {len(dataset_ld)}")

    # read a sample
    data_samples, fre_center, fre_fault, spacing_sample, label_theta, label_SNR = dataset_ld[211]

    args.antenna_distance = spacing_sample
    args.frequency_center = fre_center
    args.frequency_fault = fre_fault

    print(f"Antenna distance: {spacing_sample}")
    print(f"Center frequency: {fre_center}")
    print(f"Fault frequency: {fre_fault}")
    print(f"Ground truth: {label_theta}")
    print(f"SNR: {label_SNR}")

    covariance_matrix_samples = torch.matmul(data_samples, data_samples.conj().transpose(1, 2)) / args.num_snapshots
    plt.imshow(np.abs(covariance_matrix_samples[0].numpy()))
    plt.show()

    # Apply wideband MUSIC algorithm
    def MUSIC(R, num_sources, num_sensors, frequency, spacing):
        """
        :param CovarianceMatrix
        :return:
        """
        CovarianceMatrix = np.array(R)
        frequency = np.array(frequency)
        w, V = np.linalg.eig(CovarianceMatrix)
        w_index_order = np.argsort(w)
        V_noise = V[:, w_index_order[0:-num_sources]]
        noise_subspace = np.matmul(V_noise, np.matrix.getH(V_noise))
        doa_search = np.linspace(-60, 60, 121)
        p_music = np.zeros((len(doa_search), 1))

        for doa_index in range(len(doa_search)):
            # a = np.exp(
            #     1j * np.pi * np.arange(num_snesnors)[:, np.newaxis] * np.sin(np.deg2rad(doa_search[doa_index])))
            a = np.exp(1j * 2 * np.pi * frequency * spacing * np.arange(num_sensors)[:, np.newaxis] * np.sin(np.deg2rad(doa_search[doa_index])) / 340)
            p_music[doa_index] = np.abs(1 / np.matmul(np.matmul(np.matrix.getH(a), noise_subspace), a).reshape(-1)[0])
        p_music = p_music / np.max(p_music)
        p_music = 10 * np.log10(p_music)
        p_norm = p_music - np.min(p_music)
        # norm to [0, 1]
        p_norm = (p_norm - np.min(p_norm)) / (np.max(p_norm) - np.min(p_norm))
        return p_norm

    def SBL(raw_data, frequency, args, max_iteration=500, error_threshold=1e-3):
        """
        :param raw_data:
        :param max_iteration:
        :param error_threshold:
        :return:
        """
        raw_data = np.array(raw_data)
        doa_search = np.linspace(-60, 60, 121)
        A = np.exp(1j * 2 * np.pi * frequency * args.antenna_distance * np.arange(args.antenna_num)[:, np.newaxis] * np.sin(np.deg2rad(doa_search)) / args.speed_of_sound)
        mu = A.T.conjugate() @ np.linalg.pinv(A @ A.T.conjugate()) @ raw_data
        sigma2 = 0.1 * np.linalg.norm(raw_data, 'fro') ** 2 / (args.antenna_num * args.num_snapshots)
        gamma = np.diag((mu @ mu.T.conjugate()).real) / args.num_snapshots
        ItrIdx = 1
        stop_iter = False
        gamma0 = gamma
        while not stop_iter and ItrIdx < max_iteration:
            gamma0 = gamma
            Q = sigma2 * np.eye(args.antenna_num) + np.dot(np.dot(A, np.diag(gamma)), A.T.conjugate())
            Qinv = np.linalg.pinv(Q)
            Sigma = np.diag(gamma) - np.dot(np.dot(np.dot(np.diag(gamma), A.T.conjugate()), Qinv),
                                            np.dot(A, np.diag(gamma)))
            mu = np.dot(np.dot(np.diag(gamma), A.T.conjugate()), np.dot(Qinv, raw_data))
            sigma2 = ((np.linalg.norm(raw_data - np.dot(A, mu), 'fro') ** 2 + args.num_snapshots * np.trace(
                np.dot(np.dot(A, Sigma), A.T.conjugate()))) /
                      (args.antenna_num * args.num_snapshots)).real
            mu_norm = np.diag(mu @ mu.T.conjugate()) / args.num_snapshots
            gamma = np.abs(mu_norm + np.diag(Sigma))

            if np.linalg.norm(gamma - gamma0) / np.linalg.norm(gamma) < error_threshold:
                stop_iter = True
            ItrIdx += 1
        return gamma

    p_all = np.zeros((args.search_numbers, 121))
    for i in range(p_all.shape[0]):
        frequency_sample = fre_center + (i - args.search_numbers // 2) * fre_fault
        p_all[i] = MUSIC(covariance_matrix_samples[i], 1, 8, frequency_sample, spacing_sample).reshape(-1)
        # p_all[i] = SBL(data_samples[i], frequency_sample, args).reshape(-1)
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
    plt.title(f"SBL Algorithm for Impulsive Signal DOA Estimation at:  \n "
              f"1. Center frequency: {fre_center} Hz,  \n "
              f"2. Fault frequency: {fre_fault} Hz,  \n "
              f"3. Antenna distance: {spacing_sample} meters, \n "
              f"4. SNR of environment, SNR of source: {np.array(label_SNR)} dB")
    plt.legend(fontsize=8)
    plt.savefig('../../Test/Figures/SBL.pdf')
    plt.show()

