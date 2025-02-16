#  Copyright (c) 2024. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.utils.data
from functions import DatasetGeneration_sample, cal_steer_vector, cal_covariance, plot_sample
from __init__ import args_doa
args = args_doa()
import time


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
        a = cal_steer_vector(frequency=frequency, spacing=spacing, num_sensors=num_sensors, angle=doa_search[doa_index], speed_sound=args.speed_of_sound)
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

    A = cal_steer_vector(frequency=frequency, spacing=args.antenna_distance, num_sensors=args.antenna_num, angle=doa_search, speed_sound=args.speed_of_sound)
    mu = A.T.conjugate() @ np.linalg.pinv(A @ A.T.conjugate()) @ raw_data
    sigma2 = 0.1 * np.linalg.norm(raw_data, 'fro') ** 2 / (args.antenna_num * args.num_snapshots)
    gamma = np.diag((mu @ mu.T.conjugate()).real) / args.num_snapshots
    ItrIdx = 1
    stop_iter = False
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


if __name__ == '__main__':
    dataset_ld = torch.load('../../data/data2test_g0.pt')

    print(f"Dataset length: {len(dataset_ld)}")

    idx_val = 3
    # read a sample
    data_samples, fre_center, fre_fault, spacing_sample, label_theta, label_SNR = dataset_ld[idx_val]

    args.antenna_distance = spacing_sample
    args.frequency_center = fre_center
    args.frequency_fault = fre_fault

    print(f"Antenna distance: {spacing_sample}")
    print(f"Center frequency: {fre_center}")
    print(f"Fault frequency: {fre_fault}")
    print(f"Ground truth: {label_theta}")
    print(f"SNR: {label_SNR}")

    covariance_matrix_samples = cal_covariance(data_samples, args)
    plt.imshow(np.abs(covariance_matrix_samples[0].numpy()))
    plt.show()

    method = 'MUSIC'

    p_all = np.zeros((args.search_numbers, 121))
    time_start = time.time()
    for i in range(p_all.shape[0]):
        frequency_sample = fre_center + (i - args.search_numbers // 2) * fre_fault
        if method.upper() == 'MUSIC':
            p_all[i] = MUSIC(covariance_matrix_samples[i], 1, 8, frequency_sample, spacing_sample).reshape(-1)
        elif method.upper() == 'SBL':
            p_all[i] = SBL(data_samples[i], frequency_sample, args).reshape(-1)
    time_end = time.time()
    print(f"Time cost: {time_end - time_start}")
    p_ave = np.mean(p_all, axis=0)
    thete = np.linspace(-60, 60, 121)

    plot_sample(method.upper(), p_all, label_theta, thete, label_SNR, args, save_path="../../results/plots/result.pdf")

