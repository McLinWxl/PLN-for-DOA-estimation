#  Copyright (c) 2024. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

#  Copyright (c) 2024. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

import torch

from ISTA import ISTA
from __init__ import args_doa, args_unfolding_doa
from proposedMethods import proposedMethods

args = args_doa()
args_unfolding = args_unfolding_doa()
import matplotlib.pyplot as plt
import numpy as np
from functions import Spect2DoA_no_insert, calculate_error, DatasetGeneration_sample

# from functions import Spect2DoA_no_insert, calculate_error, DatasetGeneration_sample

if __name__ == '__main__':
    SNR='inf'
    print(f"SNR: {SNR}")
    dataset_ld = torch.load(f'../../data/data2test_g{SNR}.pt')
    print(f"Dataset length: {len(dataset_ld)}")
    test_loader = torch.utils.data.DataLoader(dataset_ld, batch_size=len(dataset_ld), shuffle=True)
    # test_loader = torch.utils.data.DataLoader(dataset_ld, batch_size=1, shuffle=True)

    method = 'Pro' # Pro, p-ISTA
    alpha = 0.25  # params for ISTA
    num_iters = args_unfolding.num_layers
    print(f"Method: {method}, Num Iters: {num_iters}, Alpha:{alpha}")
    num_sources = 1
    match method:
        case 'Pro':
            model = proposedMethods()
            def fit_theta(x):
                return 0.3075626 * np.exp(-5.26656769 * x) + 2.04751651 * np.exp(0.02712636 * x)
            def fit_gamma(x):
                return 1.49515499 * np.exp(-0.42760748 * x) + 1.49528979 * np.exp(
                    -0.42763038 * x) - 2.94534908 * np.exp(
                    -4.66312438 * x)
            theta = [fit_theta(a/num_iters) for a in range(num_iters)]
            gamma = [fit_gamma(a/num_iters) for a in range(num_iters)]
            model.theta_amp = torch.nn.Parameter(torch.tensor(theta), requires_grad=False)
            model.gamma_amp = torch.nn.Parameter(torch.tensor(gamma), requires_grad=False)
            plt.plot(theta, label='theta')
            plt.plot(gamma, label='gamma')
            plt.legend()
            plt.show()

        case 'ISTA':
            model = ISTA(alpha)

        case 'p-ISTA':
            model = proposedMethods()
            model.theta_amp = torch.nn.Parameter(alpha*torch.ones(num_iters), requires_grad=False)
            model.gamma_amp = torch.nn.Parameter(torch.ones(num_iters), requires_grad=False)

    plt.style.use(['science', 'ieee', 'grid'])
    for idx_epc, (data_samples, fre_center, fre_fault, spacing_sample, label_theta, label_SNR) in enumerate(
            test_loader):
        # if idx_epc == 0:
        #     print(label_SNR)
        batch_size = data_samples.shape[0]
        args.antenna_distance = spacing_sample
        args.frequency_center = fre_center
        args.frequency_fault = fre_fault

        covariance_matrix_samples = torch.matmul(data_samples, data_samples.conj().transpose(2, 3)) / args.num_snapshots

        # covariance_vector = covariance_matrix_samples.transpose(2, 3).reshape(covariance_matrix_samples.shape[0],
        #                                                                       covariance_matrix_samples.shape[1],
        #                                                                       args.antenna_num ** 2, 1)
        result_mulchannels, result, result_init_all = model(args, covariance_matrix_samples)

        result_ = result.reshape(-1, 121, 1)
        result_ = result_.cpu().detach().numpy()
        label_theta = label_theta.cpu().detach().numpy()
        label_theta = label_theta.reshape(-1, batch_size, num_sources)

        doa_est = Spect2DoA_no_insert(result_, num_sources=num_sources, start_bias=60)
        doa_est = doa_est.reshape(-1, batch_size, num_sources)
        error_, RMSE_doa, NMSE_doa, prob_doa = calculate_error(doa_est, label_theta, num_sources)
        print(f"RMSE: {RMSE_doa}, NMSE: {NMSE_doa}, prob: {prob_doa}")

        label = np.zeros_like(result_)
        for bat_id in range(result.shape[0]):
            for idx in range(result.shape[-2]):
                if int(idx - 60) == int(label_theta[:, bat_id, :]):
                    label[bat_id, idx, :] = 1

        for iii in range(result_.shape[0]):
            if result_[iii, 0] > 1:
                unexpectedValue = result_[iii]
                ...

        RMSE = np.sqrt(np.mean((result_ - label) ** 2))
        NMSE = np.mean((result_ - label) ** 2) / np.mean(label ** 2)
        NMSE_dB = 10 * np.log10(np.mean((result_ - label) ** 2) / np.mean(label ** 2))
        print(f"RMSE: {RMSE}, NMSE: {NMSE}, NMSE_dB: {NMSE_dB}")

        if batch_size == 1:

            label = np.zeros_like(result.cpu().detach().numpy())
            for bat_id in range(result.shape[0]):
                for idx in range(result.shape[-2]):
                    if int(idx) == int(label_theta[bat_id] + 60):
                        # label[bat_id, :, idx, :] = 1
                        label[bat_id, :, idx, :] = 1

            label = torch.from_numpy(label).to(torch.float32)
            if idx_epc % 1 == 0:
                for chanel in range(result_mulchannels.shape[1]):
                    plt.plot(result_mulchannels[0, chanel].cpu().detach().numpy(), ls='-', alpha=0.5,
                             linewidth=0.4)
                plt.plot(result[0].cpu().detach().numpy().reshape(-1), label=f'Result: {fre_center.item()}Hz', ls='-',
                         color='k', linewidth=1.5)
                # for x_idx in range(label.shape[-2]):
                plt.axvline(x=label_theta[0, 0] + 60, color='r', linestyle='--', linewidth=1, label='Ground Truth')
                # plt.plot(label[0].cpu().detach().numpy().reshape(-1), label='Label')
                plt.xlabel('Angles (Degrees)')
                plt.ylabel('Normalized Amplitude')
                # plt.plot(label[0].cpu().detach().numpy().reshape(-1), label='Label')
                # plt.title(f"Estimated Power Spectrum \n  \n {fre_center.item()} / {fre_fault.item()} \n {label_SNR[0][0].item()} / {label_SNR[0][1].item()} ")
                plt.title(f"Estimated Power Spectrum")
                plt.legend()
                plt.savefig(f"../../Test/Test_model/SNR_-10/{idx_epc}.pdf")
                plt.show()
