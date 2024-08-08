#  Copyright (c) 2024. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.
from crypt import methods

#  Copyright (c) 2024. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

import torch
from scipy.stats import gamma

from proposedMethods import proposedMethods
from ISTA import ISTA
from functions import test_proposed, DatasetGeneration_sample
from __init__ import args_doa, args_unfolding_doa
args = args_doa()
args_unfolding = args_unfolding_doa()
import matplotlib.pyplot as plt
import numpy as np
from functions import Spect2DoA_no_insert, calculate_error

if __name__ == '__main__':
    dataset_ld = torch.load('../../data/data2test_-5.pt')
    print(f"Dataset length: {len(dataset_ld)}")
    test_loader = torch.utils.data.DataLoader(dataset_ld, batch_size=1, shuffle=False)

    method = 'ISTA'
    num_iters = args_unfolding.num_layers
    num_sources = 1
    match method:
        case 'Pro':
            model = proposedMethods()


            fixed_iters = 40
            x_axis = np.arange(fixed_iters)
            wights_ = [6.69332518e-14, -1.46119797e-11, 1.35070284e-09, -6.88957794e-08,
                       2.10771018e-06, -3.90486205e-05, 4.15174983e-04, -2.18875963e-03,
                       4.76018000e-03, -4.14020957e-03, -6.24989189e-04]
            wights_ = wights_[::-1]
            fixed_theta = np.zeros_like(x_axis)
            for i in range(len(wights_)):
                fixed_theta = fixed_theta + wights_[i] * x_axis ** i

            # fixed_theta = fixed_theta * 0

            # fixed_theta = fixed_theta + 1
            fixed_theta = fixed_theta
            fixed_gamma = -fixed_theta

            theta = np.zeros(num_iters)
            gamma = np.zeros(num_iters)
            if num_iters >= fixed_iters:
                theta[-fixed_iters:] = fixed_theta
                gamma[-fixed_iters:] = fixed_gamma

            else:
                theta = fixed_theta[-num_iters:]
                gamma = fixed_gamma[-num_iters:]

            model.theta_amp = torch.nn.Parameter(torch.tensor(theta), requires_grad=False)
            model.gamma_amp = torch.nn.Parameter(torch.tensor(gamma), requires_grad=False)

        case 'ISTA':
            model = ISTA()

    plt.style.use(['science', 'ieee', 'grid'])
    for idx_epc, (data_samples, fre_center, fre_fault, spacing_sample, label_theta, label_SNR) in enumerate(test_loader):
        batch_size = data_samples.shape[0]
        args.antenna_distance = spacing_sample
        args.frequency_center = fre_center
        args.frequency_fault = fre_fault

        covariance_matrix_samples = torch.matmul(data_samples, data_samples.conj().transpose(2, 3)) / args.num_snapshots

        covariance_vector = covariance_matrix_samples.transpose(2, 3).reshape(covariance_matrix_samples.shape[0],
                                                                              covariance_matrix_samples.shape[1],
                                                                              args.antenna_num ** 2, 1)
        result_mulchannels, result, result_init_all = model(args, covariance_vector)

        result_ = result.reshape(-1, 121, 1)
        result_ = result_.cpu().detach().numpy()
        doa_est = Spect2DoA_no_insert(result_, num_sources=num_sources, start_bias=60)
        doa_est = doa_est.reshape(-1, batch_size, num_sources)
        label_theta = label_theta.cpu().detach().numpy()
        label_theta = label_theta.reshape(-1, batch_size, num_sources)
        error_, RMSE, NMSE, prob = calculate_error(doa_est, label_theta, num_sources)
        print(f"RMSE: {RMSE}, NMSE: {NMSE}, Prob: {prob}")


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
                plt.plot(result[0].cpu().detach().numpy().reshape(-1), label=f'Result: {fre_center.item()}Hz', ls='-', color='k', linewidth=1.5)
                # for x_idx in range(label.shape[-2]):
                plt.axvline(x=label_theta[0,0]+60, color='r', linestyle='--', linewidth=1, label='Ground Truth')
                # plt.plot(label[0].cpu().detach().numpy().reshape(-1), label='Label')
                plt.xlabel('Angles (Degrees)')
                plt.ylabel('Normalized Amplitude')
                # plt.plot(label[0].cpu().detach().numpy().reshape(-1), label='Label')
                # plt.title(f"Estimated Power Spectrum \n  \n {fre_center.item()} / {fre_fault.item()} \n {label_SNR[0][0].item()} / {label_SNR[0][1].item()} ")
                plt.title(f"Estimated Power Spectrum")
                plt.legend()
                plt.savefig(f"../../Test/Test_model/SNR_-10/{idx_epc}.pdf")
                plt.show()

