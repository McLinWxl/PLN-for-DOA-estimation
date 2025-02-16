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
import time
# from functions import Spect2DoA_no_insert, calculate_error, DatasetGeneration_sample

def cal_fre(signal, fs):
    signal = signal.reshape(-1)
    fft_result = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(len(fft_result), 1 / fs)
    fft_magnitude = np.abs(fft_result) / len(signal)
    single_sided_magnitude = fft_magnitude[:len(fft_magnitude) // 2]
    single_sided_freq = fft_freq[:len(fft_freq) // 2]
    return single_sided_freq, single_sided_magnitude

if __name__ == '__main__':
    SNR=0
    print(f"SNR: {SNR}")
    dataset_ld = torch.load(f'../../data/data2train_type1_source1.pt')
    print(f"Dataset length: {len(dataset_ld)}")
    # test_loader = torch.utils.data.DataLoader(dataset_ld, batch_size=len(dataset_ld), shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset_ld, batch_size=1, shuffle=False)

    method = 'Pro' # Pro, p-ISTA
    alpha = 1  # params for ISTA
    num_iters = args_unfolding.num_layers
    print(f"Method: {method}, Num Iters: {num_iters}, Alpha:{alpha}")
    # num_sources = 1
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

    # plt.style.use(['science', 'ieee', 'grid'])
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.figure(dpi=500)
    time_costs = 0
    signal_saved = np.zeros((len(dataset_ld), args.frequency_sampling))
    result_saved = np.zeros((len(dataset_ld), args.search_numbers, 121))
    label_saved = np.zeros((len(dataset_ld), 121))
    args_saved = np.zeros((len(dataset_ld), args.num_sources, 6))
    for idx_epc, (data_samples, fre_center, fre_fault, spacing_sample, label_theta, label_SNR, signal_sample) in enumerate(
            test_loader):
        # x_fre, y_fre = cal_fre(signal_sample, args.frequency_sampling)
        # plt.plot(x_fre, y_fre)
        # plt.xlim(0, 10000)
        # plt.show()
        start_time = time.time()
        # if idx_epc == 0:
        #     print(label_SNR)
        batch_size = data_samples.shape[0]
        args.antenna_distance = [spacing_sample[0][0].item()]
        args.frequency_center = [fre_center[0][0].item()]
        args.frequency_fault = [fre_fault[0][0].item()]

        covariance_matrix_samples = torch.matmul(data_samples, data_samples.conj().transpose(2, 3)) / args.num_snapshots

        # covariance_vector = covariance_matrix_samples.transpose(2, 3).reshape(covariance_matrix_samples.shape[0],
        #                                                                       covariance_matrix_samples.shape[1],
        #                                                                       args.antenna_num ** 2, 1)
        result_mulchannels, result, result_init_all = model(args, covariance_matrix_samples)

        time_costs += time.time() - start_time

        result_ = result.reshape(-1, 121, 1)
        result_ = result_.cpu().detach().numpy()
        label_theta = label_theta.cpu().detach().numpy()
        label_theta = label_theta.reshape(-1, batch_size, args.num_sources)

        if batch_size == 1:
            for i_source in range(args.num_sources):
                args_saved[idx_epc, i_source] = [fre_center[0][i_source].item(), fre_fault[0][i_source].item(), spacing_sample[0][i_source].item(), label_theta[0][0][i_source].item(), label_SNR[0][0].item(), label_SNR[0][1].item()]
                # args_saved[idx_epc, i_source] = [fre_center[0][0].item(), fre_fault[0][0].item(), spacing_sample[0][0].item(), label_theta, label_SNR]
            signal_saved[idx_epc] = signal_sample.reshape(-1)
            result_mul = result_mulchannels.reshape(-1, 121)
            result_mul = result_mul.cpu().detach().numpy()

            # for chanel in range(result_mul.shape[0]):
            #     plt.plot(result_mul[chanel], ls='-', alpha=0.5, linewidth=0.7)
            # for i_source in range(args.num_sources):
            #     plt.axvline(x=label_theta[0][0][i_source].item() - args.theta_min, color='r', linestyle='--', linewidth=1, label='Ground Truth')
            #
            # plt.show()
            # for x_idx in range(label.shape[-2]):

            result_saved[idx_epc] = result_mul

            for i_source in range(args.num_sources):
                label_saved[idx_epc, int(label_theta[0][0][i_source].item()) - args.theta_min] = 1


        doa_est = Spect2DoA_no_insert(result_, num_sources=args.num_sources, start_bias=60)
        doa_est = doa_est.reshape(-1, batch_size, args.num_sources)
        error_, RMSE_doa, NMSE_doa, prob_doa = calculate_error(doa_est, label_theta, args.num_sources)
        print(f"RMSE: {RMSE_doa}, NMSE: {NMSE_doa}, prob: {prob_doa}")

        label = np.zeros_like(result_)
        for bat_id in range(result.shape[0]):
            for i_source in range(args.num_sources):
                label_saved[idx_epc, int(label_theta[0][0][i_source].item()) - args.theta_min] = 1

            # for idx in range(result.shape[-2]):
            #     if int(idx - 60) == int(label_theta[:, bat_id, :]):
            #         label[bat_id, idx, :] = 1

        for iii in range(result_.shape[0]):
            if result_[iii, 0] > 1:
                unexpectedValue = result_[iii]
                ...

        RMSE = np.sqrt(np.mean((result_ - label) ** 2))
        NMSE = np.mean((result_ - label) ** 2) / np.mean(label ** 2)
        NMSE_dB = 10 * np.log10(np.mean((result_ - label) ** 2) / np.mean(label ** 2))
        # print(f"RMSE: {RMSE}, NMSE: {NMSE}, NMSE_dB: {NMSE_dB}")

        if batch_size == 1 and idx_epc < 4 and False:

            label = np.zeros_like(result.cpu().detach().numpy())
            for bat_id in range(result.shape[0]):
                for idx in range(result.shape[-2]):
                    if int(idx) == int(label_theta[bat_id] + 60):
                        # label[bat_id, :, idx, :] = 1
                        label[bat_id, :, idx, :] = 1

            label = torch.from_numpy(label).to(torch.float32)
            if idx_epc % 1 == 0:
                x_mesh = np.linspace(-60, 60, 121)
                for chanel in range(result_mulchannels.shape[1]):
                    plt.plot(x_mesh, result_mulchannels[0, chanel].cpu().detach().numpy(), ls='-', alpha=0.5,
                             linewidth=0.7)
                plt.plot(x_mesh, result[0].cpu().detach().numpy().reshape(-1), label=f'Result: {fre_center.item()}Hz', ls='-',
                         color='k', linewidth=2)
                # for x_idx in range(label.shape[-2]):
                plt.xlim(-66, 66)
                plt.axvline(x=label_theta[0, 0], color='r', linestyle='--', linewidth=1, label='Ground Truth')
                # plt.plot(label[0].cpu().detach().numpy().reshape(-1), label='Label')
                plt.xlabel('Angles (Degrees)')
                plt.ylabel('Normalized Amplitude')
                # plt.plot(label[0].cpu().detach().numpy().reshape(-1), label='Label')
                # plt.title(f"Estimated Power Spectrum \n  \n {fre_center.item()} / {fre_fault.item()} \n {label_SNR[0][0].item()} / {label_SNR[0][1].item()} ")
                plt.title(f"Estimated Power Spectrum")
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.8)

                plt.savefig(f"../../Test/Test_model/SNR_-10/{idx_epc}.pdf")
                plt.show()
        # print(f"Time costs: {time_costs}, Average time costs: {time_costs / len(dataset_ld)}")
    # save signal_saved, result_saved, label_saved and args_saved
    dataFrame = {
        'signal': signal_saved,
        'result': result_saved,
        'label': label_saved,
        'args': args_saved
    }
    torch.save(dataFrame, f'../../data/estimated_type1_source1.pt')