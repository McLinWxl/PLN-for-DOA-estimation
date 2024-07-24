#  Copyright (c) 2024. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.


import numpy as np
import torch
import matplotlib.pyplot as plt
import time


class DatasetGeneration_sample(torch.utils.data.Dataset):
    # def __init__(self, args_):
        # signal = fault_generator(args_)
        # data_samples_, label_theta_, label_SNR_, paras = snapshot_exactor(signal, args_)
        #
        # # paras = {
        # #     'frequency_center': args.frequency_center,
        # #     'frequency_fault': args.frequency_fault,
        # #     'num_bands': args.search_numbers,
        # # }
        #
        # # a_frequency[i] = args.frequency_center + (i - args.search_numbers // 2) * narrow_band
        #
        # self.data_samples = torch.from_numpy(data_samples_)
        # # self.data_frequency = torch.from_numpy(data_frequency)
        # self.label_theta = torch.from_numpy(label_theta_)
        # self.label_SNR = torch.from_numpy(label_SNR_)
        # self.paras = paras

    def __getitem__(self, index):
        return (self.data_samples[index],
                self.frequency_center[index],
                self.frequency_fault[index],
                self.antenna_distance[index],
                self.label_theta[index],
                self.label_SNR[index])

    def __len__(self):
        return len(self.data_samples)


def cal_steer_vector(frequency, spacing, num_sensors, angle, speed_sound):
    """
    Calculate the steering vector for a given frequency, spacing, number of sensors, angle, and speed of sound.
    :param frequency:
    :param spacing:
    :param num_sensors:
    :param angle:
    :param speed_sound:
    :return:
    """
    a = np.exp(1j * 2 * np.pi * frequency * spacing * np.arange(num_sensors)[:, np.newaxis] * np.sin(np.deg2rad(angle)) / speed_sound)
    return a

def cal_covariance(data_samples, args):
    """
    Calculate the covariance matrix for a given data sample.
    :param data_sample:
    :return:
    """
    covariance_matrix_samples = torch.matmul(data_samples, data_samples.conj().transpose(1, 2)) / args.num_snapshots
    return covariance_matrix_samples

def cal_dictionary(frequency, args, idx):
    dictionary = np.zeros((args.antenna_num**2, args.num_meshes), dtype=np.complex64)
    num_grids = args.num_meshes
    w_m = np.zeros((args.antenna_num, 121)) + 1j * np.zeros((args.antenna_num, 121))
    for i in range(args.antenna_num):
        theta_grids = np.arange(args.theta_min, args.theta_max+1, int((args.theta_max+1-args.theta_min)/args.num_meshes)).reshape(-1)

        manifold = np.exp(1j * 2 * np.pi * frequency * args.antenna_distance[idx] * np.arange(args.antenna_num)[:, np.newaxis] * np.sin(np.deg2rad(theta_grids)) / args.speed_of_sound)

        for j in range(args.num_meshes):
            steer_vec = manifold[:, j].reshape(-1, 1)
            steer_map = torch.matmul(steer_vec, steer_vec.conj().T)
            w_m[:, j] = steer_map[:, i]
        dictionary[i*args.antenna_num:(i+1)*args.antenna_num, :] = w_m
    return dictionary

def plot_sample(method: str, results, label_theta, grid_theta, args, save_path=None):
    result_ave = np.mean(results, axis=0)
    plt.style.use(['science', 'ieee', 'grid'])
    plt.figure(dpi=800)
    for i in range(results.shape[0]):
        plt.plot(grid_theta, results[i], alpha=0.3, linewidth=0.7, linestyle='-')
    plt.plot(grid_theta, result_ave, alpha=1, linewidth=1.5, color='k', linestyle='-', label='Estimated spectrum')
    plt.axvline(x=label_theta[0], color='r', linestyle='--', label='Ground truth')
    plt.xlabel("DOA (°)")
    plt.ylabel("Amplitude")
    plt.title(f"SBL Algorithm for Impulsive Signal DOA Estimation at:  \n "
              f"1. Center frequency: {args.frequency_center} Hz,  \n "
              f"2. Fault frequency: {args.frequency_fault} Hz,  \n "
              f"3. Antenna distance: {args.antenna_distance} meters, \n ")
              # f"4. SNR of environment, SNR of source: {np.array(label_SNR)} dB")
    plt.legend(fontsize=8)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def train_proposed(model, epoch, dataloader, optimizer, criterion, args):
    model.train()
    plt.style.use(['science', 'ieee', 'grid'])
    time_start = time.time()
    for epc in range(epoch):
        if epc == 10:
            print(f'Time costs for 10 epc: {time.time() - time_start} seconds')

        if epc <= 20 or epc % 10 == 0:
            torch.save({'model': model.state_dict()}, f"../../model/model_{epc}.pth")
        losses = []
        for idx_epc, (data_samples, fre_center, fre_fault, spacing_sample, label_theta, label_SNR) in enumerate(dataloader):

            args.antenna_distance = spacing_sample
            args.frequency_center = fre_center
            args.frequency_fault = fre_fault

            covariance_matrix_samples = torch.matmul(data_samples, data_samples.conj().transpose(2, 3)) / args.num_snapshots

            covariance_vector = covariance_matrix_samples.transpose(2, 3).reshape(covariance_matrix_samples.shape[0], covariance_matrix_samples.shape[1], args.antenna_num ** 2, 1)

            result_mulchannels, result, result_init_all = model(args, covariance_vector)

            # print(f"Antenna distance: {spacing_sample}")
            # print(f"Center frequency: {fre_center}")
            # print(f"Fault frequency: {fre_fault}")
            # print(f"Ground truth: {label_theta}")
            # print(f"SNR: {label_SNR}")

            # plot_sample('ISTA', result.cpu().detach().numpy().reshape(9, 121, 1), label_theta.reshape(-1), np.linspace(-60, 60, 121), args, '../../Test/Figures/ISTA-500.pdf')

            label = torch.zeros_like(result)
            for bat_id in range(result.shape[0]):
                for idx in range(result.shape[-2]):
                    if int(idx - 60) == int(label_theta[bat_id]):
                        label[bat_id, :, idx, :] = 1

            label_all_layers = torch.zeros_like(result_init_all)
            for search_idx in range(result_init_all.shape[1]):
                for layer_idx in range(result_init_all.shape[2]):
                    label_all_layers[:, search_idx, layer_idx, :, :] = label.reshape(-1, label.shape[-2], 1)

            # label = torch.from_numpy(label).to(torch.float32)

            # Calculate the sparse regularization term
            lambda_sparse = 0.1
            loss_recovery = criterion(result, label)
            loss_sparse = torch.nn.L1Loss()(result_init_all, label_all_layers)
            loss = (1-lambda_sparse) * loss_recovery + lambda_sparse * loss_sparse

            if idx_epc % 10 == 0:
                for chanel in range(result_mulchannels.shape[1]):
                    plt.plot(result_mulchannels[0, chanel].cpu().detach().numpy(), ls='-', color='k', alpha=0.6, linewidth=0.7)
                plt.plot(result[0].cpu().detach().numpy().reshape(-1), label='Result', color='b', linewidth=1.5)
                plt.plot(label[0].cpu().detach().numpy().reshape(-1), label='Label')
                plt.legend()
                plt.show()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(f"Epoch: {epc}, Loss: {loss.item()}, Threshold: {model.theta.item()}, Step size: {model.gamma.item()}")
            print(f"Epoch: {epc}, Loss: {loss.item()}, lambda_sparse: {lambda_sparse}, Threshold: {model.theta.item()}, Step size: {model.gamma.item()}, \n, Loss recovery: {loss_recovery}, Loss sparse: {loss_sparse}, Costs Time: {time.time() - time_start} Seconds")
            losses.append(loss.item())
    return losses


def test_proposed(model, checkpoint, dataloader, args):
    model.load_state_dict(torch.load(checkpoint)['model'])
    model.eval()
    plt.style.use(['science', 'ieee', 'grid'])
    time_start = time.time()
    for idx_epc, (data_samples, fre_center, fre_fault, spacing_sample, label_theta, label_SNR) in enumerate(dataloader):

        args.antenna_distance = spacing_sample
        args.frequency_center = fre_center
        args.frequency_fault = fre_fault

        covariance_matrix_samples = torch.matmul(data_samples, data_samples.conj().transpose(2, 3)) / args.num_snapshots

        covariance_vector = covariance_matrix_samples.transpose(2, 3).reshape(covariance_matrix_samples.shape[0], covariance_matrix_samples.shape[1], args.antenna_num ** 2, 1)

        result_mulchannels, result, result_init_all = model(args, covariance_vector)

        # print(f"Antenna distance: {spacing_sample}")
        # print(f"Center frequency: {fre_center}")
        # print(f"Fault frequency: {fre_fault}")
        # print(f"Ground truth: {label_theta}")
        # print(f"SNR: {label_SNR}")

        # plot_sample('ISTA', result.cpu().detach().numpy().reshape(9, 121, 1), label_theta.reshape(-1), np.linspace(-60, 60, 121), args, '../../Test/Figures/ISTA-500.pdf')

        label = np.zeros_like(result.cpu().detach().numpy())
        for bat_id in range(result.shape[0]):
            for idx in range(result.shape[-2]):
                if int(idx - 60) == int(label_theta[bat_id]):
                    label[bat_id, :, idx, :] = 1

        label = torch.from_numpy(label).to(torch.float32)
        if idx_epc % 2 == 0:
            for chanel in range(result_mulchannels.shape[1]):
                plt.plot(result_mulchannels[0, chanel].cpu().detach().numpy(), ls='-', color='k', alpha=0.6, linewidth=0.7)
            plt.plot(result[0].cpu().detach().numpy().reshape(-1), label='Result', color='b', linewidth=1.5)
            plt.plot(label[0].cpu().detach().numpy().reshape(-1), label='Label')
            plt.legend()
            plt.show()













