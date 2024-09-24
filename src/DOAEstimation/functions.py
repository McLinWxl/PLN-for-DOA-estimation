#  Copyright (c) 2024. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.


import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import scipy.signal
import heapq
import os


def calculate_error(peak, label, num_sources):
    """
    :param peak: (num_lists, samples, 2)
    :return: error, RMSE, NMSE, prob
    """
    DOA_train = label
    num_list, num_id, _ = peak.shape
    RMSE = np.zeros(num_list)
    NMSE = np.zeros(num_list)
    prob = np.zeros(num_list)
    predict_st = peak
    DOA_train_st = DOA_train
    error_ = np.zeros((num_list, num_id, num_sources))
    for snr in range(num_list):
        for itt in range(num_id):
            predict_st[snr, itt] = np.sort(peak[snr, itt])
            DOA_train_st[snr, itt] = np.sort(DOA_train[snr, itt])
    for snr in range(num_list):
        error = np.abs(np.sort(predict_st[snr]) - DOA_train_st[snr])
        error_[snr] = (predict_st[snr]) - DOA_train_st[snr]
        for idx in range(num_id):
            for i in range(num_sources):
                prob[snr] += np.sum(error[idx, i] <= 4.4)
                if error[idx, i] > 4.4:
                    error[idx, i] = 10
                # if error[idx, i] == 0:
                #     error[idx, i] =
        RMSE[snr] = np.sqrt(np.mean(error ** 2))
        NMSE[snr] = (np.mean(error ** 2) / np.mean(DOA_train_st[snr] ** 2))
        prob[snr] = prob[snr] / num_id

    return error_, RMSE, NMSE, prob / num_sources

def Spect2DoA(Spectrum, num_sources=2, start_bias=60):
    """
    :param Spectrum: (num_samples, num_meshes, 1)
    :param num_sources:
    :param height_ignore:
    :param start_bias:
    :return: (num_samples, num_sources)
    """
    num_samples, num_meshes, _ = Spectrum.shape
    angles = np.zeros((num_samples, num_sources))
    for num in range(num_samples):
        li_0 = Spectrum[num, :].reshape(-1)
        # li_0[li_0 < 0] = 0
        li = li_0
        angle = np.zeros(num_sources) - 5
        peaks_idx = np.zeros(num_sources)
        grids_mesh = np.arange(num_meshes) - start_bias
        peaks, _ = scipy.signal.find_peaks(li)
        max_spectrum = heapq.nlargest(num_sources, li[peaks])
        for i in range(len(max_spectrum)):
            peaks_idx[i] = np.where(li == max_spectrum[i])[0][0]
            angle[i] = (
                li[int(peaks_idx[i] + 1)] / (li[int(peaks_idx[i] + 1)]
                                             + li[int(peaks_idx[i])]) * grids_mesh[int(peaks_idx[i] + 1)]
                + li[int(peaks_idx[i])] / (li[int(peaks_idx[i] + 1)]
                                           + li[int(peaks_idx[i])]) * grids_mesh[int(peaks_idx[i])]
                if li[int(peaks_idx[i] - 1)] < li[int(peaks_idx[i] + 1)]
                else li[int(peaks_idx[i] - 1)] / (li[int(peaks_idx[i] - 1)]
                                                  + li[int(peaks_idx[i])]) * grids_mesh[int(peaks_idx[i] - 1)]
                     + li[int(peaks_idx[i])] / (li[int(peaks_idx[i] - 1)]
                                                + li[int(peaks_idx[i])]) * grids_mesh[int(peaks_idx[i])]
            )
        angles[num] = angle.reshape(-1)
    if num_sources > 1:
        angles = np.sort(angles, axis=1)[::-1]
    return angles

def Spect2DoA_no_insert(Spectrum, num_sources=2, start_bias=60):
    """
    :param Spectrum: (num_samples, num_meshes, 1)
    :param num_sources:
    :param height_ignore:
    :param start_bias:
    :return: (num_samples, num_sources)
    """
    num_samples, num_meshes, _ = Spectrum.shape
    angles = np.zeros((num_samples, num_sources))
    grids_mesh = np.arange(num_meshes) - start_bias
    for num in range(num_samples):
        li_0 = Spectrum[num, :].reshape(-1)
        # li_0[li_0 < 0] = 0
        li = li_0
        angle = np.zeros(num_sources) - 5
        peaks, _ = scipy.signal.find_peaks(li)
        max_spectrum = heapq.nlargest(num_sources, li[peaks])
        for i in range(len(max_spectrum)):
            angle[i] = grids_mesh[np.where(li == max_spectrum[i])[0][0]]
        angles[num] = angle.reshape(-1)
        if num_sources > 1:
            angles = np.sort(angles, axis=1)[::-1]
    return angles

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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=25,
                                                           threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0.0001,
                                                           eps=1e-08)
    losses = []
    loss_min = 10
    name_file = 'L12-V2'
    for epc in range(epoch):
        if epc == 10:
            print(f'Time costs for 10 epc: {time.time() - time_start} seconds')

        # if epc <= 20 or epc % 10 == 0:
        #     torch.save({'model': model.state_dict()}, f"../../model/model_{epc}.pth")


        for idx_epc, (data_samples, fre_center, fre_fault, spacing_sample, label_theta, label_SNR) in enumerate(dataloader):

            args.antenna_distance = spacing_sample
            args.frequency_center = fre_center
            args.frequency_fault = fre_fault

            covariance_matrix_samples = torch.matmul(data_samples, data_samples.conj().transpose(2, 3)) / args.num_snapshots


            result_mulchannels, result, result_init_all = model(args, covariance_matrix_samples)

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

            label_all_layers = torch.zeros_like(result_init_all)  # shape of (batch_size, search_numbers, num_layers, num_meshes, 1)
            for search_idx in range(result_init_all.shape[1]):
                for layer_idx in range(result_init_all.shape[2]):
                    label_all_layers[:, search_idx, layer_idx, :, :] = label.reshape(-1, label.shape[-2], 1)

            # label = torch.from_numpy(label).to(torch.float32)

            # Calculate the sparse regularization term
            # lambda_sparse = 0.1
            # # loss_recovery = criterion(result, label)
            # loss_recovery = torch.nn.L1Loss()(result, label)
            # loss_sparse = torch.nn.L1Loss()(result_init_all, label_all_layers)
            # loss = (1-lambda_sparse) * loss_recovery + lambda_sparse * loss_sparse
            loss = torch.nn.MSELoss()(result_init_all[:,:,-1], label_all_layers[:,:,-1])

            if loss.item() <= loss_min:
                with open(f'../../Test/{name_file}/Weights/weights_best.txt', 'w') as f:
                    f.write(f"Epoch: {epc}-{idx_epc}, Loss: {loss.item()}, Lr: {optimizer.param_groups[0]['lr']} , \n ,Threshold: \n{[model.theta_amp[i].item() for i in range(len(model.theta_amp))]} \n , Step size: \n {[model.gamma_amp[i].item() for i in range(len(model.gamma_amp))]}")
                loss_min = loss.item()

            if idx_epc  == 3:
                x_label = [i-60 for i in range(label.shape[-2])]
                for chanel in range(result_mulchannels.shape[1]):
                    plt.plot(x_label, result_mulchannels[0, chanel].cpu().detach().numpy(), ls='-', alpha=0.5, linewidth=0.4)
                plt.plot(x_label, result[0].cpu().detach().numpy().reshape(-1), ls='-', label='Estimated Spectrum', color='k', linewidth=1.5)
                for xlabel_idx in range(label.shape[-2]):
                    if label[0, 0, xlabel_idx] == 1:
                        plt.axvline(x=xlabel_idx-60, color='r', linestyle='--', linewidth=1, label='Ground truth')
                plt.xlabel('Angles (Degrees)')
                plt.ylabel('Normalized Amplitude')
                # plt.plot(label[0].cpu().detach().numpy().reshape(-1), label='Label')
                plt.title(f"Output of {epc}-{idx_epc}")
                plt.legend()
                plt.savefig(f"../../Test/{name_file}/Figs_weights/{epc}-{idx_epc}.pdf")
                # plt.show()
                plt.close()
                print(
                    # f"Epoch: {epc}-{idx_epc}, Loss: {loss.item()}, lambda_sparse: {lambda_sparse}, \n, Loss recovery: {loss_recovery}, Loss sparse: {loss_sparse}, Costs Time: {time.time() - time_start} Seconds")
                f"Epoch: {epc}-{idx_epc}, Loss: {loss.item()}, Lr: {optimizer.param_groups[0]['lr']}, Costs Time: {time.time() - time_start} Seconds")
                with open(f'../../Test/{name_file}/Weights/weights_{epc}_{idx_epc}.txt', 'w') as f:
                    if epc == 0 and idx_epc == 0:
                        f.write(f"Time of start: {time_start}")
                    # f.write(f"Epoch: {epc}-{idx_epc}, Loss: {loss.item()}, lambda_sparse: {lambda_sparse}, Loss recovery: {loss_recovery}, Loss sparse: {loss_sparse}, \n ,Threshold: \n{[model.theta[i].item() for i in range(len(model.theta))]} \n , Step size: \n {[model.gamma[i].item() for i in range(len(model.gamma))]}")
                    f.write(f"Epoch: {epc}-{idx_epc}, Loss: {loss.item()}, Lr: {optimizer.param_groups[0]['lr']} , \n ,Threshold: \n{[model.theta_amp[i].item() for i in range(len(model.theta_amp))]} \n , Step size: \n {[model.gamma_amp[i].item() for i in range(len(model.gamma_amp))]}")


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            # print(f"Epoch: {epc}, Loss: {loss.item()}, Threshold: {model.theta.item()}, Step size: {model.gamma.item()}")
            losses.append(loss.item())
    with open(f'../../Test/{name_file}/Weights/loss.txt', 'w') as f:
        # f.write(f"Epoch: {epc}-{idx_epc}, Loss: {loss.item()}, lambda_sparse: {lambda_sparse}, Loss recovery: {loss_recovery}, Loss sparse: {loss_sparse}, \n ,Threshold: \n{[model.theta[i].item() for i in range(len(model.theta))]} \n , Step size: \n {[model.gamma[i].item() for i in range(len(model.gamma))]}")
        f.write(
            f"{losses}")

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
                plt.plot(result_mulchannels[0, chanel].cpu().detach().numpy(), ls='-', color='r', alpha=0.4, linewidth=0.5)
            plt.plot(result[0].cpu().detach().numpy().reshape(-1), label='Result', color='k', linewidth=1.5)
            for x_idx in range(label.shape[-2]):
                if label[0, 0, x_idx] == 1:
                    plt.axvline(x=x_idx, color='r', linestyle='--', linewidth=0.5)
            # plt.plot(label[0].cpu().detach().numpy().reshape(-1), label='Label')
            plt.legend()
            plt.show()


class expDataset(torch.utils.data.Dataset):

    def __init__(self, folder_path, num_antennas, num_snps, **kwargs):
        self.num_antennas = num_antennas
        self.num_snps = num_snps
        self.fre_sample = kwargs.get('fre_sample', 51200)
        self.target_fre = kwargs.get('target_fre', 5666)
        self.length_window = kwargs.get('length_window', 8192)
        self.target_fre_width = kwargs.get('target_fre_width', 0)
        self.is_half_overlapping = kwargs.get('is_half_overlapping', True)
        self.stride = kwargs.get('stride', 100)
        self.num_sources = kwargs.get('num_sources', 2)
        self.folder_path = folder_path
        self.is_saved = kwargs.get('is_saved', True)

        self.data, self.label = self.split_data(saved=self.is_saved, path_save=self.folder_path)
        ...

    # def make_snapshots_data(self, file_path) -> tuple:
    #     files = os.listdir(file_path)
    #     data_all = np.zeros((0, 0))
    #     label_all = np.zeros((0, 0))
    #     identity = file_path.split('/')[-1]
    #     for idx, file in enumerate(files):
    #         # DEBUG
    #         if identity in ['Complete', 'Inner', 'Outer', 'Ball', 'I2535']:
    #             label = [file.split('.')[0].split('_')[1], file.split('.')[0].split('_')[3]]
    #             label = list(map(int, label))
    #         elif identity == 'S5666':
    #             label = list(map(int, file.split('.')[0].split('_')))
    #         else:
    #             raise ValueError('The file name is not correct!')
    #         file_path = os.path.join(folder_name, file)
    #         data = np.load(file_path)
    #         snp_gen = GenSnapshot(data, self.fre_sample, self.target_fre, self.length_window,
    #                               target_fre_width=self.target_fre_width, is_half_overlapping=self.is_half_overlapping)
    #         data_snp = snp_gen.get_snapshots(num_antennas=self.num_antennas, num_snapshots=self.num_snps,
    #                                          stride=self.stride)
    #         if idx == 0:
    #             data_all = data_snp
    #             label = np.array(label).reshape(-1, self.num_sources).repeat(data_snp.shape[0], axis=0)
    #             label_all = label
    #         else:
    #             data_all = np.vstack((data_all, data_snp))
    #             label = np.array(label).reshape(-1, self.num_sources).repeat(data_snp.shape[0], axis=0)
    #             label_all = np.vstack((label_all, label))
    #         # release memory
    #         del data, data_snp
    #     return data_all, label_all


    def split_data(self, saved, **kwargs):
        path_save = kwargs.get('path_save', '../../Data/ULA_0.03/S5666')
        if not saved:
            # data_, label_ = self.make_snapshots_data(folder_name)
            # # save data to npy file
            # data_save = {
            #     'data': data_,
            #     'label': label_
            # }
            # np.save(os.path.join(path_save, 'data_snp_4372.npy'), data_save)
            ...
        else:
            data_save = np.load(os.path.join(path_save, 'data_snp.npy'), allow_pickle=True)
            data_ = data_save.item().get('data')
            label_ = data_save.item().get('label')
        return data_, label_

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.data.shape[0]










