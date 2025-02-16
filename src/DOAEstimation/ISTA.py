#  Copyright (c) 2024. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.utils.data
from functions import DatasetGeneration_sample, cal_steer_vector, cal_covariance, plot_sample, cal_dictionary, \
    train_proposed
from __init__ import args_doa, args_unfolding_doa

args = args_doa()
args_unfolding = args_unfolding_doa()
import time


class ISTA(torch.nn.Module):
    def __init__(self, alpha=1):
        super(ISTA, self).__init__()
        self.args = args
        self.num_sensors = args.antenna_num
        self.num_search = args.search_numbers
        M2 = self.num_sensors ** 2
        self.num_grids = args.num_meshes
        self.M2 = M2
        self.num_layers = args_unfolding.num_layers
        self.device = args_unfolding.device
        self.alpha = alpha

        self.relu = torch.nn.ReLU()

    def forward(self, paras, covariance_matrix):
        """
        :param alpha:
        :param paras:
        :param covariance_vector: (batch_size, search_numbers, M2, 1) -> (1, 9, 64, 1)
        :return:
        """
        covariance_vector = covariance_matrix.transpose(2, 3).reshape(covariance_matrix.shape[0],
                                                                      covariance_matrix.shape[1],
                                                                      args.antenna_num ** 2, 1)

        num_batch = covariance_vector.shape[0]
        assert covariance_vector.shape[1] == self.args.search_numbers and covariance_vector.shape[2] == self.M2 and \
               covariance_vector.shape[3] == 1
        # covariance_vector = covariance_vector / torch.linalg.matrix_norm(covariance_vector, ord=np.inf, dim=2, keepdim=True)
        covariance_vector = covariance_vector.to(torch.complex64)
        # covariance_vector = covariance_vector / torch.linalg.norm(covariance_vector, ord=np.inf, dim=2, keepdim=True)
        spacing_sample = paras.antenna_distance
        fre_center = paras.frequency_center
        fre_fault = paras.frequency_fault

        self.args.antenna_distance = spacing_sample
        self.args.frequency_center = fre_center
        self.args.frequency_fault = fre_fault

        # make high dim dictionary, shape: (search_numbers, M2, num_grids) -> (9, 64, 121)
        narrow_band = torch.Tensor(
            [int(self.args.frequency_center[i] / 20) for i in range(len(self.args.frequency_center))]).to(self.device)
        dictionary_band = torch.zeros((num_batch, self.args.search_numbers, self.M2, self.num_grids),
                                      dtype=torch.complex64, device=self.device)
        for bat in range(num_batch):
            for i in range(self.args.search_numbers):
                fre_center_nb = self.args.frequency_center[bat] + (i - self.args.search_numbers // 2) * narrow_band
                dictionary_band[bat, i] = torch.from_numpy(cal_dictionary(fre_center_nb[bat], self.args, bat))

        # Forward
        result_init = torch.matmul(dictionary_band.conj().transpose(2, 3), covariance_vector).real.float()
        # result_init = result_init / (torch.norm(result_init, dim=2) + 1e-20).reshape(-1)
        result_init = torch.div(result_init,
                                torch.norm(result_init, dim=2).reshape(result_init.shape[0], result_init.shape[1], 1,
                                                                       result_init.shape[3]) + 1e-20)
        result_init_all = torch.zeros(result_init.shape[0], self.args.search_numbers, self.num_layers, self.num_grids,
                                      1).to(self.device)

        result = result_init
        identity_matrix = (torch.eye(self.num_grids) + 1j * torch.zeros([self.num_grids, self.num_grids])).to(
            self.device)

        gamma_init_from_dictionary = 1 / torch.max(torch.abs(torch.linalg.eigvals(torch.matmul(dictionary_band.conj().transpose(2, 3), dictionary_band))), dim=2, keepdim=True)[0]

        lambda_ = 0.5 * torch.max(torch.abs(torch.matmul(dictionary_band.conj().transpose(2, 3), covariance_vector)), dim=2, keepdim=True)[0].reshape(num_batch, self.args.search_numbers, 1)

        theta_init_from_dictionary = self.alpha * lambda_ * gamma_init_from_dictionary

        for i in range(self.num_layers):

            gamma = gamma_init_from_dictionary.reshape(num_batch, self.args.search_numbers, 1, 1)
            theta = theta_init_from_dictionary.reshape(num_batch, self.args.search_numbers, 1, 1)

            Wt = identity_matrix - gamma * torch.matmul(dictionary_band.conj().transpose(2, 3), dictionary_band)
            We = gamma * dictionary_band.conj().transpose(2, 3)

            s = torch.matmul(Wt, result.to(torch.complex64)) + torch.matmul(We, covariance_vector)
            # s = s / (torch.norm(s, dim=2, keepdim=True) + 1e-20)
            s_abs = torch.abs(s)
            # TODO: theta is trainable, and relu can be replaced.
            # TODO: Soft-threshold can be replaced a neural model.

            # s_abs = (s_abs - torch.min(s_abs, dim=2, keepdim=True)[0]) / (
            #             torch.max(s_abs, dim=2, keepdim=True)[0] - torch.min(s_abs, dim=2, keepdim=True)[0] + 1e-20)

            result = self.relu(s_abs - theta)

            # result = (result - torch.min(result, dim=2, keepdim=True)[0]) / (
            #             torch.max(result, dim=2, keepdim=True)[0] - torch.min(result, dim=2, keepdim=True)[0] + 1e-20)

            # result = result / (torch.norm(result, dim=2, keepdim=True) + 1e-20)
            result_init_all[:, :, i] = result
        # norm every channel of result to [0, 1]
        # result = result / (torch.norm(result, dim=2) + 1e-20).reshape(result_init.shape[0], result_init.shape[1], 1, result_init.shape[3])
        # result = torch.nn.functional.softmax(result, dim=2)
        result_ave = torch.mean(result, dim=1, keepdim=True).to(torch.float32)
        # result_ave = self.combination_module(result)

        # normalize result_ave to [0, 1]
        result_ave = (result_ave - torch.min(result_ave, dim=2, keepdim=True)[0]) / (
                    torch.max(result_ave, dim=2, keepdim=True)[0] - torch.min(result_ave, dim=2, keepdim=True)[
                0] + 1e-20)
        result = (result - torch.min(result, dim=2, keepdim=True)[0]) / (
                    torch.max(result, dim=2, keepdim=True)[0] - torch.min(result, dim=2, keepdim=True)[0] + 1e-20)

        return result, result_ave, result_init_all


if __name__ == '__main__':
    dataset_ld = torch.load('../../data/data2train.pt')
    print(f"Dataset length: {len(dataset_ld)}")
    train_loader = torch.utils.data.DataLoader(dataset_ld, batch_size=50, shuffle=True)

    model = ISTA()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # # criterion = torch.nn.MSELoss()
    # criterion = torch.nn.MSELoss()
    #
    # epoch = 8
    #
    # losses = train_proposed(model=model, epoch=epoch, dataloader=train_loader, optimizer=optimizer, criterion=criterion,
    #                         args=args)
    # # save losses as csv file
    # np.savetxt('../../Test/losses.csv', losses, delimiter=',')
    #
    # plt.style.use(['science', 'ieee', 'grid'])
    # plt.figure(dpi=800)
    # plt.plot(losses)
    # plt.xlabel('Batch')
    # plt.ylabel('Loss')
    # plt.title('Training Loss')
    # plt.savefig('../../Test/Figures/TrainingLoss.pdf')
    # plt.show()

    # read a sample
