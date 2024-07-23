#  Copyright (c) 2024. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.utils.data
from functions import DatasetGeneration_sample, cal_steer_vector, cal_covariance, plot_sample, cal_dictionary, train_proposed
from __init__ import args_doa, args_unfolding_doa
args = args_doa()
args_unfolding = args_unfolding_doa()
import time


class proposedMethods(torch.nn.Module):
    def __init__(self):
        super(proposedMethods, self).__init__()
        self.args = args
        self.num_sensors = args.antenna_num
        self.num_search = args.search_numbers
        M2 = self.num_sensors ** 2
        self.num_grids = args.num_meshes
        self.M2 = M2
        self.num_layers = args_unfolding.num_layers
        self.device = args_unfolding.device

        # self.theta = torch.nn.Parameter(torch.Tensor([0.001]), requires_grad=True) # smaller
        # self.gamma = torch.nn.Parameter(torch.Tensor([0.001]), requires_grad=True)

        self.leakly_relu = torch.nn.LeakyReLU(0.01)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self.conv1 = torch.nn.Conv1d(in_channels=self.num_search, out_channels=18, kernel_size=3, stride=1, padding='same')
        self.batch_norm_conv1 = torch.nn.BatchNorm1d(18)
        self.conv2 = torch.nn.Conv1d(in_channels=18, out_channels=12, kernel_size=3, stride=1, padding='same')
        self.batch_norm_conv2 = torch.nn.BatchNorm1d(12)
        self.conv3 = torch.nn.Conv1d(in_channels=12, out_channels=self.num_search, kernel_size=3, stride=1, padding='same')
        self.batch_norm_conv3 = torch.nn.BatchNorm1d(9)

        assert self.M2 == 64
        # self.conv1_ =
        self.linear1 = torch.nn.Linear(in_features=self.num_grids, out_features=int(self.num_grids*self.num_grids / 15))
        self.batch_norm_linear1 = torch.nn.BatchNorm1d(int(self.num_grids*self.num_grids / 15))
        self.linear2 = torch.nn.Linear(in_features=int(self.num_grids*self.num_grids / 15), out_features=int(self.num_grids*self.num_grids / 10))
        self.batch_norm_linear2 = torch.nn.BatchNorm1d(int(self.num_grids*self.num_grids / 10))
        self.linear3 = torch.nn.Linear(in_features=int(self.num_grids*self.num_grids / 10), out_features=self.num_grids*self.num_grids)
        self.batch_norm_linear3 = torch.nn.BatchNorm1d(self.num_grids*self.num_grids)

    def thresholding_module(self, x):
        """
        Apply thresholding to the input data
        :param x: shape of (batch_size, search_numbers, num_grids, 1)
        :return: shape of (batch_size, search_numbers, num_grids, 1)
        """
        x_shortcut = x
        search_num = x.shape[1]
        x = x.reshape(-1, search_num, self.num_grids)
        # x = self.leakly_relu(x)
        x = self.conv1(x)
        x = self.batch_norm_conv1(x)
        x = self.leakly_relu(x)
        x = self.conv2(x)
        x = self.batch_norm_conv2(x)
        x = self.leakly_relu(x)
        x = self.conv3(x)
        x = self.batch_norm_conv3(x)
        x = x.reshape(-1, search_num, self.num_grids, 1)
        x = self.relu(x)
        x = x_shortcut - x
        x = self.relu(x)
        return x

    def step_module(self, x_):
        """
        Apply thresholding to the input data
        :param x: shape of (batch_size, search_numbers, num_grids, 1)
        :return: shape of (batch_size, search_numbers, M2, M2)
        """
        batch_size, search_numbers, num_grids, _ = x_.shape
        gamma = (torch.zeros(batch_size, search_numbers, num_grids, num_grids) + 1j * torch.zeros(batch_size, search_numbers, num_grids, num_grids)).to(self.device)
        for i in range(search_numbers):
            x = x_[:, i]
            x = x.reshape(-1, num_grids)
            x = self.linear1(x)
            x = self.batch_norm_linear1(x)
            x = self.sigmoid(x)
            x = self.linear2(x)
            x = self.batch_norm_linear2(x)
            x = self.sigmoid(x)
            x = self.linear3(x)
            x = self.batch_norm_linear3(x)
            x = x.reshape(-1, num_grids, num_grids)
            x = self.relu(x)
            gamma[:, i] = x
        return gamma

    # def combination_module(self, x):
    #     """
    #     Combine the multi-channles to calculate a final output, using attention module
    #     :param x: shape of (batch_size, search_numbers, num_grids, 1)
    #     :return: shape of (batch_size, num_grids, 1)
    #     """
    #     x_shortcut = x
    #     x = self.conv1(x)
    #     x = self.leakly_relu(x)
    #     x = self.conv2(x)
    #     x = self.leakly_relu(x)
    #     x = self.conv3(x)
    #     x = x + torch.mean(x_shortcut, dim=1, keepdim=True)
    #     x = x - torch.mean(x).item()
    #     x = self.relu(x)
    #     return x

    def forward(self, paras, covariance_vector):
        """
        :param paras:
        :param covariance_vector: (batch_size, search_numbers, M2, 1) -> (1, 9, 64, 1)
        :return:
        """
        num_batch = covariance_vector.shape[0]
        assert covariance_vector.shape[1] == self.args.search_numbers and covariance_vector.shape[2] == self.M2 and covariance_vector.shape[3] == 1
        # covariance_vector = covariance_vector / torch.linalg.matrix_norm(covariance_vector, ord=np.inf, keepdim=True)
        covariance_vector = covariance_vector.to(torch.complex64)
        spacing_sample = paras.antenna_distance
        fre_center = paras.frequency_center
        fre_fault = paras.frequency_fault

        self.args.antenna_distance = spacing_sample
        self.args.frequency_center = fre_center
        self.args.frequency_fault = fre_fault

        # make high dim dictionary, shape: (search_numbers, M2, num_grids) -> (9, 64, 121)
        narrow_band = torch.Tensor([int(self.args.frequency_center[i] / 15) for i in range(len(self.args.frequency_center))]).to(self.device)
        dictionary_band = torch.zeros((num_batch, self.args.search_numbers, self.M2, self.num_grids), dtype=torch.complex64, device=self.device)
        for bat in range(num_batch):
            for i in range(self.args.search_numbers):
                fre_center_nb = self.args.frequency_center[bat] + (i - self.args.search_numbers // 2) * narrow_band
                dictionary_band[bat, i] = torch.from_numpy(cal_dictionary(fre_center_nb[bat], self.args, bat))

        # Forward
        result_init = torch.matmul(dictionary_band.conj().transpose(2, 3), covariance_vector).real.float()
        # result_init = result_init / (torch.norm(result_init, dim=2) + 1e-20).reshape(-1)
        result_init = torch.div(result_init, torch.norm(result_init, dim=2).reshape(result_init.shape[0], result_init.shape[1], 1, result_init.shape[3]) + 1e-20)
        result_init_all = torch.zeros(result_init.shape[0], self.args.search_numbers, self.num_layers, self.num_grids, 1).to(self.device)

        result = result_init
        identity_matrix = (torch.eye(self.num_grids) + 1j * torch.zeros([self.num_grids, self.num_grids])).to(
            self.device)
        gamma_mat = self.step_module(result_init)
        for i in range(self.num_layers):
            # TODO: gamma is trainable. It should be learned from the former layer.
            # Wt = identity_matrix - gamma * torch.matmul(dictionary_band.conj().transpose(2, 3), dictionary_band)
            # We = gamma * dictionary_band.conj().transpose(2, 3)
            Wt = identity_matrix - torch.matmul(gamma_mat, torch.matmul(dictionary_band.conj().transpose(2, 3), dictionary_band))
            We = torch.matmul(gamma_mat, dictionary_band.conj().transpose(2, 3))
            s = torch.matmul(Wt, result + 1j * torch.zeros_like(result)) + torch.matmul(We, covariance_vector)
            s = s / (torch.norm(s, dim=2, keepdim=True) + 1e-20)
            s_abs = torch.abs(s)
            gamma_mat = self.step_module(s_abs)
            # TODO: theta is trainable, and relu can be replaced.
            # TODO: Soft-threshold can be replaced a neural model.
            # result = self.relu(s_abs - self.theta)
            result = self.thresholding_module(s_abs)
            # result = result / (torch.norm(result, dim=1, keepdim=True) + 1e-20)
            result_init_all[:, :, i] = result
        # norm every channel of result to [0, 1]
        result = result / (torch.norm(result, dim=2) + 1e-20).reshape(result_init.shape[0], result_init.shape[1], 1, result_init.shape[3])
        # result = torch.nn.functional.softmax(result, dim=2)
        result_ave = torch.mean(result, dim=1, keepdim=True).to(torch.float32)
        # result_ave = self.combination_module(result)

        # normalize result_ave to [0, 1]
        result_ave = (result_ave - torch.min(result_ave)) / (torch.max(result_ave) - torch.min(result_ave) + 1e-20)

        return result, result_ave, result_init_all



if __name__ == '__main__':
    dataset_ld = torch.load('../../data/data2train_new.pt')
    print(f"Dataset length: {len(dataset_ld)}")
    train_loader = torch.utils.data.DataLoader(dataset_ld, batch_size=100, shuffle=True)

    model = proposedMethods()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)
    # criterion = torch.nn.MSELoss()
    criterion = torch.nn.MSELoss()



    epoch = 100

    losses = train_proposed(model=model, epoch=epoch, dataloader=train_loader, optimizer=optimizer, criterion=criterion, args=args)
    # save losses as csv file
    np.savetxt('../../Test/losses.csv', losses, delimiter=',')

    plt.style.use(['science', 'ieee', 'grid'])
    plt.figure(dpi=800)
    plt.plot(losses)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('../../Test/Figures/TrainingLoss.pdf')
    plt.show()

    # read a sample
