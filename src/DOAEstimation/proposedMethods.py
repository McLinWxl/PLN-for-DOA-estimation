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


class proposedMethods(torch.nn.Module):
    def __init__(self):
        super(proposedMethods, self).__init__()
        self.args = args
        self.num_sensors = args.antenna_num
        M2 = self.num_sensors ** 2
        self.num_grids = args.num_meshes
        self.M2 = M2
        self.num_layers = args_unfolding.num_layers
        self.device = args_unfolding.device

        self.theta = torch.nn.Parameter(0.0001 * torch.ones(self.num_layers), requires_grad=True) # smaller
        self.gamma = torch.nn.Parameter(0.0001 * torch.ones(self.num_layers), requires_grad=True)
        self.leakly_relu = torch.nn.LeakyReLU()
        self.relu = torch.nn.ReLU()
        ...
    def forward(self, paras, covariance_vector):
        """
        :param paras:
        :param covariance_vector: (batch_size, search_numbers, M2, 1) -> (1, 9, 64, 1)
        :return:
        """
        assert covariance_vector.shape[1] == self.args.search_numbers and covariance_vector.shape[2] == self.M2 and covariance_vector.shape[3] == 1
        # covariance_vector = covariance_vector / torch.linalg.matrix_norm(covariance_vector, ord=np.inf, keepdim=True)
        covariance_vector = covariance_vector.to(torch.complex64)
        spacing_sample = paras['antenna_distance']
        fre_center = paras['frequency_center']
        fre_fault = paras['frequency_fault']

        self.args.antenna_distance = spacing_sample
        self.args.frequency_center = fre_center
        self.args.frequency_fault = fre_fault

        # make high dim dictionary, shape: (search_numbers, M2, num_grids) -> (9, 64, 121)
        narrow_band = int(args.frequency_center / 15)
        dictionary_band = torch.zeros((self.args.search_numbers, self.M2, self.num_grids), dtype=torch.complex64, device=self.device)
        for i in range(self.args.search_numbers):
            fre_center_nb = self.args.frequency_center + (i - self.args.search_numbers // 2) * narrow_band
            dictionary_band[i] = torch.from_numpy(cal_dictionary(fre_center_nb, self.args))

        # Forward
        result_init = torch.matmul(dictionary_band.conj().transpose(1, 2), covariance_vector).real.float()
        result_init_all = torch.zeros(result_init.shape[0], self.args.search_numbers, self.num_layers, self.num_grids, 1).to(self.device)

        result = result_init
        for i in range(self.num_layers):
            identity_matrix = (torch.eye(self.num_grids) + 1j * torch.zeros([self.num_grids, self.num_grids])).to(self.device)
            # TODO: gamma is trainable.
            Wt = identity_matrix - self.gamma[i] * torch.matmul(dictionary_band.conj().transpose(1, 2), dictionary_band)
            We = self.gamma[i] * dictionary_band.conj().transpose(1, 2)
            s = torch.matmul(Wt, result + 1j * torch.zeros_like(result)) + torch.matmul(We, covariance_vector)
            s_abs = torch.abs(s)
            # TODO: theta is trainable, and relu can be replaced.
            result = self.relu(s_abs - self.theta[i])
            # result = result / (torch.norm(result, dim=1, keepdim=True) + 1e-20)
            result_init_all[:, :, i] = result
        # norm every channel of result to [0, 1]
        # result = result / (torch.norm(result) + 1e-20)
        result = torch.nn.functional.softmax(result, dim=2)



        return result, result_init_all



if __name__ == '__main__':
    dataset_ld = torch.load('../../data/data2train_2750_9channels.pt')
    print(f"Dataset length: {len(dataset_ld)}")
    train_loader = torch.utils.data.DataLoader(dataset_ld, batch_size=1, shuffle=True)

    model = proposedMethods()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)
    criterion = torch.nn.MSELoss()

    epoch = 1

    losses = train_proposed(model=model, epoch=epoch, dataloader=train_loader, optimizer=optimizer, criterion=criterion, args=args)
    # save losses as csv file
    np.savetxt('../../Test/losses.csv', losses, delimiter=',')

    plt.style.use(['science', 'ieee', 'grid'])
    plt.figure(dip=800)
    plt.plot(losses)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('../../Test/Figures/TrainingLoss.pdf')
    plt.show()

    # read a sample
