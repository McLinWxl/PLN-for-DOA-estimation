#  Copyright (c) 2024. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

import torch
from proposedMethods import proposedMethods
from functions import test_proposed, DatasetGeneration_sample
from __init__ import args_doa, args_unfolding_doa
args = args_doa()
args_unfolding = args_unfolding_doa()

if __name__ == '__main__':
    dataset_ld = torch.load('../../data/data2train_new.pt')
    print(f"Dataset length: {len(dataset_ld)}")
    test_loader = torch.utils.data.DataLoader(dataset_ld, batch_size=100, shuffle=True)

    model = proposedMethods()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # # criterion = torch.nn.MSELoss()
    # criterion = torch.nn.MSELoss()

    # epoch = 100
    checkpoint = ('../../model/model.pth')
    losses = test_proposed(model, checkpoint, test_loader, args)

