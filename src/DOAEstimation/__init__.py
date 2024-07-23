#  Copyright (c) 2024. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

import argparse


def args_doa():
    ARGS = argparse.Namespace(
        frequency_center=6000,
        frequency_fault=500,  # must less than frequency_center / 10
        damping_ratio=0.1,
        time=9,  # 20

        amp_sample=20,

        # must greater than (if half-overlapping): (snapshot_length // 2 * (num_snapshots + 1)) / frequency_sampling
        frequency_sampling=51200,
        slipping_factor=0.03,
        # SNR=0,
        speed_of_sound=340,

        antenna_num=8,
        antenna_distance=0.03,  # speed_of_sound / (2*frequency_center),
        snapshot_length=8192,
        num_snapshots=100,
        search_numbers=9,

        SNR_source_min=-10,
        SNR_source_max=0,

        SNR_env_min=-10,
        SNR_env_max=0,

        theta_min=-60,
        theta_max=60,
        num_meshes=121,

        samples=50,  # different SNR and theta
        samples_repeat=2,  # different source waves
    )
    return ARGS

def args_unfolding_doa():
    ARGS = argparse.Namespace(
        num_layers=10,
        device='cpu',

    )
    return ARGS

