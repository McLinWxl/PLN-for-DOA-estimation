import argparse


def args_data_generator():
    ARGS = argparse.Namespace(
        frequency_center=5000,
        frequency_fault=400,  # must less than frequency_center / 10
        damping_ratio=0.1,
        time=21,  # 20
        # must greater than (if half-overlapping): (snapshot_length // 2 * (num_snapshots + 1)) / frequency_sampling
        frequency_sampling=51200,
        slipping_factor=0.01,
        SNR=0,
        speed_of_sound=340,

        antenna_num=8,
        antenna_distance=0.034,  # speed_of_sound / (2*frequency_center),
        snapshot_length=8192,
        num_snapshots=256,
        search_numbers=9,

        SNR_min=-10,
        SNR_max=0,
        theta_min=-60,
        theta_max=60,
        samples=2,

        samples_repeat=3,
    )
    return ARGS
