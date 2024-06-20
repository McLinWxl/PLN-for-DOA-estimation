import argparse


def args_data_generator():
    ARGS = argparse.Namespace(
        frequency_center=5000,
        frequency_fault=400,
        damping_ratio=0.1,
        time=1,
        frequency_sampling=51200,
        slipping_factor=0.01,
        SNR=0,
        speed_of_sound=340,

        anrenna_num=8,
        antenna_distance=0.034, # speed_of_sound / (2*frequency_center),
        snapshot_length=8192,
        search_numbers=4
    )
    return ARGS



