import argparse


def args_data_generator():
    ARGS = argparse.Namespace(
        frequency_center=5000,
        frequency_fault=400,
        damping_ratio=0.1,
        time=1,
        frequency_sampling=51200,
        slipping_factor=0.01,
        SNR=10
    )
    return ARGS



