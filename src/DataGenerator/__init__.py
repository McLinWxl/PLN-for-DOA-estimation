import argparse


def args_data_generator():
    ARGS = argparse.Namespace(
        fault_type = 0,   # 0 for impulsive, 1 for tftp

        frequency_center=None,
        frequency_fault=107.4,  # must less than frequency_center / 10
        damping_ratio=0.1,
        time=17,  # 17

        number_sources = 1,

        amp_sample=20,

        # must greater than (if half-overlapping): (snapshot_length // 2 * (num_snapshots + 1)) / frequency_sampling
        frequency_sampling=51200,
        slipping_factor=0,
        # SNR=0,
        speed_of_sound=340,

        antenna_num=8,
        antenna_distance=0.03,  # speed_of_sound / (2*frequency_center),
        snapshot_length=8192,
        num_snapshots=100,
        search_numbers=5,

        SNR_source_min=-10,
        SNR_source_max=0,

        SNR_env_min=-10,
        SNR_env_max=0,

        theta_min=-50,
        theta_max=50,

        samples=100,  # different SNR and theta-50
        samples_repeat=1,  # different source waves-2
    )
    return ARGS
