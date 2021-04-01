import os

import re


def get_free_gpus():
    regex = r"MB \|(.*?)\n"
    processes = [re.search(regex, l).groups()[0] for l in os.popen('gpustat --no-header').readlines()]
    free_gpus = [i for i, p in enumerate(processes) if
                 all([('nathan' in s) for s in p.split()])]  # gpus where only my processes are running
    return free_gpus


def rank_to_device(rank):
    '''
    For when not all GPUs on the device can be used
    '''
    free_gpus = get_free_gpus()
    assert len(free_gpus) > rank, "Rank larger than number of available GPUs"
    return free_gpus[rank]