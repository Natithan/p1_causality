import os

import re
import sys

def get_free_gpus():
    regex = r"MB \|(.*?)\n"
    processes = [re.search(regex, l) for l in os.popen('gpustat --no-header').readlines()]
    free_gpus = [i for i, p in enumerate(processes) if p is not None and
                 all([('nathan' in s) for s in p.groups()[0].split()])]  # gpus where only my processes are running
    return free_gpus


def get_really_free_gpus():  # Not even my processes
    regex = r"MB \|(.*?)\n"
    processes = [re.search(regex, l) for l in os.popen('gpustat --no-header').readlines()]
    # free_gpus = [i for i, p in enumerate(processes) if p is not None and
    #              all([('nathan' in s) for s in p.groups()[0].split()])]
    free_gpus = [i for i, p in enumerate(processes) if p is not None and
                 (len(p.groups()[0].split()) == 0)]
    return free_gpus


def rank_to_device(rank):
    '''
    For when not all GPUs on the device can be used
    '''
    free_gpus = get_free_gpus()
    assert len(free_gpus) > rank, "Rank larger than number of available GPUs"
    return free_gpus[rank]


def assign_visible_gpus():
    # This needs to happen before torch / pytorch lightning import statements
    if '--visible_gpus' in sys.argv and sys.argv[sys.argv.index('--visible_gpus') + 1]:
        print("=" * 100, f"Manually setting os.environ['CUDA_VISIBLE_DEVICES'] to {sys.argv[sys.argv.index('--visible_gpus') + 1]}",
              "=" * 100)
        os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[sys.argv.index('--visible_gpus') + 1]
    else:
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            print(os.environ['CUDA_VISIBLE_DEVICES'])
        else:
            print("os.environ['CUDA_VISIBLE_DEVICES'] not set")
            os.environ['CUDA_VISIBLE_DEVICES'] = ",".join([str(i) for i in get_free_gpus()])
            print("Now set to", os.environ['CUDA_VISIBLE_DEVICES'])