from logging import Logger, DEBUG

from time import time, strftime

import os
import pandas as pd
import ray

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
import numpy as np
from torch import distributed as dist
import torch
from constants import DEVLBERT_ROOT_DIR, HOST
from pretorch_util import rank_to_device
from DeVLBert.tools.DownloadConcptualCaption.download_data import _file_name
from pytorch_lightning.utilities.xla_device import XLADeviceUtils


def is_on_tpus() -> bool:
    return XLADeviceUtils.tpu_device_exists()


if is_on_tpus():
    import torch_xla.core.xla_model as xm
from datetime import datetime
from pytz import timezone

with open(f"{DEVLBERT_ROOT_DIR.as_posix()}/dic/objects_vocab.txt", "r") as vocab:
    CLASSES = ['background'] + [line.strip() for line in vocab]

IMAGE_DIR = "/cw/liir/NoCsBack/testliir/datasets/ConceptualCaptions/training"
TIME = round(time())


def word_to_id(word: str):
    return CLASSES.index(word) if word in CLASSES else None


def distributed(args) -> bool:
    return args.local_rank != -1


def myprint(*msg):
    pre_msg = _get_pre_msg()
    print(pre_msg, *msg)


def my_maybe_print(*msg):
    pass
    # pre_msg = _get_pre_msg()
    # print(pre_msg, *msg)


def _get_pre_msg():
    rank = get_rank()
    pre_msg = f'rank {rank} ' \
              f'pid {os.getpid()},' \
              f'{datetime.now(timezone("Europe/Brussels")).strftime("%d %H:%M:%S")}: '
    return pre_msg


class MyLogger(Logger):
    def debug(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'DEBUG'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.debug("Houston, we have a %s", "thorny problem", exc_info=1)
        """
        rank = get_rank()
        msg = f"rank {rank}: " + msg
        if self.isEnabledFor(DEBUG):
            self._log(DEBUG, msg, args, **kwargs)


def show_from_tuple(rpn_tuple):
    _, cls_probs, bboxes, _, _, _, img_id, caption = rpn_tuple

    image = Image.open(os.path.join(IMAGE_DIR, img_id))
    plt.imshow(np.asarray(image))
    # Get the current reference
    ax = plt.gca()  # Create a Rectangle patch
    colors = ['black', 'darkblue', 'darkmagenta']
    for b, probs in zip(bboxes, cls_probs):
        color = random.choice(colors)
        rect = Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], linewidth=1, edgecolor=color, facecolor='none')
        cls = CLASSES[np.argmax(probs)]
        ax.text(b[0], b[1] - 12 / 2, cls, color=color)
        # Add the patch to the Axes
        ax.add_patch(rect)
    plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment='center', fontsize=12)
    plt.show()


@ray.remote
def index_df_column(dataframe, df_column):
    col_for_ids = {}
    for i, img in enumerate(dataframe.iterrows()):
        col = img[1][df_column]  # .decode("utf8")
        img_name = _file_name(img[1])
        image_id = img_name.split('/')[1]
        # image_id = str(i)
        col_for_ids[image_id] = col
    return col_for_ids


def open_tsv(fname, folder):
    print("Opening %s Data File..." % fname)
    df = pd.read_csv(fname, sep='\t', names=["caption", "url"], usecols=range(0, 2))
    df['folder'] = folder
    print("Processing", len(df), " Images:")
    return df


def get_world_size() -> int:
    if not is_on_tpus():
        if not dist.is_available():
            return 1
        if not dist.is_initialized():
            return 1
        return dist.get_world_size()
    else:
        return xm.xrt_world_size()


def get_rank() -> int:
    if not is_on_tpus():
        if not dist.is_available():
            return 0
        if not dist.is_initialized():
            return 0
        return torch.distributed.get_rank()
    else:
        return xm.get_ordinal()


def is_master_rank() -> bool:
    return get_rank() == 0




def setup(rank, world_size,tasks=None):
    os.environ['MASTER_ADDR'] = 'localhost'
    if not tasks is None:
        print(tasks)
        if '0' in tasks:
            os.environ['MASTER_PORT'] = '12365'
        else:
            os.environ['MASTER_PORT'] = '12366'
    else:
        os.environ['MASTER_PORT'] = '1236z5'

    print("*" * 100, os.environ['MASTER_PORT'], rank, world_size, "*" * 100)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # # initialize the process group
    # try:
    #     print(os.environ['MASTER_PORT'],rank,world_size)
    #     dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # except RuntimeError:
    #     print("SWITCHING MASTER PORT, ALREADY IN USE")
    #     os.environ['MASTER_PORT'] = '12356'
    #     print(os.environ['MASTER_PORT'],rank,world_size)
    #     dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def sz(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def print_gpu_mem():
    if HOST == 'VSC':
        totmem = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r - a  # free inside reserved
        myprint(f""
                f"Total mem: {sz(totmem)}\t"
                f"Reserved mem: {sz(r)}\t"
                f"Allocated mem: {sz(a)}\t"
                f"Free inside reserved: {sz(f)}"
                f"")
