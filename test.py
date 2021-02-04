from tensorpack import RNGDataFlow, PrefetchDataZMQ, LMDBSerializer
import torch
from random import random
import os

NUM_CPUS = 2
import multiprocessing as mp


# mp.set_start_method('spawn')
class TestDataset(RNGDataFlow):
    """
    """

    def __init__(self):
        DIM = 3
        NB_SAMPLES = 100000
        self.data = [[random() for _ in range(DIM)] for _ in range(NB_SAMPLES)]
        self.model = torch.nn.Linear(DIM, 1)
        self.model.cuda()

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for d in self.data:
            out = self.model(torch.tensor(d).cuda())
            yield out.cpu()


ds = TestDataset()
ds = PrefetchDataZMQ(ds, num_proc=NUM_CPUS)  # TODO get LMDB saving to speed up with parallelization via PrefetchDataZMQ
if os.path.exists("tmp_test.lmdb"):
    os.remove("tmp_test.lmdb")
LMDBSerializer.save(ds, "tmp_test.lmdb")
