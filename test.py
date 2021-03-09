from tensorpack.dataflow import DataFlow, RNGDataFlow, PrefetchDataZMQ, LMDBSerializer, BatchData
import lmdb
# ds = LMDBSerializer.load("/cw/liir/NoCsBack/testliir/nathan/p1_causality/DeVLBert/features_lmdb/CC/training_feat_all_debug_1613474882.lmdb")
# ds.reset_state()
# a = next(ds.get_data())



import pathlib
import torchvision
import torch
from torch import nn
import tensorpack
# from tensorpack import RNGDataFlow, PrefetchDataZMQ, LMDBSerializer
# import torch
# from random import random
# import os
#
# # NUM_CPUS = 2
# # import multiprocessing as mp
# #
# #
# # # mp.set_start_method('spawn')
# # class TestDataset(RNGDataFlow):
# #     """
# #     """
# #
# #     def __init__(self):
# #         DIM = 3
# #         NB_SAMPLES = 100000
# #         self.data = [[random() for _ in range(DIM)] for _ in range(NB_SAMPLES)]
# #         self.model = torch.nn.Linear(DIM, 1)
# #         self.model.cuda()
# #
# #     def __len__(self):
# #         return len(self.data)
# #
# #     def __iter__(self):
# #         for d in self.data:
# #             out = self.model(torch.tensor(d).cuda())
# #             yield out.cpu()
# #
# #
# # ds = TestDataset()
# # ds = PrefetchDataZMQ(ds, num_proc=NUM_CPUS)  # TODO get LMDB saving to speed up with parallelization via PrefetchDataZMQ
# # if os.path.exists("tmp_test.lmdb"):
# #     os.remove("tmp_test.lmdb")
# # LMDBSerializer.save(ds, "tmp_test.lmdb")
#
#
# import ray
# import time
# from argparse import Namespace
# import numpy as np
# from raw__img_text__to__lmdb_region_feats_text import CoCaDataFlow, setup
#
# IMAGE_DIR = '/cw/liir/NoCsBack/testliir/datasets/ConceptualCaptions/training'
# args = Namespace(bbox_dir='bbox',
#                  config_file='buatest/configs/bua-caffe/extract-bua-caffe-r101.yaml',
#                  extract_mode='roi_feats',
#                  gpu_id="'0,1,2,3'",
#                  image_dir=IMAGE_DIR,
#                  min_max_boxes='10,100',
#                  mode='caffe',
#                  mydebug=True,
#                  num_cpus=32,
#                  num_samples=0,
#                  opts=[],
#                  output_dir='features',
#                  resume=False
#                  )
#
#
#
#
# cfg = setup(args)
# ds = CoCaDataFlow(cfg, args)
# print(5)


import multiprocessing as mp


def get_parallel(idx_range, out_list):
    db = lmdb.open(f,
                   subdir=os.path.isdir(f),
                   readonly=True, lock=False, readahead=True,
                   map_size=1099511627776 * 2, max_readers=100)
    txn = db.begin()
    print(f"Running for idx_range {str(idx_range)}")
    for i in idx_range:
        out_list.append(txn.get(bb(i)))


s = time.time()
jobs = []
out_list = []
procs = 32
for i in range(procs):
    idx_range = range(10000)[cpu::NUM_CPUS]
    jobs.append(mp.Process(target=get_parallel, args=out_list, idx_range))
results = ray.get(futures)
print(time.time() - s)

s = time.time()
futures = []
NUM_CPUS = 32
db = lmdb.open(f,
               subdir=os.path.isdir(f),
               readonly=True, lock=False, readahead=True,
               map_size=1099511627776 * 2, max_readers=100)
txn = db.begin()
for o in range(10000):
    txn.get(bb(i))
results = ray.get(futures)
print(time.time() - s)


import multiprocessing as mp
import time
from tensorpack.utils.serialize import loads
import lmdb
f = "/cw/liir/NoCsBack/testliir/nathan/p1_causality/DeVLBert/features_lmdb/CC/full_coca.lmdb"
def bb(num: int):
    return u'{:08}'.format(num).encode('ascii')

def get_parallel(idx_range, q):
    db = lmdb.open(f,
                   subdir=os.path.isdir(f),
                   readonly=True, lock=False, readahead=True,
                   map_size=1099511627776 * 2, max_readers=100)
    txn = db.begin()
    print(f"Running for idx_range {str(idx_range)}")
    for i in idx_range:
        q.put(loads(txn.get(bb(i))))


s = time.time()
jobs = []
q = mp.Queue()
procs = 32
for i in range(procs):
    idx_range = range(1000)[i::procs]
    jobs.append(mp.Process(target=get_parallel, args=(idx_range, q)))

for j in jobs:
    j.start()
print("Joining processes")
for j in jobs:
    j.join()
print(time.time() - s)

s = time.time()
db = lmdb.open(f,
               subdir=os.path.isdir(f),
               readonly=True, lock=False, readahead=True,
               map_size=1099511627776 * 2, max_readers=100)
txn = db.begin()
out_list_serial = []
for o in range(1000):
    out_list_serial.append(loads(txn.get(bb(i))))
print(time.time() - s)

from multiprocessing import Process, Queue


def f(q, id):
    time.sleep(1)
    q.put([id, None, 'hello'])


q = Queue()

jobs = []
for p in range(32):
    j = Process(target=f, args=(q, p))
    jobs.append(j)
    j.start()
for j in jobs:
    print(q.get())  # prints "[42, None, 'hello']"
    j.join()
