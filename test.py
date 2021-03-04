from tensorpack.dataflow import DataFlow, RNGDataFlow, PrefetchDataZMQ, LMDBSerializer, BatchData

ds = LMDBSerializer.load("/cw/liir/NoCsBack/testliir/nathan/p1_causality/DeVLBert/features_lmdb/CC/training_feat_all_debug_1613474882.lmdb")
ds.reset_state()
a = next(ds.get_data())



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
