# import time
#
# import numpy as np
# import os
# ROOT = '/cw/liir/NoCsBack/testliir/nathan/p1_causality'
# c = np.load(os.path.join(ROOT, "DeVLBert/dic", "id2class.npy"), allow_pickle=True).item()
# c1155_mine = np.load(os.path.join(ROOT, "DeVLBert/dic", "id2class1155_mine.npy"), allow_pickle=True).item()
# c1155_og = np.load(os.path.join(ROOT, "DeVLBert/dic", "id2class1155.npy"), allow_pickle=True).item()
# from pytorch_pretrained_bert.tokenization import BertTokenizer
#
# tokenizer = BertTokenizer.from_pretrained(
#     "bert-base-uncased", do_lower_case=True
# )
# id2word = {v: k for k, v in tokenizer.vocab.items()}
# nouns = [id2word[i] for i in c.keys()]
# nouns1155 = [id2word[i] for i in c1155_mine.keys()]
# nouns1155_og = [id2word[i] for i in c1155_og.keys()]
# objects = [o[:-1] for o in open(os.path.join(ROOT, "DeVLBert/dic","objects_vocab.txt"), "r")]
# import bnlearn as bn
#
# gt_dag_names_ex = ['titanic', 'sprinkler', 'alarm', 'andes', 'asia', 'pathfinder', 'sachs', 'water']
# gt_dag_names_dag = ['sprinkler', 'alarm', 'andes', 'asia', 'pathfinder', 'sachs', 'miserables']
# import signal
# def handler(signum, frame):
#      print("Forever is over!")
#      raise Exception("end of time")
# signal.signal(signal.SIGALRM, handler)
#
#
# # examples_dict = {k: list(bn.import_example(k).columns) for k in gt_dag_names_ex}
# examples_dict = {}
# for k in gt_dag_names_ex:
#     print(k)
#     signal.alarm(2)
#     try:
#         examples_dict[k] = list(bn.import_example(k).columns)
#         signal.alarm(0)
#     except Exception as e:
#         print(e)
#         signal.alarm(0)
#         continue
#
# DAG_dict = {}
# for k in gt_dag_names_dag:
#     print(k)
#     s = time.time()
#     DAG_dict[k] = list(bn.import_DAG(k,CPD=False)['model'].nodes)
#     print(time.time() - s)

# -*- coding: utf-8 -*-
# File: serialize.py
import pickle

import torch.distributed as dist
import lmdb
import numpy as np
import os
import platform
from collections import defaultdict

from tensorpack.utils import logger
from tensorpack.utils.serialize import dumps, loads
from tensorpack.utils.develop import create_dummy_class  # noqa
from tensorpack.utils.utils import get_tqdm
from tensorpack.dataflow.base import DataFlow
from tensorpack.dataflow.common import FixedSizeData, MapData
from tensorpack.dataflow.format import HDF5Data, LMDBData
from tensorpack.dataflow.raw import DataFromGenerator, DataFromList

__all__ = ['MyLMDBSerializer']

from preprocess_cfg import FGS
from constants import CHECKPOINT_FREQUENCY


def _reset_df_and_get_size(df):
    df.reset_state()
    try:
        sz = len(df)
    except NotImplementedError:
        sz = 0
    return sz

def enc(real_idx):
    return u'{:08}'.format(real_idx).encode('ascii')

class MyLMDBSerializer:
    """
    Serialize a Dataflow to a lmdb database, where the keys are indices and values
    are serialized datapoints.

    You will need to ``pip install lmdb`` to use it.

    Example:

    .. code-block:: python

        LMDBSerializer.save(my_df, "output.lmdb")

        new_df = LMDBSerializer.load("output.lmdb", shuffle=True)
    """

    @staticmethod
    def save(df, path, write_frequency=5000):
        """
        Args:
            df (DataFlow): the DataFlow to serialize.
            path (str): output path. Either a directory or an lmdb file.
            write_frequency (int): the frequency to write back data to disk.
                A smaller value reduces memory usage.
        """
        assert isinstance(df, DataFlow), type(df)
        isdir = os.path.isdir(path)
        if isdir:
            if FGS.from_scratch:
                assert not os.path.isfile(os.path.join(path, 'data.mdb')), "LMDB file exists!"
            else:
                print(f"Continuing in file {os.path.join(path, 'data.mdb')}")
        else:
            if FGS.from_scratch:
                assert not os.path.isfile(path), "LMDB file {} exists!".format(path)
            else:
                print(f"Continuing in file {path}")
        # It's OK to use super large map_size on Linux, but not on other platforms
        # See: https://github.com/NVIDIA/DIGITS/issues/206
        map_size = 1099511627776 * 2 if platform.system() == 'Linux' else 128 * 10**6
        db = lmdb.open(path, subdir=isdir,
                       map_size=map_size, readonly=False,
                       meminit=False, map_async=True)    # need sync() at the end
        size = _reset_df_and_get_size(df)

        # put data into lmdb, and doubling the size if full.
        # Ref: https://github.com/NVIDIA/DIGITS/pull/209/files
        def put_or_grow(txn, key, value):
            try:
                txn.put(key, value)
                return txn
            except lmdb.MapFullError:
                pass
            txn.abort()
            curr_size = db.info()['map_size']
            new_size = curr_size * 2
            logger.info("Doubling LMDB map_size to {:.2f}GB".format(new_size / 10**9))
            db.set_mapsize(new_size)
            txn = db.begin(write=True)
            txn = put_or_grow(txn, key, value)
            return txn

        txn = db.begin(write=True)
        previous_idx = txn.stat()['entries']
        if FGS.local_rank == 0:
            with get_tqdm(total=size,initial=df.non_batched_img_to_input_df.clean_count) as pbar:
                idx = 0
                real_idx = previous_idx + idx
                # LMDB transaction is not exception-safe!
                # although it has a context manager interface
                for idx, dp in enumerate(df):
                    real_idx = previous_idx + idx
                    txn = put_or_grow(txn, enc(real_idx), dumps(dp))
                    pbar.update()
                    if real_idx % write_frequency == 0:
                        txn.commit()
                        txn = db.begin(write=True)
                txn.commit()

                keys = [enc(k) for k in range(real_idx)]
                with db.begin(write=True) as txn:
                    txn = put_or_grow(txn, b'__keys__', dumps(keys))

                logger.info("Flushing database ...")
                db.sync()
        else:
            idx = 0
            real_idx = previous_idx + idx
            # LMDB transaction is not exception-safe!
            # although it has a context manager interface
            for idx, dp in enumerate(df):
                real_idx = previous_idx + idx
                txn = put_or_grow(txn, enc(real_idx), dumps(dp))
                # if (real_idx + 1) % write_frequency == 0:
                if real_idx % write_frequency == 0:
                    txn.commit()
                    txn = db.begin(write=True)
            txn.commit()
            keys = [enc(k) for k in range(real_idx)]
            with db.begin(write=True) as txn:
                txn = put_or_grow(txn, b'__keys__', dumps(keys))

            logger.info("Flushing database ...")
            db.sync()
        db.close()



    @staticmethod
    def load(path, shuffle=True):
        """
        Note:
            If you found deserialization being the bottleneck, you can use :class:`LMDBData` as the reader
            and run deserialization as a mapper in parallel.
        """
        # df = LMDBData(path, shuffle=shuffle)
        # #TODO if I end up storing multiple LMDB databases, this interleaved loading is not necessary anymore
        df = MyLMDBData(path, shuffle=shuffle,rank=rank,nb_processes=nb_processes)
        return MapData(df, MyLMDBSerializer._deserialize_lmdb)

    @staticmethod
    def _deserialize_lmdb(dp):
        return loads(dp[1])


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


class MyLMDBData(LMDBData):
    def __init__(self, lmdb_path, shuffle=True, keys=None, rank=None, nb_processes=None):
        self.rank = rank
        self.nb_processes = nb_processes
        super().__init__(lmdb_path, shuffle, keys)

    def _set_keys(self, keys=None):
        all_keys = loads(self._txn.get(b'__keys__'))
        self.keys = all_keys[self.rank::self.nb_processes]

    def __iter__(self):
        with self._guard:
            if self._shuffle:
                self.rng.shuffle(self.keys)
            for k in self.keys:
                v = self._txn.get(k)
                yield [k, v]

    def __len__(self):
        return len(self.keys)