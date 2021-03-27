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
from pathlib import Path
import lmdb
import os
import platform
import torch
from typing import List
from tensorpack.utils import logger, get_rng
from tensorpack.utils.serialize import dumps, loads
from tensorpack.utils.develop import create_dummy_class  # noqa
from tensorpack.utils.utils import get_tqdm
from tensorpack.dataflow.base import DataFlow, DataFlowReentrantGuard
from tensorpack.dataflow.common import MapData
from tensorpack.dataflow.format import LMDBData
from util import rank, world_size
import glob

__all__ = ['MyLMDBSerializer']

from constants import MODEL_CKPT_DIR
NB_SAVED_SPLITS = 4

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
    def save(df, path, write_frequency=5000,args=None):
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
            if args.from_scratch:
                assert not os.path.isfile(os.path.join(path, 'data.mdb')), "LMDB file exists!"
            else:
                print(f"Continuing in file {os.path.join(path, 'data.mdb')}")
        else:
            if args.from_scratch:
                assert not os.path.isfile(path), "LMDB file {} exists!".format(path)
            else:
                print(f"Continuing in file {path}")
        # It's OK to use super large map_size on Linux, but not on other platforms
        # See: https://github.com/NVIDIA/DIGITS/issues/206
        map_size = 1099511627776 * 2 if platform.system() == 'Linux' else 128 * 10 ** 6
        db = lmdb.open(path, subdir=isdir,
                       map_size=map_size, readonly=False,
                       meminit=False, map_async=True)  # need sync() at the end
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
            logger.info("Doubling LMDB map_size to {:.2f}GB".format(new_size / 10 ** 9))
            db.set_mapsize(new_size)
            txn = db.begin(write=True)
            txn = put_or_grow(txn, key, value)
            return txn

        txn = db.begin(write=True)
        previous_idx = txn.stat()['entries']
        if args.local_rank == 0:
            with get_tqdm(total=size, initial=df.non_batched_img_to_input_df.clean_count) as pbar:
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
    def load(paths : List[str], shuffle=True,savePath=MODEL_CKPT_DIR):
        """
        Note:
            If you found deserialization being the bottleneck, you can use :class:`LMDBData` as the reader
            and run deserialization as a mapper in parallel.
        """
        # df = LMDBData(path, shuffle=shuffle)
        # #TODO if I end up storing multiple LMDB databases, this interleaved loading is not necessary anymore
        df = MyLMDBData(paths, shuffle=shuffle,savePath=savePath)
        return MapData(df, MyLMDBSerializer._deserialize_lmdb)

    @staticmethod
    def _deserialize_lmdb(dp):
        return loads(dp[1])


class MyLMDBData(LMDBData):
    def __init__(self, lmdb_paths : List[str], shuffle=True, keys=None, savePath=MODEL_CKPT_DIR):
        self.rank = rank()
        self.nb_processes = world_size()
        self.savePath = savePath

        # Data is stored in multiple LMDB files. # of processes doesn't necessarily match number of LMDB files:
        # might have multiple LMDB files per process, or multiple processes per LMDB file
        self.key_indexes_checkpoint_path = Path(self.savePath, f"key_index_{self.rank}_of_{self.nb_processes}.pickle")
        self.shuffled_keys_list_checkpoint_path = Path(self.savePath, f"shuffled_keys_{self.rank}_of_{self.nb_processes}.pickle")
        self._lmdb_paths = lmdb_paths
        self._shuffle = shuffle

        self._open_lmdbs()
        self._sizes = [txn.stat()['entries'] for txn in self._txns]
        self.rng = get_rng(self)
        self._set_keys(keys)
        logger.info("Found {} entries in {}".format([s for s in self._sizes], [p for p in self._lmdb_paths]))
        self.current_key_idx_list = [0]*len(self._lmdbs)
        self.maybe_load_checkpoint()

        # Clean them up after finding the list of keys, since we don't want to fork them
        self._close_lmdbs()

    def _open_lmdbs(self):
        self._lmdbs = [lmdb.open(p,
                               subdir=os.path.isdir(p),
                               readonly=True, lock=False, readahead=True,
                               map_size=1099511627776 * 2, max_readers=100)
                       for p in self._lmdb_paths
                       ]
        self._txns = [env.begin() for env in self._lmdbs]

    def _close_lmdbs(self):
        for env in self._lmdbs:
            env.close()
        del self._lmdbs
        del self._txns


    def _set_keys(self, keys=None):
        if Path.exists(self.shuffled_keys_list_checkpoint_path):
            print(f"Loading keys_list from existing file {self.shuffled_keys_list_checkpoint_path}")
            with open(self.shuffled_keys_list_checkpoint_path, 'rb') as f:
                self.keys_list = pickle.load(f)
        else:
            print(f'\r\nNo keys found, shuffling and storing at {self.shuffled_keys_list_checkpoint_path}')
            all_keys_list = [loads(txn.get(b'__keys__')) for txn in self._txns]
            self.keys_list = [akeys[self.rank::self.nb_processes] for akeys in all_keys_list]
            if self._shuffle:
                [self.rng.shuffle(ks) for ks in self.keys_list] # Modifies in-place
            with open(self.shuffled_keys_list_checkpoint_path, 'wb') as f:
                pickle.dump(self.keys_list, f)
        # self.keys = all_keys[self.rank::self.nb_processes]# Changed to having different lmdb files.
        # self.keys_list = all_keys

    def reset_state(self):
        self._guard = DataFlowReentrantGuard()
        self.rng = get_rng(self)
        self._open_lmdbs()  # open the LMDB in the worker process


    def reset_index(self):
        self.reset_state()

        # Reset index in keys (no need to restore keys themselves)
        self.start_keys = [0]*len(self._lmdbs)
        self.current_key_idx_list = [0]*len(self._lmdbs)
        self.dump_idxs_with_value_in_name(self.start_keys)

    def dump_idxs_with_value_in_name(self, keys):
        # Remove previous checkpoint
        old_path = self.get_existing_idxs_with_value_in_name_path()
        if old_path is not None:
            os.remove(old_path)
        new_path = self.key_indexes_checkpoint_path.as_posix()[:-(len('.pickle'))] + f'_{str(keys)}.pickle'
        with open(new_path, 'wb') as f:
            pickle.dump(keys, f)

    def load_idxs_with_value_in_name_path(self):
        path = self.get_existing_idxs_with_value_in_name_path()
        assert path is not None
        with open(path, 'rb') as f:
            self.start_keys = pickle.load(f)

        print(f"Starting at indexes {[sk for sk in self.start_keys]} out of {[len(ks) for ks in self.keys_list]}")

    def get_existing_idxs_with_value_in_name_path(self):
        search_regex = self.key_indexes_checkpoint_path.as_posix()[:-(len('.pickle'))] + f'_*.pickle'
        res = glob.glob(search_regex)
        assert (len(res) <= 1)
        new_path = res[0] if len(res) != 0 else None
        return new_path

    def maybe_load_checkpoint(self):
        p = self.get_existing_idxs_with_value_in_name_path()
        if p is not None:
            print(f"Loading key indexes from existing files {p}")
            self.load_idxs_with_value_in_name_path()
            print(f"Starting at indexes {[sk for sk in self.start_keys]} out of {[len(ks) for ks in self.keys_list]}")
        else:
            print('\r\nNo key index found: starting from scratch')
            self.start_keys = [0]*len(self._lmdbs)
            self.dump_idxs_with_value_in_name(self.start_keys)

    def store_checkpoint(self):
        self.dump_idxs_with_value_in_name(self.current_key_idx_list)

    def __iter__(self):
        with self._guard:
            for j, (txn, keys, start_key) in enumerate(zip(self._txns, self.keys_list, self.start_keys)):
                for i, k in enumerate(keys[start_key:]):
                    if self.mini:
                        if start_key + i > 100:
                            break
                    self.current_key_idx_list[j] = start_key + i
                    v = txn.get(k)
                    yield [k, v]

    def __len__(self):
        return sum(len(ks) for ks in self.keys_list)

    def set_mini(self, mini):
        self.mini = mini