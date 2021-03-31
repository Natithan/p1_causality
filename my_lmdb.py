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
import itertools

import sys

import pickle
import zmq
from pathlib import Path
import lmdb
import os
import platform
import torch
from tensorpack.dataflow.parallel import _repeat_iter, _ExceptionWrapper, _zmq_catch_error, _get_pipe_name, _bind_guard
from tensorpack.utils.concurrency import enable_death_signal
from time import time as t
from typing import List
from tensorpack.utils import logger, get_rng
from tensorpack.utils.serialize import dumps, loads
from tensorpack.utils.develop import create_dummy_class  # noqa
from tensorpack.utils.utils import get_tqdm
from tensorpack.dataflow.base import DataFlow, DataFlowReentrantGuard
from tensorpack.dataflow import BatchData, MultiProcessRunnerZMQ
from tensorpack.dataflow.common import MapData
from tensorpack.dataflow.format import LMDBData
from util import get_rank, get_world_size, MyLogger, myprint, get_core_ds
import glob
import logging

import multiprocessing as mp

logging.setLoggerClass(MyLogger)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger('__main__')
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


class MyBatchData(BatchData):
    """
    Subclass for profiling.

    OG:
    Stack datapoints into batches.
    It produces datapoints of the same number of components as ``ds``, but
    each component has one new extra dimension of size ``batch_size``.
    The batch can be either a list of original components, or (by default)
    a numpy array of original components.
    """

    def __iter__(self):
        """
        Yields:
            Batched data by stacking each component on an extra 0th dimension.
        """
        holder = []
        s = t()
        times = []
        sd = t()
        for data in self.ds:
            times.append(t() - sd)
            holder.append(data)
            if len(holder) == self.batch_size:
                myprint(f"BatchData iteration took {t() - s}")
                myprint(f"Iterating through {self.batch_size} elements of {self.ds} took {sum(times)}")
                times = []
                ss = t()
                aggregated_batch = BatchData.aggregate_batch(holder, self.use_list)
                myprint(f"BatchData.aggregate_batch took {t() - ss}")
                yield aggregated_batch
                s = t()
                del holder[:]
            sd = t()
        if self.remainder and len(holder) > 0:
            yield BatchData.aggregate_batch(holder, self.use_list)


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
    def save(df, path, write_frequency=5000, args=None):
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
    def load(paths: List[str], shuffle=True, savePath=None):
        """
        Note:
            If you found deserialization being the bottleneck, you can use :class:`LMDBData` as the reader
            and run deserialization as a mapper in parallel.
        """
        # df = LMDBData(path, shuffle=shuffle)
        # #TODO if I end up storing multiple LMDB databases, this interleaved loading is not necessary anymore
        if savePath is None:
            df = MyNonResamplingLMDBData(paths, shuffle=shuffle, savePath=savePath)
        else:
            df = MyLMDBData(paths, shuffle=shuffle)
        return MapData(df, MyLMDBSerializer._deserialize_lmdb)

    @staticmethod
    def _deserialize_lmdb(dp):
        return loads(dp[1])


class MyLMDBData(LMDBData):
    def __init__(self, lmdb_paths: List[str], shuffle=True, keys=None):
        self.rank = get_rank()
        self.nb_processes = get_world_size()

        self._lmdb_paths = lmdb_paths
        self._shuffle = shuffle

        self._open_lmdbs()
        self._sizes = [txn.stat()['entries'] for txn in self._txns]

        self.rng = get_rng(self)
        logger.info("Found {} entries in {}".format([s for s in self._sizes], [p for p in self._lmdb_paths]))

        # Clean them up after finding the list of keys, since we don't want to fork them
        self._close_lmdbs()

    def _set_keys(self, keys=None):

        all_keys_list = [loads(txn.get(b'__keys__')) for txn in self._txns]
        self.keys_list = [akeys[self.rank::self.nb_processes] for akeys in all_keys_list]
        if self._shuffle:
            self.shuffle_keys()

    def shuffle_keys(self):
        [self.rng.shuffle(ks) for ks in self.keys_list]  # Modifies in-place

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

    def reset_state(self):
        self._guard = DataFlowReentrantGuard()
        self.rng = get_rng(self)
        self.shuffle_keys()
        self._open_lmdbs()  # open the LMDB in the worker process

    def __iter__(self):
        ys = t()

        y_times = []
        BATCH_SIZE = 96
        with self._guard:
            for j, (txn, keys) in enumerate(zip(self._txns, self.keys_list)):
                for i, k in enumerate(keys):
                    v = txn.get(k)
                    y_times.append(t() - ys)
                    if i % BATCH_SIZE == 0:
                        myprint(f"MyLMDBData between-yield time for {BATCH_SIZE} elements was {sum(y_times)}")
                        y_times = []
                    yield [k, v]
                    ys = t()

    def __len__(self):
        return sum(len(ks) for ks in self.keys_list)



class MyMultiProcessRunnerZMQ(MultiProcessRunnerZMQ):
    """
    Run a DataFlow in >=1 processes, with ZeroMQ for communication.
    It will fork the calling process of :meth:`reset_state()`,
    and collect datapoints from the given dataflow in each process by ZeroMQ IPC pipe.
    This is typically faster than :class:`MultiProcessRunner`.

    Note:
        1. (Data integrity) An iterator cannot run faster automatically -- what's happening is
           that the process will be forked ``num_proc`` times.
           There will be ``num_proc`` dataflow running in parallel and **independently**.
           As a result, we have the following guarantee on the dataflow correctness:

           a. When ``num_proc=1``, this dataflow produces the same data as the
              given dataflow in the same order.
           b. When ``num_proc>1``, if each sample from the given dataflow is i.i.d.,
              then this dataflow produces the **same distribution** of data as the given dataflow.
              This implies that there will be duplication, reordering, etc.
              You probably only want to use it for training.

              For example, if your original dataflow contains no randomness and produces the same first datapoint,
              then after parallel prefetching, the datapoint will be produced ``num_proc`` times
              at the beginning.
              Even when your original dataflow is fully shuffled, you still need to be aware of the
              `Birthday Paradox <https://en.wikipedia.org/wiki/Birthday_problem>`_
              and know that you'll likely see duplicates.

           To utilize parallelism with more strict data integrity, you can use
           the parallel versions of :class:`MapData`: :class:`MultiThreadMapData`, :class:`MultiProcessMapData`.
        2. `reset_state()` of the given dataflow will be called **once and only once** in the worker processes.
        3. The fork of processes happened in this dataflow's `reset_state()` method.
           Please note that forking a TensorFlow GPU session may be unsafe.
           If you're managing this dataflow on your own,
           it's better to fork before creating the session.
        4. (Fork-safety) After the fork has happened, this dataflow becomes not fork-safe.
           i.e., if you fork an already reset instance of this dataflow,
           it won't be usable in the forked process. Therefore, do not nest two `MultiProcessRunnerZMQ`.
        5. (Thread-safety) ZMQ is not thread safe. Therefore, do not call :meth:`get_data` of the same dataflow in
           more than 1 threads.
        6. This dataflow does not support windows. Use `MultiProcessRunner` which works on windows.
        7. (For Mac only) A UNIX named pipe will be created in the current directory.
           However, certain non-local filesystem such as NFS/GlusterFS/AFS doesn't always support pipes.
           You can change the directory by ``export TENSORPACK_PIPEDIR=/other/dir``.
           In particular, you can use somewhere under '/tmp' which is usually local.

           Note that some non-local FS may appear to support pipes and code
           may appear to run but crash with bizarre error.
           Also note that ZMQ limits the maximum length of pipe path.
           If you hit the limit, you can set the directory to a softlink
           which points to a local directory.
    """

    class _Worker(mp.Process):
        def __init__(self, ds, conn_name, hwm, idx, num_workers):
            super(MyMultiProcessRunnerZMQ._Worker, self).__init__()
            self.ds = ds
            core_ds = get_core_ds(self.ds)
            core_ds.set_key_state_files(idx, num_workers)
            self.conn_name = conn_name
            self.hwm = hwm
            self.idx = idx

        def run(self):
            enable_death_signal(_warn=self.idx == 0)
            self.ds.reset_state()
            get_core_ds(self.ds)
            itr = _repeat_iter(lambda: self.ds)

            context = zmq.Context()
            socket = context.socket(zmq.PUSH)
            socket.set_hwm(self.hwm)
            socket.connect(self.conn_name)
            try:
                while True:
                    try:
                        dp = next(itr)
                        socket.send(dumps(dp), copy=False) # Nathan: assuming this blocks when queue is filled till hwm
                    except Exception:
                        dp = _ExceptionWrapper(sys.exc_info()).pack()
                        socket.send(dumps(dp), copy=False)
                        raise
            # sigint could still propagate here, e.g. when nested
            except KeyboardInterrupt:
                pass
            finally:
                socket.close(0)
                context.destroy(0)

    def __init__(self, ds, num_proc=1, hwm=50):
        """
        Args:
            ds (DataFlow): input DataFlow.
            num_proc (int): number of processes to use.
            hwm (int): the zmq "high-water mark" (queue size) for both sender and receiver.
        """
        super(MultiProcessRunnerZMQ, self).__init__()

        self.ds = ds
        self.num_proc = num_proc
        self._hwm = hwm

        if num_proc > 1:
            logger.info("[MultiProcessRunnerZMQ] Will fork a dataflow more than one times. "
                        "This assumes the datapoints are i.i.d.")
        try:
            self._size = ds.__len__()
        except NotImplementedError:
            self._size = -1

    def _recv(self):
        ret = loads(self.socket.recv(copy=False))
        exc = _ExceptionWrapper.unpack(ret)
        if exc is not None:
            logger.error("Exception '{}' in worker:".format(str(exc.exc_type)))
            raise exc.exc_type(exc.exc_msg)
        return ret

    def __len__(self):
        return self.ds.__len__()

    def __iter__(self):
        with self._guard, _zmq_catch_error('MultiProcessRunnerZMQ'):
            for k in itertools.count():
                if self._size > 0 and k >= self._size:
                    break
                yield self._recv()

    def reset_state(self):
        super(MultiProcessRunnerZMQ, self).reset_state()
        self._guard = DataFlowReentrantGuard()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.set_hwm(self._hwm)
        pipename = _get_pipe_name('dataflow')
        _bind_guard(self.socket, pipename)

        self._procs = [MyMultiProcessRunnerZMQ._Worker(self.ds, pipename, self._hwm, idx, self.num_proc)
                       for idx in range(self.num_proc)]
        self._start_processes()


class MyNonResamplingLMDBData(LMDBData):
    def __init__(self, lmdb_paths: List[str], shuffle=True, keys=None, savePath=MODEL_CKPT_DIR):
        raise NotImplementedError
        self.rank = get_rank()
        self.nb_processes = get_world_size()
        self.savePath = savePath

        self._lmdb_paths = lmdb_paths
        self._shuffle = shuffle

        self._open_lmdbs()
        self._sizes = [txn.stat()['entries'] for txn in self._txns]

        self.rng = get_rng(self)
        logger.info("Found {} entries in {}".format([s for s in self._sizes], [p for p in self._lmdb_paths]))
        self.key_indexes_checkpoint_dir =  Path(self.savePath,
                                                f"key_index_rank{self.rank}_wsize{self.nb_processes}")
        self.shuffled_keys_list_checkpoint_dir = Path(self.savePath,
                                                       f"shuffled_keys_rank{self.rank}_wsize{self.nb_processes}")
        for directory in (self.key_indexes_checkpoint_dir, self.shuffled_keys_list_checkpoint_dir):
            if not Path.exists(directory):
                Path.mkdir(directory)
        # Clean them up after finding the list of keys, since we don't want to fork them
        self._close_lmdbs()


    def set_key_state_files(self, worker_rank, num_workers):
        '''
        # __init__ only happens for all workers at once. This happens per worker
        '''
        # Data is stored in multiple LMDB files. # of processes doesn't necessarily match number of LMDB files:
        # might have multiple LMDB files per process, or multiple processes per LMDB file
        worker_identifying_addendum = f"workerrank{worker_rank}_workerwsize{num_workers}"
        self.key_indexes_checkpoint_path = Path(self.key_indexes_checkpoint_dir,
                                                worker_identifying_addendum),
        self.shuffled_keys_list_checkpoint_path = Path(self.shuffled_keys_list_checkpoint_dir,
                                                       worker_identifying_addendum)
        self._set_keys()
        self._set_key_start_idxs()

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
                [self.rng.shuffle(ks) for ks in self.keys_list]  # Modifies in-place
            with open(self.shuffled_keys_list_checkpoint_path, 'wb') as f:
                pickle.dump(self.keys_list, f)
        # self.keys = all_keys[self.rank::self.nb_processes]# Changed to having different lmdb files.
        # self.keys_list = all_keys

    def _set_key_start_idxs(self):
        self.current_key_idx_list = [0] * len(self._lmdbs)
        p = self.get_existing_key_idxs_file_path()
        if p is not None:
            print(f"Loading key indexes from existing files {p}")
            self.load_idxs_with_value_in_name_path()
        else:
            print('\r\nNo key index found: starting from scratch')
            self.start_key_idxs = [0] * len(self._lmdbs)
            self.dump_idxs_with_value_in_name(self.start_key_idxs)

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

    def reset_state(self):
        self._guard = DataFlowReentrantGuard()
        self.rng = get_rng(self)
        self._open_lmdbs()  # open the LMDB in the worker process

    # def reset_index(self):
    #     self.reset_state()
    #
    #     # Reset index in keys (no need to restore keys themselves)
    #     self.start_keys = [0]*len(self._lmdbs)
    #     self.current_key_idx_list = [0]*len(self._lmdbs)
    #     self.dump_idxs_with_value_in_name(self.start_keys)

    def dump_idxs_with_value_in_name(self, key_idxs):
        # Remove previous checkpoint
        old_path = self.get_existing_key_idxs_file_path()
        if old_path is not None:
            os.remove(old_path)
        new_path = self.key_indexes_checkpoint_path.as_posix() + f'_{str(key_idxs)}'
        with open(new_path, 'wb') as f:
            pickle.dump(key_idxs, f)

    def load_idxs_with_value_in_name_path(self):
        path = self.get_existing_key_idxs_file_path()
        assert path is not None
        with open(path, 'rb') as f:
            self.start_key_idxs = pickle.load(f)

        myprint(f"Starting at indexes {[sk for sk in self.start_key_idxs]} out of {[len(ks) for ks in self.keys_list]}")

    def get_existing_key_idxs_file_path(self):
        res = self.get_key_idxs_path_list()
        assert (len(res) <= 1)
        new_path = res[0] if len(res) != 0 else None
        return new_path

    def get_key_idxs_path_list(self):
        search_regex = self.key_indexes_checkpoint_path.as_posix() + f'_*'
        res = glob.glob(search_regex)
        return res

    def store_checkpoint(self):
        myprint(self.current_key_idx_list)
        self.dump_idxs_with_value_in_name(self.current_key_idx_list)

    def __iter__(self):
        ys = t()

        y_times = []
        BATCH_SIZE = 96
        with self._guard:
            for j, (txn, keys, start_key) in enumerate(zip(self._txns, self.keys_list, self.start_key_idxs)):
                for i, k in enumerate(keys[start_key:]):
                    if self.mini:
                        if start_key + i > 100:
                            break
                    self.current_key_idx_list[j] = start_key + i
                    v = txn.get(k)
                    y_times.append(t() - ys)
                    if i % BATCH_SIZE == 0:
                        myprint(f"MyLMDBData between-yield time for {BATCH_SIZE} elements was {sum(y_times)}")
                        y_times = []
                    yield [k, v]
                    ys = t()

    def __len__(self):
        return sum(len(ks) for ks in self.keys_list)

    def set_mini(self, mini):
        self.mini = mini
