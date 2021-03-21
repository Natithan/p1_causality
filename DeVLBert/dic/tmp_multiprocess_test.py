# from multiprocessing import Queue, Process
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorpack.dataflow as td
from constants import LMDB_PATH
import lmdb
import os
from tensorpack.dataflow.serialize import loads
from tqdm import tqdm
# NB_PROCS = 8 #TODO fix: one process at a time = 40000 it/s, 32 processes at a time = 4 it/s
# env = lmdb.open(LMDB_PATH,
#                subdir=os.path.isdir(LMDB_PATH),
#                readonly=True, lock=False, readahead=True,
#                map_size=1099511627776 * 2, max_readers=100)
# txn = env.begin()
# all_keys = loads(txn.get(b'__keys__'))
# del txn
# def run(q, p):
#     txn = env.begin()
#     keys = all_keys[p::NB_PROCS]
#     total_bbox_count = 0
#     iterator = tqdm(enumerate(keys)) if (p == 0) else enumerate(keys)
#     for i,k in iterator:
#         bbox_count = loads(txn.get(k))[-5]
#         total_bbox_count += bbox_count
#         if i % 1000 == 0:
#             print(f"process {p} has done {i}")
#     q.put(total_bbox_count)
#
#
#
# pool = []
# q = Queue()
# for i in range(NB_PROCS):
#     process = Process(target=run, args=(q, i))
#     pool.append(process)
#
# for process in pool:
#     process.start()
# total_bbox_count = 0
# for i in range(NB_PROCS):
#     partial_count = q.get()
#     total_bbox_count += partial_count
#
# for process in pool:
#     process.join()
# print("join Done")
#
# env.close()
from time import  time as t
import concurrent.futures
import math

PRIMES = [
    112272535095293,
    112582705942171,
    112272535095293,
    115280095190773,
    115797848077099,
    1099726899285419]

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    sqrt_n = int(math.floor(math.sqrt(n)))
    for i in range(3, sqrt_n + 1, 2):
        if n % i == 0:
            return False
    return True


NB_PROCS = os.cpu_count()

def run(p,keys):
    env = lmdb.open(LMDB_PATH,
                    subdir=os.path.isdir(LMDB_PATH),
                    readonly=True, lock=False, readahead=True,
                    map_size=1099511627776 * 2, max_readers=100)
    txn = env.begin()
    total_bbox_count = 0
    iterator = tqdm(enumerate(keys)) if (p == 0) else enumerate(keys)
    for i,k in iterator:
        bbox_count = loads(txn.get(k))[-5]
        total_bbox_count += bbox_count
        if i % 1000 == 0:
            print(f"process {p} has done {i}")
    return total_bbox_count

def main():
    env = lmdb.open(LMDB_PATH,
                subdir=os.path.isdir(LMDB_PATH),
                readonly=True, lock=False, readahead=True,
                map_size=1099511627776 * 2, max_readers=100)
    txn = env.begin()
    all_keys = loads(txn.get(b'__keys__'))[:50000]
    split_keys = [
        all_keys[p::NB_PROCS] for p in range(NB_PROCS)
    ]
    s = t()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for sum in executor.map(run, range(NB_PROCS),split_keys):
            print(sum)
    print(t()-s)
    s = t()
    print(run(0,all_keys))
    print(t()-s)
if __name__ == '__main__':
    main()