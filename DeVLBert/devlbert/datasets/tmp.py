# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
from constants import LMDB_PATH
lmdb_file = LMDB_PATH
import tensorpack.dataflow as td
from time import time as t

s = t()
ds = td.LMDBSerializer.load(LMDB_PATH, shuffle=False)
ds.reset_state()
print(t() - s)
s = t()
ds = td.LMDBSerializer.load(lmdb_file, shuffle=True)
print(t() - s)