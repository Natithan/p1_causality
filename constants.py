from pathlib import Path

CHECKPOINT_FREQUENCY = 100
ROOT_DIR = "/cw/liir/NoCsBack/testliir/nathan/p1_causality"
STORAGE_DIR = Path(ROOT_DIR, "DeVLBert/features_lmdb/CC/")
BUA_ROOT_DIR = "buatest"
URL_PATH = Path(STORAGE_DIR, 'url_train.json')
LMDB_PATH = "/cw/working-arwen/nathan/features_CoCa_lmdb/full_coca_3.lmdb"