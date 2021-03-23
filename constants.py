from pathlib import Path

CHECKPOINT_FREQUENCY = 1000
ROOT_DIR = "/cw/liir/NoCsBack/testliir/nathan/p1_causality"
DEVLBERT_ROOT_DIR = Path(ROOT_DIR, "DeVLBert")
STORAGE_DIR = Path(ROOT_DIR, "DeVLBert/features_lmdb/CC/")
BUA_ROOT_DIR = "buatest"
URL_PATH = Path(STORAGE_DIR, 'url_train.json')
OLD_10_100_LMDB_PATH = "/cw/working-arwen/nathan/features_CoCa_lmdb/full_coca_3.lmdb"
LMDB_PATHS = [f"/export/home1/NoCsBack/hci/nathan/features_CoCa_lmdb/full_coca_{i}_of_4.lmdb" for i in range(4)]
MODEL_CKPT_DIR = "/cw/working-arwen/nathan/devlbert_ckpts"
CAPTION_PATH = "/cw/working-arwen/nathan/features_CoCa_lmdb/caption_train.json"
ID2CLASS_PATH = Path(ROOT_DIR,"DeVLBert","dic","id2class1141.npy")
MTURK_DIR = Path(ROOT_DIR,"mturk")