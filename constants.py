import os

from pathlib import Path
import socket

CHECKPOINT_FREQUENCY = 1000
hostname = socket.gethostname()
CKPT_DIR_NAME = 'devlbert_checkpunten'
if hostname == 'sceps':  # GCP:
    CORE_ROOT_DIR = "/home/nathan_e_d_cornille_gmail_com"
    DATA_ROOT_DIR = "/mnt/disks/concap"  #
    HOST = 'GCP'
    # "/home/nathan_e_d_cornille_gmail_com/tmp_data_root_dir"
    # "/mnt/disks/ssd_concap"
    # "/mnt/disks/concap"
    MODEL_CKPT_DIR = f"{CORE_ROOT_DIR}/p1_causality/{CKPT_DIR_NAME}"
elif hostname in ['arwen', 'frodo', 'gimli', 'rose', 'sauron']:  # LIIR-servers
    CORE_ROOT_DIR = "/cw/liir/NoCsBack/testliir/nathan"
    DATA_ROOT_DIR = "/cw/working-gimli/nathan/features_CoCa_lmdb"
    MODEL_CKPT_DIR = f"/cw/working-gimli/nathan/{CKPT_DIR_NAME}"
    HOST = 'LIIR'
    FINETUNE_DATA_ROOT_DIR = "/cw/working-gimli/nathan/downstream_data"
else:  # VSC
    print(hostname)
    CORE_ROOT_DIR = "/data/leuven/336/vsc33642/"
    DATA_ROOT_DIR = "/scratch/leuven/336/vsc33642/features_CoCa_lmdb"
    MODEL_CKPT_DIR = f"/scratch/leuven/336/vsc33642/{CKPT_DIR_NAME}"
    HOST = 'VSC'
    FINETUNE_DATA_ROOT_DIR = "/scratch/leuven/336/vsc33642/downstream_datasets"

PROJECT_ROOT_DIR = Path(CORE_ROOT_DIR, "p1_causality")
DEVLBERT_ROOT_DIR = Path(PROJECT_ROOT_DIR, "DeVLBert")

STORAGE_DIR = Path(PROJECT_ROOT_DIR, "DeVLBert/features_lmdb/CC/")
URL_PATH = Path(STORAGE_DIR, 'url_train.json')

BUA_ROOT_DIR = "buatest"
OLD_10_100_LMDB_PATH = "/cw/working-arwen/nathan/features_CoCa_lmdb/full_coca_3.lmdb"
# LMDB_PATHS = [f"/export/home1/NoCsBack/hci/nathan/features_CoCa_lmdb/full_coca_{i}_of_4.lmdb" for i in range(4)]
ARWEN_LMDB_PATHS = [f"/cw/working-arwen/nathan/features_CoCa_lmdb/full_coca_36_{i}_of_4.lmdb" for i in range(4)]
LMDB_PATHS = [f"{DATA_ROOT_DIR}/full_coca_36_{i}_of_4.lmdb" for i in range(4)]
MINI_LMDB_PATHS = [f"{DATA_ROOT_DIR}/mini_coca_36_{i}_of_4.lmdb" for i in range(4)]
CENTI_LMDB_PATHS = [f"{DATA_ROOT_DIR}/centi_coca_36_{i}_of_4.lmdb" for i in range(4)]
CAPTION_PATH = f"{DATA_ROOT_DIR}/caption_train.json"
ID2CLASS_PATH = Path(DEVLBERT_ROOT_DIR, "dic", "id2class1141.npy")
MTURK_DIR = Path(PROJECT_ROOT_DIR, "mturk")


def next_path(path_pattern):  # From https://stackoverflow.com/a/47087513/6297057
    """
    Finds the next free path in an sequentially named list of files

    e.g. path_pattern = 'file-%s.txt':

    file-1.txt
    file-2.txt
    file-3.txt

    Runs in log(n) time where n is the number of existing files in sequence
    """
    i = 1

    # First do an exponential search
    while os.path.exists(path_pattern % i):
        i = i * 2

    # Result lies somewhere in the interval (i/2..i]
    # We call this interval (a..b] and narrow it down until a + 1 = b
    a, b = (i // 2, i)
    while a + 1 < b:
        c = (a + b) // 2  # interval midpoint
        a, b = (c, b) if os.path.exists(path_pattern % c) else (a, c)

    return path_pattern % b


MEM_PROF_DIR = 'mem_prof_logs'
os.makedirs(MEM_PROF_DIR, exist_ok=True)
PROFILING_LOG_FILE_HANDLE = open(next_path(os.path.join(MEM_PROF_DIR, 'v_%s.log')), 'a')


def memprof_log_handle_for_name(name):
    return open(next_path(os.path.join(MEM_PROF_DIR, f'{name}_v_%s.log')), 'a')
