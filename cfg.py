from time import time

from pathlib import Path

import sys
from absl import flags

ROOT_DIR = "/cw/liir/NoCsBack/testliir/nathan/p1_causality"

FGS = flags.FLAGS
flags.DEFINE_list("pixel_mean", [102.9801, 115.9465, 122.7717], "")
flags.DEFINE_string("image_dir", "/cw/liir/NoCsBack/testliir/datasets/ConceptualCaptions/training", "")
flags.DEFINE_integer("max_nb_images", -1, "")
flags.DEFINE_integer("batch_size", 6, "")
flags.DEFINE_integer("num_cpus", 32, "")
flags.DEFINE_string("gpus", "0,1,2,3", "")
flags.DEFINE_list("min_max_boxes", [0, 1, 2, 3], "")
flags.DEFINE_string("extract_mode", "roi_feats", "")
flags.DEFINE_string("config_file", "buatest/configs/bua-caffe/extract-bua-caffe-r101.yaml", "")
flags.DEFINE_string("mode", "caffe", "Flag to allow python console command line argument")
flags.DEFINE_bool("verbose", False, "")
flags.DEFINE_bool("resume", False, "")
flags.DEFINE_bool("from_scratch", False, "")
flags.DEFINE_string("index_file", "input_index.p", "")
flags.DEFINE_string("bbox_dir", "bbox", "directory with bbox")
flags.DEFINE_list("opts", [], "Modify config options using the command-line")
flags.DEFINE_integer("num_samples", 0, "number of samples to convert, 0 means all")
flags.DEFINE_string("lmdb_file", f'training_feat_debug_{int(time())}', "")
flags.DEFINE_string("lmdb_folder", "/cw/liir/NoCsBack/testliir/nathan/p1_causality/DeVLBert/features_lmdb/CC", "")

FGS(sys.argv)
