import ray
import time
from argparse import Namespace
import numpy as np
from tensorpack import LMDBSerializer

from raw__img_text__to__lmdb_region_feats_text import CoCaDataFlow, setup

IMAGE_DIR = '/cw/liir/NoCsBack/testliir/datasets/ConceptualCaptions/training'
args = Namespace(bbox_dir='bbox',
                 config_file='buatest/configs/bua-caffe/extract-bua-caffe-r101.yaml',
                 extract_mode='roi_feats',
                 gpu_id="'0,1,2,3'",
                 image_dir=IMAGE_DIR,
                 min_max_boxes='10,100',
                 mode='caffe',
                 mydebug=True,
                 num_cpus=32,
                 num_samples=100,
                 opts=[],
                 output_dir='features',
                 resume=False,
                 visual_features_lmdb="/cw/liir/NoCsBack/testliir/nathan/p1_causality/DeVLBert/features_lmdb/CC/training_feat_all_debug_1613474882.lmdb"
                 )


def main():  # TODO make this (a lot) faster
    ds = LMDBSerializer.load(args.visual_features_lmdb)
    ds.reset_state()
    print("Starting co-occurrence counting")
    # cooc_matrix = get_cooc_matrix.remote(ds)
    # for i in range(args.num_cpus):
    start = time.time()
    cooc_matrix = get_cooc_matrix(ds)
    print(time.time() - start)

    # ray.get(cooc_matrix_list)
    print("Saving co-occurrence counts")
    np.save('cooc_matrix.npy', cooc_matrix)


def get_cooc_matrix(ds):
    NB_CLASSES = 1601
    cooc_matrix = np.zeros((NB_CLASSES, NB_CLASSES))
    for e, (_, cls_probs, _, _, _, _, _, _) in enumerate(ds.get_data()):
        cls_idxs = np.argmax(cls_probs, 1)
        seen = []
        for n, i in enumerate(cls_idxs):
            if i in seen:  # We don't care about multiple co-occurrences: we only care about either [zero] or [one or more] co-occurrences
                continue
            seen.append(i)
            for j in cls_idxs[n+1:]:
                if j in seen:  # This also means we don't count self-co-occurrences, as they are always true anyway
                    continue
                if i > j:
                    cooc_matrix[j, i] += 1
                else:
                    cooc_matrix[i, j] += 1
    return cooc_matrix


if __name__ == '__main__':
    main()
