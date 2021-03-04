import pickle
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
                 visual_features_lmdb="/cw/liir/NoCsBack/testliir/nathan/p1_causality/DeVLBert/features_lmdb/CC/training_feat_all_debug_1614879706.lmdb"
                 )


def main():  # TODO make this (a lot) faster
    ds = LMDBSerializer.load(args.visual_features_lmdb)
    ds.reset_state()
    print("Starting co-occurrence counting")
    # cooc_matrix = get_cooc_matrix.remote(ds)
    # for i in range(args.num_cpus):
    start = time.time()
    total_count, marginal_count, joint_count = get_class_counts(ds)
    print(time.time() - start)

    # ray.get(cooc_matrix_list)




def get_class_counts(ds):
    NB_CLASSES = 1601
    joint_count = np.zeros((NB_CLASSES, NB_CLASSES))
    for total_count, (_, cls_probs, _, _, _, _, _, _) in enumerate(ds.get_data()):
        cls_idxs = np.argmax(cls_probs, 1)
        seen_i = []
        for n, i in enumerate(cls_idxs):
            if i in seen_i:  # We don't care about multiple co-occurrences: we only care about either [zero] or [one or more] co-occurrences
                continue
            seen_i.append(i)
            seen_j = []
            for j in cls_idxs[n+1:]:
                if j in seen_j:
                    continue
                joint_count[i, j] += 1

    marginal_count = joint_count.diagonal()
    marginal = marginal_count / total_count
    # conditional = joint_count / marginal_count # conditional[i,j] is p(i|j)
    # excess_conditional = conditional - np.expand_dims(marginal,axis=1) # excess_conditional[i,j] is p(i|j) - p(i)
    joint = joint_count / total_count
    independent_joint = np.outer(marginal, marginal)
    excess_joint = abs(joint - independent_joint)
    THRESHOLD = 0.1
    significant_pairs = np.where(excess_joint > THRESHOLD)
    significant_pairs = [p for p in significant_pairs if (p[0] != p[1])] # Self-pairs are not interesting of course
    # TODO run for big data, and test for small data: link classes to idxs of significant pairs
    pickle.dump((total_count, joint_count), open("class_counts.p", "wb"))
    return total_count, joint_count


if __name__ == '__main__':
    main()
