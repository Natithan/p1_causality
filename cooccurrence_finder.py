import os

import pickle
import ray
import time
from argparse import Namespace
import numpy as np
from tensorpack import LMDBSerializer
from tqdm import tqdm

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
                 visual_features_lmdb="/cw/liir/NoCsBack/testliir/nathan/p1_causality/DeVLBert/features_lmdb/CC/full_coca.lmdb"
                 )


def main():  # TODO make this (a lot) faster
    ds = LMDBSerializer.load(args.visual_features_lmdb, shuffle=False)
    ds.reset_state()
    NB_CLASSES = 1601
    joint_count = np.zeros((NB_CLASSES, NB_CLASSES))
    count_file = 'cooc_counts_tmp_2'         # TODO run for big data
    if os.path.exists(count_file):
        total_count,joint_count = pickle.load(open(count_file,'rb'))
    else:
        for total_count, (_, cls_probs, _, _, _, _, _, _) in tqdm(enumerate(ds.get_data()),total=len(ds)):
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
                    seen_j.append(j)
                    joint_count[i, j] += 1
        pickle.dump((total_count,joint_count),open(count_file,'wb'))
    marginal_count = joint_count.diagonal()
    marginal = marginal_count / total_count

    joint = joint_count / total_count
    independent_joint = np.outer(marginal, marginal)
    excess_joint = abs(joint - independent_joint) #TODO check for relative excess with log(quotient)

    with  open("DeVLBert/dic/objects_vocab.txt", "r") as vocab:
        object_list = ['background'] + [line.strip() for line in vocab]
    THRESHOLDS = [0.005,0.01,0.05,0.1]
    for THRESHOLD in THRESHOLDS:
        significant_idxs= np.where(excess_joint > THRESHOLD)
        probs = excess_joint[significant_idxs]
        significant_pairs = [p for p in zip(*significant_idxs,probs) if (p[0] != p[1])] # Self-pairs are not interesting of course
        word_pairs = [(object_list[i], object_list[j],prob) for (i, j,prob) in significant_pairs]
        with open(f"word_pairs_{THRESHOLD}.txt", "w") as wp_file:
            for p in sorted(word_pairs,key=lambda d: d[-1],reverse=True):
                wp_file.write(str(p) + "\r\n")
        with open(f"word_pairs_no_background_{THRESHOLD}.txt", "w") as wp_file:
            significant_pairs_no_background = [p for p in significant_pairs if not 0 in p[:2]]
            word_pairs_no_background = [(object_list[i], object_list[j], prob) for (i, j, prob) in significant_pairs_no_background]
            for p in sorted(word_pairs_no_background,key=lambda d: d[-1],reverse=True):
                wp_file.write(str(p) + "\r\n")



if __name__ == '__main__':
    main()
