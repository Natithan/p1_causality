import os

import pickle
import ray
import spacy
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
                 visual_features_lmdb="/cw/working-arwen/nathan/features_CoCa_lmdb/full_coca_1.lmdb"
                 )


def main():  # TODO make this (a lot) faster
    ds = LMDBSerializer.load(args.visual_features_lmdb, shuffle=False)
    ds.reset_state()
    NB_CLASSES = 1601
    count_file = 'cooc_counts_tmp_2'  # TODO run for big data
    if os.path.exists(count_file):
        total_count, joint_count = pickle.load(open(count_file, 'rb'))
    else:
        joint_count, total_count = read_and_count(NB_CLASSES, ds)
        pickle.dump((total_count, joint_count), open(count_file, 'wb'))
    marginal_count = joint_count.diagonal()
    marginal = marginal_count / total_count

    joint = joint_count / total_count
    independent_joint = np.outer(marginal, marginal)
    excess_joint = abs(joint - independent_joint)
    rel_excess_joint = np.absolute(np.log(joint / independent_joint))

    with  open("DeVLBert/dic/objects_vocab.txt", "r") as vocab:
        object_list = ['background'] + [line.strip() for line in vocab]

    for X_joint, X in ((excess_joint, "absolute"), (rel_excess_joint, "relative")):

        nan_filtered_idxs = np.where(~np.isnan(X_joint)) # Only relevant for rel_excess_joint in fact
        filtered_idxs_zipped = [p for p in zip(*nan_filtered_idxs) if (0 not in p)]  # Not interested in pairs with <background>
        filtered_idxs = (np.array([i for i, _ in filtered_idxs_zipped]), np.array([j for _, j in filtered_idxs_zipped]))  #unzip to be able to use as index
        idxs_and_scores = filtered_idxs + (X_joint[filtered_idxs],)
        CUTOFF = 1000
        s = sorted([t for t in zip(*idxs_and_scores) if t[0] != t[1]], key=lambda t: t[2], reverse=True)[:CUTOFF] #not interested in self-pairs
        w = result_tuple(joint_count, marginal_count, object_list, s)

        write(X, w)

        if X == "relative":
            s2 = sorted([t for t in zip(*idxs_and_scores) if (t[-1] != float('inf')) and (t[0] != t[1])],
                        key=lambda t: t[2], reverse=True)[:CUTOFF] # 'inf' only relevant for rel_excess_joint: pairs
            # that never occur together are very many
            w2 = result_tuple(joint_count, marginal_count, object_list, s2)
            write(X+"_positive", w2)



def write(name, tuple):
    with open(f"word_pairs_{name}.txt", "w") as wp_file:
        wp_file.truncate(0)
        for p in tuple:
            wp_file.write(str(p) + "\r\n")
    with open(f"word_pairs_{name}.tsv", "w") as wp_file:
        wp_file.truncate(0)
        for p in tuple:
            wp_file.write(f'{p[0]}\t{p[1]}\t{p[2]}\t{p[3][0]}\t{p[3][1]}\t{p[4]}' + "\r\n")


def result_tuple(joint_count, marginal_count, object_list, s):
    w = [(object_list[i[0]],
          object_list[i[1]],
          i[2],
          (marginal_count[i[0]], marginal_count[i[1]]),
          joint_count[i[0],i[1]]
          )
         for i in s]
    return w


def read_and_count(NB_CLASSES, ds):
    joint_count = np.zeros((NB_CLASSES, NB_CLASSES))
    nlp = spacy.load('en_core_web_sm')
    for total_count, (_, cls_probs, _, _, _, _, _, caption) in tqdm(enumerate(ds.get_data()), total=len(ds)):
        if total_count > 5000:
            print('Only doing first 5000 for debugging')
            break
        cls_idxs = np.argmax(cls_probs, 1)
        seen_i = []
        for n, i in enumerate(cls_idxs):
            if i in seen_i:  # We don't care about multiple co-occurrences: we only care about either [zero] or [one or more] co-occurrences
                continue
            seen_i.append(i)
            seen_j = []
            for j in cls_idxs[n:]:  # Also count self-cooccurrence to get marginal info on the diagonal
                if j in seen_j:
                    continue
                seen_j.append(j)
                joint_count[i, j] += 1

        # TODO finish this
        doc = nlp(caption)
        cn = [str(t) for t in doc if t.pos_ == "NOUN"]
        # extend the data structures if new words encountered
        for t1 in cn:
            if t1 not in id_for_c_noun:
                id_for_c_noun[t1] = len(id_for_c_noun)
                c_joint_count = np.resize(c_joint_count, [i + 1 for i in c_joint_count.shape])
            seen_t1 = []

        seen_t1 = []
        for n, t1 in enumerate(cn):
            if t1 in seen_t1:
                continue
            seen_t1.append(t1)
            seen_t2 = []
            for t2 in cn[n:]:  # Also count self-cooccurrence to get marginal info on the diagonal
                if t2 in seen_t2:
                    continue
                seen_t2.append(t2)
                c_joint_count[id_for_c_noun[t1], id_for_c_noun[t2]] += 1
    return joint_count, total_count


if __name__ == '__main__':
    main()
