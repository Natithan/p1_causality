from pathlib import Path

import sys

import os

import pickle
import ray
import spacy
import time
from absl import flags
from argparse import Namespace
import numpy as np
from tensorpack import LMDBSerializer
from tqdm import tqdm

from constants import LMDB_PATH, ROOT_DIR, MTURK_DIR
from preprocess import CoCaDataFlow, setup
IMG_ONLY = True
COUNT_FILE = 'cooc_counts_img'
# region Flags stuff
FGS = flags.FLAGS
flags.DEFINE_bool("sandbox", True, "")
flags.DEFINE_integer("max_count", -1, "")
FGS(sys.argv)


# endregion


def main():  # TODO make this (a lot) faster
    ds = LMDBSerializer.load(LMDB_PATH, shuffle=False)
    ds.reset_state()
    NB_CLASSES = 1601
    if os.path.exists(COUNT_FILE):
        with open(COUNT_FILE, 'rb') as f:
            if IMG_ONLY:
                with open(COUNT_FILE, 'rb') as f:
                    img_joint_count, total_count, img_id_for_img_idx_pair = pickle.load(
                        f)
                    c_joint_count, id_for_c_noun, img_id_for_c_idx_pair = None, None, None
            else:
                img_joint_count, c_joint_count, id_for_c_noun, total_count, img_id_for_img_idx_pair, img_id_for_c_idx_pair = pickle.load(
                f)
    else:
        raise Exception("Halt! Count file creation needs to be updated: creates unnecessarily ids_for_pair dicts atm")
        img_joint_count, c_joint_count, id_for_c_noun, total_count, img_id_for_img_idx_pair, img_id_for_c_idx_pair = read_and_count(
            NB_CLASSES, ds)
        with open(COUNT_FILE, 'wb') as f:
            pickle.dump((img_joint_count, c_joint_count, id_for_c_noun, total_count, img_id_for_img_idx_pair,
                         img_id_for_c_idx_pair), f)

    with open("DeVLBert/dic/objects_vocab.txt", "r") as vocab:
        object_list = ['_background_'] + [line.strip() for line in vocab]
    id_for_img_noun = {w: i for i, w in enumerate(object_list)}

    for mod, joint_count, id_for_noun, img_id_for_idx_pair in zip(("image", "caption"),
                                                                  (img_joint_count, c_joint_count),
                                                                  (id_for_img_noun, id_for_c_noun),
                                                                  (img_id_for_img_idx_pair, img_id_for_c_idx_pair)):
        if mod == "caption":
            print("Skipping caption nouns creating, as count file creation for that still needs to be updated")
            break
        noun_for_id = {i: n for (n, i) in id_for_noun.items()}

        marginal_count = joint_count.diagonal()
        marginal = marginal_count / total_count

        joint = joint_count / total_count
        independent_joint = np.tril(np.outer(marginal, marginal))
        excess_joint = joint - independent_joint
        rel_excess_joint = np.log(joint / independent_joint)

        for X_joint, X in ((excess_joint, f"absolute_{mod}"), (rel_excess_joint, f"relative_{mod}")):

            nan_filtered_idxs = np.where(~np.isnan(X_joint))  # Only relevant for rel_excess_joint in fact
            filtered_idxs_zipped = [p for p in zip(*nan_filtered_idxs) if
                                    (0 not in p)]  # Not interested in pairs with <background>
            filtered_idxs = (np.array([i for i, _ in filtered_idxs_zipped]),
                             np.array([j for _, j in filtered_idxs_zipped]))  # unzip to be able to use as index
            idxs_and_scores = filtered_idxs + (X_joint[filtered_idxs],)
            CUTOFF = 1000
            s = sorted([t for t in zip(*idxs_and_scores) if t[0] != t[1]], key=lambda t: abs(t[2]), reverse=True)[
                :CUTOFF]  # not interested in self-pairs

            w = result_dict(joint_count, marginal_count, noun_for_id, s, img_id_for_idx_pair)

            # Filter pairs like ('toy','toys) or ('stovetop', 'stove top')
            w = [d for d in w if not (
                    (d['words'][0] == d['words'][1] + 's') or (d['words'][1] == d['words'][0] + 's') or
                    d['words'][0].replace(" ", "") == d['words'][1].replace(" ", "")
            )]

            write_dict_to_tsv(X, w)

            if "relative" in X:
                s2 = sorted([t for t in zip(*idxs_and_scores) if (t[2] != float('-inf')) and (t[0] != t[1])],
                            key=lambda t: abs(t[2]), reverse=True)[
                     :CUTOFF]  # 'inf' only relevant for rel_excess_joint: pairs
                # that never occur together are very many
                # Filter pairs like ('toy','toys) or ('stovetop', 'stove top')

                w2 = result_dict(joint_count, marginal_count, noun_for_id, s2, img_id_for_idx_pair)
                w2 = [d for d in w2 if not (
                        (d['words'][0] == d['words'][1] + 's') or (d['words'][1] == d['words'][0] + 's') or
                        d['words'][0].replace(" ", "") == d['words'][1].replace(" ", "")
                )]
                write_dict_to_tsv(X + "_positive", w2)


def write_dict_to_tsv(name, dic):
    output_dir = Path(MTURK_DIR, 'input_mturk')
    if not Path.exists(output_dir):
        Path.mkdir(output_dir)
    with open(Path(output_dir, f"{name}.tsv"), "w") as wp_file:
        wp_file.truncate(0)
        wp_file.write('\t'.join(dic[0].keys()) + "\r\n")
        for p in dic:
            wp_file.write('\t'.join(str(v) for v in p.values()) + "\r\n")



def result_dict(joint_count, marginal_count, noun_for_id, s, img_id_for_idx_pair):
    w = [{'words': (noun_for_id[el[0]], noun_for_id[el[1]]),
          'excess_joint': el[2],
          'marginal_counts': (marginal_count[el[0]], marginal_count[el[1]]),
          'joint_count': joint_count[el[0], el[1]],
          'joint_image_ids': img_id_for_idx_pair[(el[0], el[1])] if el[2] > 0 else [], # if negative correlation, there might not be any joint image, so not inserting those
          'marginal_img_ids': (img_id_for_idx_pair[(el[0], el[0])], img_id_for_idx_pair[(el[1], el[1])])
          }
         for el in s]
    return w


# TODO make sure this runs in reasonable time for big data
def read_and_count(NB_CLASSES, ds):
    joint_count = np.zeros((NB_CLASSES, NB_CLASSES))
    nlp = spacy.load('en_core_web_sm')
    id_for_c_noun = {}
    c_joint_count = np.zeros((500, 500))
    img_id_for_img_idx_pair = {}
    img_id_for_c_idx_pair = {}
    for total_count, (_, cls_probs, _, _, _, _, img_id, caption) in tqdm(enumerate(ds.get_data()),
                                                                         total=len(ds) if not (
                                                                                 FGS.max_count > 0) else FGS.max_count):
        if (FGS.max_count > 0) and (total_count > FGS.max_count):
            print(f'\r\nOnly doing first {FGS.max_count} for debugging')
            break
        # Image object co-occurrences
        cls_idxs = np.argmax(cls_probs, 1)
        cls_idxs = list(set(
            cls_idxs))  # remove duplicates: We don't care about multiple co-occurrences: we only care about either [zero] or [one or more] co-occurrences
        for n, i in enumerate(cls_idxs):
            for j in cls_idxs[n:]:  # Also count self-cooccurrence to get marginal info on the diagonal
                joint_count[max(i, j), min(i, j)] += 1  # Only fill half triangle of matrix
                fill_id_for_idx_pair(max(i, j), min(i, j), img_id, img_id_for_img_idx_pair)

        # Caption noun co-occurrences
        doc = nlp(caption)
        cn = [str(t) for t in doc if t.pos_ == "NOUN"]
        cn = list(set(cn))  # remove duplicates

        # extend the data structures if new words encountered
        for t1 in cn:
            if t1 not in id_for_c_noun:
                id_for_c_noun[t1] = len(id_for_c_noun)

                # Only resize np array infrequently, as this takes time
                if len(id_for_c_noun) > len(c_joint_count):
                    c_joint_count = np.pad(c_joint_count, ((0, len(c_joint_count)), (0, len(c_joint_count))))

        for n, t1 in enumerate(cn):
            for t2 in cn[n:]:  # Also count self-co-occurrence to get marginal info on the diagonal
                r, c = id_for_c_noun[t1], id_for_c_noun[t2]
                c_joint_count[max(r, c), min(r, c)] += 1
                fill_id_for_idx_pair(max(r, c), min(r, c), img_id, img_id_for_c_idx_pair)
    # Trim the caption joint count matrix
    c_joint_count = c_joint_count[:len(id_for_c_noun), :len(id_for_c_noun)]
    return joint_count, c_joint_count, id_for_c_noun, total_count, img_id_for_img_idx_pair, img_id_for_c_idx_pair


# TODO test this :P
def fill_id_for_idx_pair(i, j, img_id, img_id_for_img_idx_pair):
    if (max(i, j), min(i, j)) in img_id_for_img_idx_pair:
        img_id_for_img_idx_pair[(max(i, j), min(i, j))].append(img_id)
    else:
        img_id_for_img_idx_pair[(max(i, j), min(i, j))] = [img_id]


if __name__ == '__main__':
    main()
