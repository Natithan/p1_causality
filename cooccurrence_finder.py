from pathlib import Path

import sys

import os
import copy
import pickle
import ray
# import spacy
import time
from absl import flags
from argparse import Namespace
import numpy as np
from pytorch_pretrained_bert import BertTokenizer
import json
IMG_AND_TEXT_TOKENS_LENGTH = 36
ONLY_COUNTING = True # Disable if want to find coocs with image id examples for Turkers
# Copied from codad ds 2: not in bua project
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        input_ids=None,
        input_mask=None,
        segment_ids=None,
        is_next=None,
        lm_label_ids=None,
        image_feat=None,
        image_target=None,
        image_loc=None,
        image_label=None,
        image_mask=None,
    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_next = is_next
        self.lm_label_ids = lm_label_ids
        self.image_feat = image_feat
        self.image_loc = image_loc
        self.image_label = image_label
        self.image_target = image_target
        self.image_mask = image_mask

class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(
        self, image_feat=None, image_target=None, caption=None, is_next=None, lm_labels=None, image_loc=None, num_boxes=None
    ):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            tokens_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            tokens_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.image_feat = image_feat
        self.caption = caption
        self.is_next = is_next  # nextSentence
        self.lm_labels = lm_labels  # masked words for language model
        self.image_loc = image_loc
        self.image_target = image_target
        self.num_boxes = num_boxes

class BertPreprocessBatch(object):
    def __init__(self, caption_path, tokenizer, seq_len, region_len, data_size, split="Train", encoding="utf-8",
                 predict_feature=False, visualization=False):

        self.split = split
        self.seq_len = seq_len
        self.region_len = region_len
        self.tokenizer = tokenizer
        self.predict_feature = predict_feature
        # self.num_caps = data_size
        self.captions = list(json.load(open(caption_path, 'r')).values())
        self.num_caps = len(self.captions)
        self.visualization = visualization

    def __call__(self, data):

        image_feature_wp, image_target_wp, image_location_wp, num_boxes, image_h, image_w, image_id, caption = data

        image_feature = np.zeros((self.region_len, 2048), dtype=np.float32)
        image_target = np.zeros((self.region_len, 1601), dtype=np.float32)
        image_location = np.zeros((self.region_len, 5), dtype=np.float32)

        # Nathan: I created RPN features with MIN_BOXES=10/MAX_BOXES=100, they did MIN=MAX=36.
        # For now, cropping regions to [:36] if longer, and later making sure to ignore indices > num_boxes if shorter
        # num_boxes = int(num_boxes)
        num_boxes = min(int(num_boxes), IMG_AND_TEXT_TOKENS_LENGTH)
        image_feature[:num_boxes] = image_feature_wp[:num_boxes]
        image_target[:num_boxes] = image_target_wp[:num_boxes]
        image_location[:num_boxes, :4] = image_location_wp[:num_boxes]

        image_location[:, 4] = (image_location[:, 3] - image_location[:, 1]) * (
                image_location[:, 2] - image_location[:, 0]) / (float(image_w) * float(image_h))

        image_location[:, 0] = image_location[:, 0] / float(image_w)
        image_location[:, 1] = image_location[:, 1] / float(image_h)
        image_location[:, 2] = image_location[:, 2] / float(image_w)
        image_location[:, 3] = image_location[:, 3] / float(image_h)

        image_feature_og = image_feature
        if self.predict_feature:
            image_feature = copy.deepcopy(image_feature)
            image_target = copy.deepcopy(image_feature)
        else:
            image_feature = copy.deepcopy(image_feature)
            image_target = copy.deepcopy(image_target)

        caption, label = self.random_cap(caption)

        tokens_caption = self.tokenizer.tokenize(caption)
        cur_example = InputExample(
            image_feat=image_feature,
            image_target=image_target,
            caption=tokens_caption,
            is_next=label,
            image_loc=image_location,
            num_boxes=num_boxes
        )

        # transform sample to features
        cur_features = self.convert_example_to_features(cur_example, self.seq_len, self.tokenizer, self.region_len)

        cur_tensors = (
            cur_features.input_ids,
            cur_features.input_mask,
            cur_features.segment_ids,
            cur_features.lm_label_ids,
            cur_features.is_next,
            cur_features.image_feat,
            cur_features.image_loc,
            cur_features.image_target,
            cur_features.image_label,
            cur_features.image_mask,
            int(image_id),
            image_feature_og,
            int(num_boxes)
        )
        return cur_tensors

    def random_cap(self, caption):
        """
        Get one sample from corpus consisting of two sentences. With prob. 50% these are two subsequent sentences
        from one doc. With 50% the second sentence will be a random one from another doc.
        :param index: int, index of sample.
        :return: (str, str, int), sentence 1, sentence 2, isNextSentence Label
        """

        if self.visualization:
            return caption, 0

        # if random.random() > 0.5:
        #     label = 0
        # else:
        #     caption = self.get_random_caption()
        #     label = 1

        return caption, 0

    def get_random_caption(self):
        """
        Get random caption from another document for nextSentence task.
        :return: str, content of one line
        """
        # Similar to original tf repo: This outer loop should rarely go for more than one iteration for large
        # corpora. However, just to be careful, we try to make sure that
        # the random document is not the same as the document we're processing.

        # add the hard negative mining objective here.
        rand_doc_idx = random.randint(0, self.num_caps - 1)
        caption = self.captions[rand_doc_idx]

        return caption

    def convert_example_to_features(self, example, max_seq_length, tokenizer, max_region_length):
        """
        Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
        IDs, LM labels, input_mask, CLS and SEP tokens etc.
        :param example: InputExample, containing sentence input as strings and is_next label
        :param max_seq_length: int, maximum length of sequence.
        :param tokenizer: Tokenizer
        :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
        """
        image_feat = example.image_feat
        caption = example.caption
        image_loc = example.image_loc
        image_target = example.image_target
        num_boxes = int(example.num_boxes)
        self._truncate_seq_pair(caption, max_seq_length - 2)

        caption, caption_label = self.random_word(caption, tokenizer)

        image_feat, image_loc, image_label = self.random_region(image_feat, image_loc, num_boxes)

        # concatenate lm labels and account for CLS, SEP, SEP
        # lm_label_ids = ([-1] + caption_label + [-1] + image_label + [-1])
        lm_label_ids = [-1] + caption_label + [-1]
        # image_label = ([-1] + image_label)

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []

        tokens.append("[CLS]")
        segment_ids.append(0)
        # for i in range(36):
        #     # tokens.append(0)
        #     segment_ids.append(0)

        # tokens.append("[SEP]")
        # segment_ids.append(0)
        for token in caption:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        # input_ids = input_ids[:1] input_ids[1:]
        input_mask = [1] * (len(input_ids))
        image_mask = [1] * (num_boxes)
        # Zero-pad up to the visual sequence length.
        while len(image_mask) < max_region_length:
            image_mask.append(0)
            image_label.append(-1)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            lm_label_ids.append(-1)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(lm_label_ids) == max_seq_length
        assert len(image_mask) == max_region_length
        assert len(image_label) == max_region_length

        # if example.guid < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #             [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #             "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("LM label: %s " % (lm_label_ids))
        #     logger.info("Is next sentence label: %s " % (example.is_next))

        features = InputFeatures(
            input_ids=np.array(input_ids),
            input_mask=np.array(input_mask),
            segment_ids=np.array(segment_ids),
            lm_label_ids=np.array(lm_label_ids),
            is_next=np.array(example.is_next),
            image_feat=image_feat,
            image_target=image_target,
            image_loc=image_loc,
            image_label=np.array(image_label),
            image_mask=np.array(image_mask),
        )
        return features

    def _truncate_seq_pair(self, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_b)
            if total_length <= max_length:
                break

            tokens_b.pop()

    def random_word(self, tokens, tokenizer):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of str, tokenized sentence.
        :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
        :return: (list of str, list of int), masked tokens and related labels for LM prediction
        """
        output_label = []

        for i, token in enumerate(tokens):
            output_label.append(-1)
            # prob = random.random()
            # # mask token with 15% probability
            #
            # if prob < 0.15 and not self.visualization:
            #     prob /= 0.15
            #
            #     # 80% randomly change token to mask token
            #     if prob < 0.8:
            #         tokens[i] = "[MASK]"
            #
            #     # 10% randomly change token to random token
            #     elif prob < 0.9:
            #         tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]
            #
            #     # -> rest 10% randomly keep current token
            #
            #     # append current token to output (we will predict these later)
            #     try:
            #         output_label.append(tokenizer.vocab[token])
            #     except KeyError:
            #         # For unknown words (should not occur with BPE vocab)
            #         output_label.append(tokenizer.vocab["[UNK]"])
            #         logger.warning(
            #             "Cannot find token '{}' in vocab. Using [UNK] insetad".format(token)
            #         )
            # else:
            #     # no masking token (will be ignored by loss function later)
            #     output_label.append(-1)

        return tokens, output_label

    def random_region(self, image_feat, image_loc, num_boxes):
        """
        """
        output_label = []

        for i in range(num_boxes):
            output_label.append(-1)
            # prob = random.random()
            # # mask token with 15% probability
            # if prob < 0.15 and not self.visualization:
            #     prob /= 0.15
            #
            #     # 80% randomly change token to mask token
            #     if prob < 0.9:
            #         image_feat[i] = 0
            #         output_label.append(1)
            #     else:
            #         output_label.append(-1)
            #     # 10% randomly change token to random token
            #     # elif prob < 0.9:
            #     # tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]
            #
            #     # -> rest 10% randomly keep current token
            #     # append current token to output (we will predict these later)
            #
            # else:
            #     # no masking token (will be ignored by loss function later)
            #     output_label.append(-1)

        return image_feat, image_loc, output_label


from my_lmdb import MyLMDBSerializer
from tensorpack import LMDBSerializer
from tqdm import tqdm
import tensorpack.dataflow as td

from constants import OLD_10_100_LMDB_PATH, PROJECT_ROOT_DIR, MTURK_DIR, LMDB_PATHS, ID2CLASS_PATH, CAPTION_PATH

# from preprocess import CoCaDataFlow, setup
IMG_ONLY = True
# COUNT_FILE = 'cooc_counts_img'
COUNT_FILE = 'cooc_counts_v_and_t_and_combo'
CONDITIONAL_PROBS_FILE = 'cond_probs_v_and_t_and_combo'
# region Flags stuff
FGS = flags.FLAGS
flags.DEFINE_bool("sandbox", True, "")
flags.DEFINE_integer("max_count", -1, "")
FGS(sys.argv)
NB_IMG_CLASSES = 1601

vocid2coocid = np.load(ID2CLASS_PATH, allow_pickle=True).item()
NB_TXT_CLASSES = len(vocid2coocid)

tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased', do_lower_case=True
)
special_ids = {v for k, v in tokenizer.vocab.items() if k[0] == "#"}
RECOUNT_ANYWAY = True

# endregion


def main():
    # ds = LMDBSerializer.load(OLD_10_100_LMDB_PATH, shuffle=False)
    # ds.reset_state()
    preprocess_function = BertPreprocessBatch(
        caption_path=CAPTION_PATH,
        tokenizer=tokenizer,
        seq_len=IMG_AND_TEXT_TOKENS_LENGTH,
        region_len=IMG_AND_TEXT_TOKENS_LENGTH,
        data_size=None,
        encoding="utf-8",
        predict_feature=False
    )

    ds = td.MapData(MyLMDBSerializer.load(LMDB_PATHS, shuffle=False), preprocess_function)
    ds.reset_state()
    if os.path.exists(COUNT_FILE) and not RECOUNT_ANYWAY:
        with open(COUNT_FILE, 'rb') as f:
            if IMG_ONLY:
                with open(COUNT_FILE, 'rb') as f:
                    img_joint_count, total_count, img_id_for_img_idx_pair = pickle.load(
                        f)
                    text_joint_count, id_for_c_noun, img_id_for_c_idx_pair = None, None, None
            else:
                img_joint_count, \
                text_joint_count, \
                id_for_c_noun, \
                total_count, \
                img_id_for_img_idx_pair, \
                img_id_for_c_idx_pair = pickle.load(
                    f)
    else:
        # raise Exception("Halt! Count file creation needs to be updated: creates unnecessarily ids_for_pair dicts atm")
        img_joint_count, \
        image_text_joint_count, \
        text_joint_count, \
        total_count = read_and_count(ds)
        with open(COUNT_FILE, 'wb') as f:
            pickle.dump((img_joint_count, text_joint_count,image_text_joint_count, total_count), f)
        with open(CONDITIONAL_PROBS_FILE,'wb') as f:
            conditionals = {}
            for LT_joint_count,mode in zip((img_joint_count,text_joint_count),('image','text')):
                marginal_count = LT_joint_count.diagonal()
                joint_count = LT_joint_count.T + LT_joint_count
                np.fill_diagonal(joint_count, np.diag(LT_joint_count))
                conditionals[f'{mode}'] = joint_count / marginal_count
            for marginal_count,joint_count, mode in zip(
                    (text_joint_count.diagonal(),   img_joint_count.diagonal()),
                    (image_text_joint_count,        image_text_joint_count.transpose()),
                    ('image_given_text',            'text_given_image')):
                conditionals[f'{mode}'] = joint_count / marginal_count
            pickle.dump(conditionals, f)
    if not ONLY_COUNTING:
        with open("DeVLBert/dic/objects_vocab.txt", "r") as vocab:
            object_list = ['_background_'] + [line.strip() for line in vocab]
        id_for_img_noun = {w: i for i, w in enumerate(object_list)}
        for mod, joint_count, id_for_noun, img_id_for_idx_pair in zip(("image", "caption"),
                                                                      (img_joint_count, text_joint_count),
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
          'joint_image_ids': img_id_for_idx_pair[(el[0], el[1])] if el[2] > 0 else [],
          # if negative correlation, there might not be any joint image, so not inserting those
          'marginal_img_ids': (img_id_for_idx_pair[(el[0], el[0])], img_id_for_idx_pair[(el[1], el[1])])
          }
         for el in s]
    return w


def read_and_count(ds):
    img_joint_count = np.zeros((NB_IMG_CLASSES, NB_IMG_CLASSES))
    # nlp = spacy.load('en_core_web_sm')
    id_for_c_noun = {}
    text_joint_count = np.zeros((NB_TXT_CLASSES, NB_TXT_CLASSES))
    img_id_for_img_idx_pair = {}
    img_id_for_c_idx_pair = {}
    img_text_joint_count = np.zeros((NB_IMG_CLASSES, NB_TXT_CLASSES))
    for total_count, (input_ids,_, _, _, _, _, _, img_cls_probs,_,_,image_id,_,_) in tqdm(enumerate(ds.get_data()),
                                                                         total=len(ds) if not (
                                                                                 FGS.max_count > 0) else FGS.max_count):
        if (FGS.max_count > 0) and (total_count > FGS.max_count):
            print(f'\r\nOnly doing first {FGS.max_count} for debugging')
            break
        # Image object co-occurrences
        img_cls_idxs = np.argmax(img_cls_probs, 1)
        img_cls_idxs = list(set(
            img_cls_idxs))  # remove duplicates: We don't care about multiple co-occurrences: we only care about either [zero] or [one or more] co-occurrences
        for n, i in enumerate(img_cls_idxs):
            for j in img_cls_idxs[n:]:  # Also count self-cooccurrence to get marginal info on the diagonal
                img_joint_count[max(i, j), min(i, j)] += 1  # Only fill half triangle of matrix
                # fill_id_for_idx_pair(max(i, j), min(i, j), img_id, img_id_for_img_idx_pair)
        l = len(input_ids)
        for pos1, id1 in enumerate(input_ids):
            cl1 = vocid2coocid.get(id1) # .get to return 'None' if not found
            if cls_can_count(cl1, pos1, input_ids, l):

                for extra_pos,id2 in enumerate(input_ids[pos1:]):
                    pos2 = pos1 + extra_pos# Also count self-cooccurrence to get marginal info on the diagonal
                    cl2 = vocid2coocid.get(id2)
                    if cls_can_count(cl2, pos2, input_ids, l):
                        text_joint_count[max(cl1, cl2), min(cl1, cl2)] += 1 # Only fill half triangle of matrix Only fill half triangle of matrix

                for img_cls in img_cls_idxs:
                    img_text_joint_count[img_cls, cl1] += 1
    #     # Caption noun co-occurrences
    #     doc = nlp(caption)
    #     cn = [str(t) for t in doc if t.pos_ == "NOUN"]
    #     cn = list(set(cn))  # remove duplicates
    #
    #     # extend the data structures if new words encountered
    #     for t1 in cn:
    #         if t1 not in id_for_c_noun:
    #             id_for_c_noun[t1] = len(id_for_c_noun)
    #
    #             # Only resize np array infrequently, as this takes time
    #             if len(id_for_c_noun) > len(c_joint_count):
    #                 c_joint_count = np.pad(c_joint_count, ((0, len(c_joint_count)), (0, len(c_joint_count))))
    #
    #     for n, t1 in enumerate(cn):
    #         for t2 in cn[n:]:  # Also count self-co-occurrence to get marginal info on the diagonal
    #             r, c = id_for_c_noun[t1], id_for_c_noun[t2]
    #             c_joint_count[max(r, c), min(r, c)] += 1
    #             fill_id_for_idx_pair(max(r, c), min(r, c), img_id, img_id_for_c_idx_pair)
    # # Trim the caption joint count matrix
    # c_joint_count = c_joint_count[:len(id_for_c_noun), :len(id_for_c_noun)]
    # return joint_count, c_joint_count, id_for_c_noun, total_count, img_id_for_img_idx_pair, img_id_for_c_idx_pair
    return img_joint_count, img_text_joint_count, text_joint_count, total_count


def cls_can_count(count_matrix_idx, pos, vocab_id_array, vocab_id_array_len):
    return count_matrix_idx is not None and (
                (pos != vocab_id_array_len - 1 and int(vocab_id_array[pos + 1])) not in special_ids or pos == vocab_id_array_len - 1)


# TODO test this :P
def fill_id_for_idx_pair(i, j, img_id, img_id_for_img_idx_pair):
    if (max(i, j), min(i, j)) in img_id_for_img_idx_pair:
        img_id_for_img_idx_pair[(max(i, j), min(i, j))].append(img_id)
    else:
        img_id_for_img_idx_pair[(max(i, j), min(i, j))] = [img_id]


if __name__ == '__main__':
    main()
