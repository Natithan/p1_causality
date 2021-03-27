import copy
import json
import logging
import os
import random

import lmdb
import numpy as np
import tensorpack.dataflow as td

import torch
from constants import CAPTION_PATH, LMDB_PATHS
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import torch.distributed as dist
import sys
import pdb
from constants import ID2CLASS_PATH
from typing import List

REGION_LEN = 36
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
from my_lmdb import MyLMDBSerializer, MyLMDBData

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(
            self, image_feat=None, image_target=None, caption=None, is_next=None, lm_labels=None, image_loc=None,
            num_boxes=None
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


# class InputFeatures(object):
#     """A single set of features of data."""
#
#     def __init__(
#         self,
#         input_ids=None,
#         input_mask=None,
#         segment_ids=None,
#         is_next=None,
#         lm_label_ids=None,
#         image_feat=None,
#         image_target=None,
#         image_loc=None,
#         image_label=None,
#         image_mask=None,
#         input_ids_og=None,
#         noun_label=None
#     ):
#         self.input_ids = input_ids
#         self.input_mask = input_mask
#         self.segment_ids = segment_ids
#         self.is_next = is_next
#         self.lm_label_ids = lm_label_ids
#         self.image_feat = image_feat
#         self.image_loc = image_loc
#         self.image_label = image_label
#         self.image_target = image_target
#         self.image_mask = image_mask
#         self.input_ids_og = input_ids_og
#         self.noun_label = noun_label

class ConceptCapLoaderTrain(object):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.
    Arguments:
        mode (str, required): mode of dataset to operate in, one of ['train', 'val']
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 0)
        cache (int, optional): cache size to use when loading data,
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
        cuda (bool, optional): set to ``True`` and the PyTorch tensors will get preloaded
            to the GPU for you (necessary because this lets us to uint8 conversion on the 
            GPU, which is faster).
    """

    def __init__(
            self,
            tokenizer,
            seq_len,
            encoding="utf-8",
            lmdb_paths: List = None,
            predict_feature=False,
            hard_negative=False,
            batch_size=512,
            shuffle=False,
            num_workers=25,
            cache=50000,
            drop_last=False,
            cuda=False,
            distributed=True,
            visualization=False,
            savePath=None,
            mini=False
    ):
        if lmdb_paths:
            lmdb_files = lmdb_paths
        else:
            # if dist.is_available() and distributed:
            #     num_replicas = dist.get_world_size()
            #     # assert num_replicas == 8
            #     rank = dist.get_rank()
            #     # if not os.path.exists(lmdb_file):
            #     # lmdb_file = "/srv/share/datasets/conceptual_caption/training_feat_part_" + str(rank) + ".lmdb"
            # else:
            #     # lmdb_file = "/coc/dataset/conceptual_caption/training_feat_all.lmdb"
            #     # if not os.path.exists(lmdb_file):
            #     num_replicas = 1
            #     rank = 0
            #     print(f"WARNING: only loading from {LMDB_PATHS[rank]}")
            #     # lmdb_file = "/mnt3/xuesheng/features_lmdb/CC/training_feat_part_0.lmdb" #Nathan
            lmdb_files = LMDB_PATHS

        caption_path = CAPTION_PATH
        # caption_path = "/mnt3/xuesheng/features_lmdb/CC/caption_train.json"
        print(f"Loading from{lmdb_files}")

        ds = MyLMDBSerializer.load(lmdb_files, shuffle=True, savePath=savePath)
        self.num_dataset = len(ds)

        preprocess_function = BertPreprocessBatch(
            caption_path,
            tokenizer,
            seq_len,
            REGION_LEN,
            self.num_dataset,
            encoding="utf-8",
            predict_feature=predict_feature,
        )

        # ds = td.LocallyShuffleData(ds, cache)
        # ds = td.PrefetchData(ds, 5000, 1)
        ds = td.MapData(ds, preprocess_function)
        # self.ds = td.PrefetchData(ds, 1)
        # ds = td.PrefetchDataZMQ(ds, num_workers) Nathan commenting out in hope of bypassing forking-debugger incompatibility
        self.ds = td.BatchData(ds, batch_size)
        # self.ds = ds
        self.ds.reset_state()

        self.batch_size = batch_size
        self.num_workers = num_workers
        core = self.get_core_ds()
        core.set_mini(mini)

    # Nathan
    def store_checkpoint(self):
        # Digging through all the wrappers
        core_ds = self.get_core_ds()
        core_ds.store_checkpoint()

    def get_core_ds(self) -> MyLMDBData:
        core_ds = self.ds
        while not isinstance(core_ds, MyLMDBData):
            core_ds = core_ds.ds
        return core_ds

    def __iter__(self):

        for batch in self.ds.get_data():
            input_ids, input_mask, segment_ids, lm_label_ids, is_next, image_feat, \
            image_loc, image_target, image_label, image_mask, image_id, causal_label_t, causal_label_v = batch

            batch_size = input_ids.shape[0]
            g_image_feat = np.sum(image_feat, axis=1) / np.sum(image_mask, axis=1,
                                                               keepdims=True)  # Average (g for global?) feat over all bboxes
            # print(np.sum(image_feat, axis=1).shape, np.sum(image_mask, axis=1, keepdims=True).shape) #(64, 2048) (64, 1)
            image_feat = np.concatenate([np.expand_dims(g_image_feat, axis=1), image_feat], axis=1)
            image_feat = np.array(image_feat, dtype=np.float32)
            # print(image_feat.shape) # (64, 37, 2048)

            # g_image_feat_og = np.sum(image_feature_og, axis=1) / np.sum(image_mask, axis=1, keepdims=True)
            # image_feature_og = np.concatenate([np.expand_dims(g_image_feat_og, axis=1), image_feature_og], axis=1)
            # image_feature_og = np.array(image_feature_og, dtype=np.float32)

            g_image_loc = np.repeat(np.array([[0, 0, 1, 1, 1]], dtype=np.float32), batch_size, axis=0)
            image_loc = np.concatenate([np.expand_dims(g_image_loc, axis=1), image_loc], axis=1)

            image_loc = np.array(image_loc, dtype=np.float32)
            g_image_mask = np.repeat(np.array([[1]]), batch_size, axis=0)
            image_mask = np.concatenate([g_image_mask, image_mask], axis=1)
            # print(image_mask.shape) # (64, 37)

            batch = (input_ids, input_mask, segment_ids, lm_label_ids, is_next, image_feat, image_loc, image_target, \
                     image_label, image_mask, image_id, causal_label_t, causal_label_v)

            yield tuple(torch.tensor(data) for data in batch)

    def __len__(self):
        return self.ds.size()

    def reset_index(self):
        core_ds = self.get_core_ds()
        core_ds.reset_index()


class ConceptCapLoaderVal(object):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.
    Arguments:
        mode (str, required): mode of dataset to operate in, one of ['train', 'val']
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 0)
        cache (int, optional): cache size to use when loading data,
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
        cuda (bool, optional): set to ``True`` and the PyTorch tensors will get preloaded
            to the GPU for you (necessary because this lets us to uint8 conversion on the
            GPU, which is faster).
    """

    def __init__(
            self,
            corpus_path,
            tokenizer,
            seq_len,
            encoding="utf-8",
            predict_feature=False,
            batch_size=512,
            shuffle=False,
            num_workers=25,
            cache=50000,
            drop_last=False,
            cuda=False,
            distributed=False,
            visualization=False,
    ):

        lmdb_file = "/mnt3/xuesheng/features_lmdb/CC_val/validation_all.lmdb"
        if not os.path.exists(lmdb_file):
            lmdb_file = "/mnt3/xuesheng/features_lmdb/CC_val/validation_all.lmdb"
        caption_path = "/mnt3/xuesheng/features_lmdb/CC_val/caption_val.json"

        print("Loading from %s" % lmdb_file)

        ds = td.LMDBSerializer.load(lmdb_file, shuffle=False)
        self.num_dataset = len(ds)
        preprocess_function = BertPreprocessBatch(
            caption_path,
            tokenizer,
            seq_len,
            REGION_LEN,
            self.num_dataset,
            encoding="utf-8",
            predict_feature=predict_feature,
            visualization=visualization,
        )

        ds = td.MapData(ds, preprocess_function)
        self.ds = td.BatchData(ds, batch_size)
        self.ds.reset_state()

        self.batch_size = batch_size
        self.num_workers = num_workers

    def __iter__(self):
        for batch in self.ds.get_data():
            input_ids, input_mask, segment_ids, lm_label_ids, is_next, image_feat, \
            image_loc, image_target, image_label, image_mask, image_id = batch

            batch_size = input_ids.shape[0]
            g_image_feat = np.sum(image_feat, axis=1) / np.sum(image_mask, axis=1, keepdims=True)
            image_feat = np.concatenate([np.expand_dims(g_image_feat, axis=1), image_feat], axis=1)
            image_feat = np.array(image_feat, dtype=np.float32)

            g_image_loc = np.repeat(np.array([[0, 0, 1, 1, 1]], dtype=np.float32), batch_size, axis=0)
            image_loc = np.concatenate([np.expand_dims(g_image_loc, axis=1), image_loc], axis=1)

            image_loc = np.array(image_loc, dtype=np.float32)
            g_image_mask = np.repeat(np.array([[1]]), batch_size, axis=0)
            image_mask = np.concatenate([g_image_mask, image_mask], axis=1)

            # batch = (input_ids, input_mask, segment_ids, lm_label_ids, is_next, image_feat, \
            # image_loc, image_target, image_label, image_mask, image_id)
            batch = (input_ids, input_mask, segment_ids, lm_label_ids, is_next, image_feat, \
                     image_loc, image_target, image_label, image_mask, image_id)

            # yield tuple([torch.tensor(data) for data in batch] + [image_id])
            yield tuple([torch.tensor(data) for data in batch])

    def __len__(self):
        return self.ds.size()

    def reset_index(self):
        core_ds = self.get_core_ds()
        core_ds.reset_index()


class BertPreprocessBatch(object):
    def __init__(
            self,
            caption_path,
            tokenizer,
            seq_len,
            region_len,
            data_size,
            split="Train",
            encoding="utf-8",
            predict_feature=False,
            visualization=False
    ):
        self.noun = np.load(ID2CLASS_PATH, allow_pickle=True).item()
        self.vocabulary_list = list(tokenizer.vocab.items())
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
        num_boxes = min(int(num_boxes), REGION_LEN)
        image_feature[:num_boxes] = image_feature_wp[:num_boxes]
        image_target[:num_boxes] = image_target_wp[:num_boxes]
        image_location[:num_boxes, :4] = image_location_wp[:num_boxes]
        # num_boxes = int(num_boxes)
        # image_feature[:num_boxes] = image_feature_wp
        # image_target[:num_boxes] = image_target_wp
        # image_location[:num_boxes, :4] = image_location_wp

        image_location[:, 4] = (image_location[:, 3] - image_location[:, 1]) * (
                image_location[:, 2] - image_location[:, 0]) / (float(image_w) * float(image_h))

        image_location[:, 0] = image_location[:, 0] / float(image_w)
        image_location[:, 1] = image_location[:, 1] / float(image_h)
        image_location[:, 2] = image_location[:, 2] / float(image_w)
        image_location[:, 3] = image_location[:, 3] / float(image_h)

        # image_feature_og = image_feature
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
        input_ids, input_mask, segment_ids, lm_label_ids, is_next, image_feat, image_target, image_loc, \
        image_label, image_mask, causal_label_t, causal_label_v \
            = self.convert_example_to_features(cur_example, self.seq_len, self.tokenizer, self.region_len)

        cur_tensors = (
            input_ids,
            input_mask,
            segment_ids,
            lm_label_ids,
            is_next,
            image_feat,
            image_loc,
            image_target,
            image_label,
            image_mask,
            int(image_id),
            causal_label_t,
            causal_label_v
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

        if random.random() > 0.5:
            label = 0
        else:
            caption = self.get_random_caption()
            label = 1

        return caption, label

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
        # tokens_og = ["[CLS]"] + caption + ["[SEP]"]
        caption, caption_label, causal_label_t = self.random_word(caption, tokenizer)

        image_feat, image_loc, image_label, causal_label_v = self.random_region(image_feat, image_loc, num_boxes)

        lm_label_ids = [-1] + caption_label + [-1]
        causal_label_t = [-1] + causal_label_t + [-1]

        tokens = []
        segment_ids = []

        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in caption:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # input_ids_og = tokenizer.convert_tokens_to_ids(tokens_og)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        # input_ids = input_ids[:1] input_ids[1:]
        input_mask = [1] * (len(input_ids))
        image_mask = [1] * (num_boxes)
        # Zero-pad up to the visual sequence length.
        while len(image_mask) < max_region_length:
            image_mask.append(0)
            image_label.append(-1)
            causal_label_v.append(-1)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            # input_ids_og.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            lm_label_ids.append(-1)
            causal_label_t.append(-1)
            # causal_mask_t.append(np.zeros(len(self.noun), dtype=np.float32))

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

        features = (
            np.array(input_ids),
            np.array(input_mask),
            np.array(segment_ids),
            np.array(lm_label_ids),
            np.array(example.is_next),
            image_feat,
            image_target,
            image_loc,
            np.array(image_label),
            np.array(image_mask),
            # np.array(input_ids_og),
            np.array(causal_label_t),
            np.array(causal_label_v)
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
        causal_label_t = []

        for i, token in enumerate(tokens):
            prob = random.random()
            # mask token with 15% probability
            if prob < 0.15 and not self.visualization:
                prob /= 0.15
                if prob < 0.8:
                    tokens[i] = "[MASK]"
                    causal_label_t.append(-1)
                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.choice(self.vocabulary_list)[0]
                    causal_label_t.append(-1)
                else:
                    if tokenizer.vocab[token] in self.noun:
                        causal_label_t.append(tokenizer.vocab[token])
                    else:
                        causal_label_t.append(-1)

                try:
                    output_label.append(tokenizer.vocab[token])
                except KeyError:
                    # For unknown words (should not occur with BPE vocab)
                    output_label.append(tokenizer.vocab["[UNK]"])
                    logger.warning(
                        "Cannot find token '{}' in vocab. Using [UNK] insetad".format(token)
                    )
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)
                if tokenizer.vocab[token] in self.noun:
                    causal_label_t.append(tokenizer.vocab[token])
                else:
                    causal_label_t.append(-1)

        return tokens, output_label, causal_label_t

    def random_region(self, image_feat, image_loc, num_boxes):
        """
        """
        image_label = []
        causal_label_v = []

        for i in range(num_boxes):
            prob = random.random()
            # mask token with 15% probability
            if prob < 0.15 and not self.visualization:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.9:
                    image_feat[i] = 0
                    causal_label_v.append(-1)
                else:
                    causal_label_v.append(1)
                # 10% randomly change token to random token
                # elif prob < 0.9:
                # tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

                # -> rest 10% randomly keep current token
                # append current token to output (we will predict these later)
                image_label.append(1)
            else:
                # no masking token (will be ignored by loss function later)
                causal_label_v.append(1)
                image_label.append(-1)

        return image_feat, image_loc, image_label, causal_label_v


class ConceptCapLoaderRetrieval(object):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.
    Arguments:
        mode (str, required): mode of dataset to operate in, one of ['train', 'val']
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 0)
        cache (int, optional): cache size to use when loading data,
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
        cuda (bool, optional): set to ``True`` and the PyTorch tensors will get preloaded
            to the GPU for you (necessary because this lets us to uint8 conversion on the
            GPU, which is faster).
    """

    def __init__(
            self,
            corpus_path,
            tokenizer,
            seq_len,
            encoding="utf-8",
            predict_feature=False,
            batch_size=512,
            shuffle=False,
            num_workers=10,
            cache=50000,
            drop_last=False,
            cuda=False,
    ):

        lmdb_file = "/coc/dataset/conceptual_caption/validation_feat_all.lmdb"
        if not os.path.exists(lmdb_file):
            lmdb_file = "/coc/pskynet2/jlu347/multi-modal-bert/data/conceptual_caption/validation_feat_all.lmdb"
        caption_path = "/coc/pskynet2/jlu347/multi-modal-bert/data/conceptual_caption/caption_val.json"

        print("Loading from %s" % lmdb_file)

        ds = td.LMDBSerializer.load(lmdb_file, shuffle=False)
        self.num_dataset = len(ds)
        preprocess_function = BertPreprocessRetrieval(
            caption_path,
            tokenizer,
            seq_len,
            REGION_LEN,
            1000,
            encoding="utf-8",
            predict_feature=predict_feature,
        )

        ds = td.MapData(ds, preprocess_function)
        self.ds = td.BatchData(ds, 1)
        self.ds.reset_state()

        self.batch_size = 1
        self.num_workers = num_workers
        self._entry = []

        self.features_all = np.zeros((1000, 37, 2048), dtype=np.float32)
        self.spatials_all = np.zeros((1000, 37, 5), dtype=np.float32)
        self.image_mask_all = np.zeros((1000, 37), dtype=np.float32)
        self.image_ids = []
        # load first 1000 file here.
        for i, batch in enumerate(self.ds.get_data()):
            if i >= 1000:
                break
            input_ids, input_mask, segment_ids, is_next, image_feat, \
            image_loc, image_mask, image_id, caption = batch

            batch_size = input_ids.shape[0]
            g_image_feat = np.sum(image_feat, axis=1) / np.sum(image_mask, axis=1, keepdims=True)
            image_feat = np.concatenate([np.expand_dims(g_image_feat, axis=1), image_feat], axis=1)
            image_feat = np.array(image_feat, dtype=np.float32)

            g_image_loc = np.repeat(np.array([[0, 0, 1, 1, 1]], dtype=np.float32), batch_size, axis=0)
            image_loc = np.concatenate([np.expand_dims(g_image_loc, axis=1), image_loc], axis=1)

            image_loc = np.array(image_loc, dtype=np.float32)
            g_image_mask = np.repeat(np.array([[1]]), batch_size, axis=0)
            image_mask = np.concatenate([g_image_mask, image_mask], axis=1)

            batch = (input_ids, input_mask, segment_ids, image_id, caption)
            self._entry.append(batch)

            self.features_all[i] = image_feat
            self.image_mask_all[i] = np.array(image_mask)
            self.spatials_all[i] = image_loc
            self.image_ids.append(image_id)
            sys.stdout.write('%d/%d\r' % (i, 1000))
            sys.stdout.flush()

    def __iter__(self):

        for index in range(self.__len__()):
            caption_idx = int(index / 2)
            image_idx = index % 2

            if image_idx == 0:
                image_entries = self.image_ids[:500]
                features_all = self.features_all[:500]
                spatials_all = self.spatials_all[:500]
                image_mask_all = self.image_mask_all[:500]

            else:
                image_entries = self.image_ids[500:]
                features_all = self.features_all[500:]
                spatials_all = self.spatials_all[500:]
                image_mask_all = self.image_mask_all[500:]

            caption, input_mask, segment_ids, txt_image_id, caption = self._entry[caption_idx]
            target_all = np.zeros((500))
            for i, image_id in enumerate(image_entries):
                if image_id == txt_image_id:
                    target_all[i] = 1

            batch = (
                features_all, spatials_all, image_mask_all, caption, input_mask, segment_ids, target_all, caption_idx,
                image_idx)
            batch = [torch.tensor(data) for data in batch]
            batch.append(txt_image_id)
            batch.append(caption)

            yield batch

    def __len__(self):
        return len(self._entry) * 2


class BertPreprocessRetrieval(object):
    def __init__(
            self,
            caption_path,
            tokenizer,
            seq_len,
            region_len,
            data_size,
            split="Train",
            encoding="utf-8",
            predict_feature=False,
    ):

        self.split = split
        self.seq_len = seq_len
        self.region_len = region_len
        self.tokenizer = tokenizer
        self.predict_feature = predict_feature
        self.num_caps = data_size
        self.captions = list(json.load(open(caption_path, 'r')).values())[:data_size]

    def __call__(self, data):

        image_feature_wp, image_target_wp, image_location_wp, num_boxes, image_h, image_w, image_id, caption = data

        image_feature = np.zeros((self.region_len, 2048), dtype=np.float32)
        image_target = np.zeros((self.region_len, 1601), dtype=np.float32)
        image_location = np.zeros((self.region_len, 5), dtype=np.float32)

        num_boxes = int(num_boxes)
        image_feature[:num_boxes] = image_feature_wp
        image_target[:num_boxes] = image_target_wp
        image_location[:num_boxes, :4] = image_location_wp

        image_location[:, 4] = (image_location[:, 3] - image_location[:, 1]) * (
                image_location[:, 2] - image_location[:, 0]) / (float(image_w) * float(image_h))

        image_location[:, 0] = image_location[:, 0] / float(image_w)
        image_location[:, 1] = image_location[:, 1] / float(image_h)
        image_location[:, 2] = image_location[:, 2] / float(image_w)
        image_location[:, 3] = image_location[:, 3] / float(image_h)

        label = 0

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
            cur_features.is_next,
            cur_features.image_feat,
            cur_features.image_loc,
            cur_features.image_mask,
            float(image_id),
            caption,
        )
        return cur_tensors

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
        # image_target = example.image_target
        num_boxes = int(example.num_boxes)
        self._truncate_seq_pair(caption, max_seq_length - 2)
        # caption, caption_label = self.random_word(caption, tokenizer)
        caption_label = None
        # image_feat, image_loc, image_label = self.random_region(image_feat, image_loc, num_boxes)
        image_label = None

        tokens = []
        segment_ids = []

        tokens.append("[CLS]")
        segment_ids.append(0)

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

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(image_mask) == max_region_length

        features = InputFeatures(
            input_ids=np.array(input_ids),
            input_mask=np.array(input_mask),
            segment_ids=np.array(segment_ids),
            is_next=np.array(example.is_next),
            image_feat=image_feat,
            image_loc=image_loc,
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
