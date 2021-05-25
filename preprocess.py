# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# pylint: disable=no-member
"""
TridentNet Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import pickle
from time import time
import json
import csv
from pathlib import Path
from tensorpack.dataflow import DataFlow, RNGDataFlow, BatchData
from constants import CHECKPOINT_FREQUENCY, PROJECT_ROOT_DIR, STORAGE_DIR, BUA_ROOT_DIR
from my_lmdb import MyLMDBSerializer
import sys

csv.field_size_limit(sys.maxsize)
sys.path.append('buatest/detectron2')
sys.path.append('DeVLBert')
import os

import cv2
import numpy as np
from util import index_df_column, open_tsv, get_world_size

import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup
from detectron2.structures import Instances

from buatest.utils.extract_utils import get_image_blob, save_bbox, save_roi_features_by_bbox, save_roi_features, \
    prep_roi_features, filter_keep_boxes
from models import add_config
from models.bua.box_regression import BUABoxes
from preprocess_cfg import FGS
import ray
from ray.actor import ActorHandle

FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features', 'cls_prob']
WORLD_SIZE = len(FGS.gpus)
LOCAL_LMDB_ID = f"{FGS.lmdb_file}_{FGS.local_rank}_of_{WORLD_SIZE}"
#
# torch._C._cuda_init()
input_index_file = Path(FGS.lmdb_folder, f'{LOCAL_LMDB_ID}_{FGS.index_file}')
torch.cuda.set_device(int(FGS.gpus[FGS.local_rank]))
import torch.distributed as dist

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'
dist.init_process_group('nccl', rank=FGS.local_rank, world_size=len(FGS.gpus))


def switch_extract_mode(mode):
    if mode == 'roi_feats':
        switch_cmd = ['MODEL.BUA.EXTRACTOR.MODE', 1]
    elif mode == 'bboxes':
        switch_cmd = ['MODEL.BUA.EXTRACTOR.MODE', 2]
    elif mode == 'bbox_feats':
        switch_cmd = ['MODEL.BUA.EXTRACTOR.MODE', 3, 'MODEL.PROPOSAL_GENERATOR.NAME', 'PrecomputedProposals']
    else:
        print('Wrong extract mode! ')
        exit()
    return switch_cmd


def set_min_max_boxes(min_max_boxes):
    if min_max_boxes == 'min_max_default':
        return []
    try:
        min_boxes = int(min_max_boxes.split(',')[0])
        max_boxes = int(min_max_boxes.split(',')[1])
    except:
        print('Illegal min-max boxes setting, using config default. ')
        return []
    cmd = ['MODEL.BUA.EXTRACTOR.MIN_BOXES', min_boxes,
           'MODEL.BUA.EXTRACTOR.MAX_BOXES', max_boxes]
    return cmd


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_config(args, cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.merge_from_list(switch_extract_mode(args.extract_mode))
    cfg.merge_from_list(set_min_max_boxes(args.min_max_boxes))
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def generate_npz(extract_mode, *args):
    if extract_mode == 1:
        save_roi_features(*args)
    elif extract_mode == 2:
        save_bbox(*args)
    elif extract_mode == 3:
        save_roi_features_by_bbox(*args)
    else:
        print('Invalid Extract Mode! ')


@ray.remote(num_gpus=1)
def extract_feat(split_idx, img_list, cfg, args, actor: ActorHandle):
    num_images = len(img_list)
    print('Number of images on split{}: {}.'.format(split_idx, num_images))

    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    model.eval()

    for img_file in (img_list):
        if os.path.exists(os.path.join(args.output_dir, img_file.split('.')[0] + '.npz')):
            actor.update.remote(1)
            continue
        im = cv2.imread(os.path.join(args.image_dir, img_file))
        if im is None:
            print(os.path.join(args.image_dir, img_file), "is illegal!")
            actor.update.remote(1)
            continue
        dataset_dict = get_image_blob(im, cfg.MODEL.PIXEL_MEAN)
        # extract roi features
        if cfg.MODEL.BUA.EXTRACTOR.MODE == 1:
            attr_scores = None
            with torch.set_grad_enabled(False):
                if cfg.MODEL.BUA.ATTRIBUTE_ON:
                    boxes, scores, features_pooled, attr_scores = model([dataset_dict])
                else:
                    boxes, scores, features_pooled = model([dataset_dict])
            boxes = [box.tensor.cpu() for box in boxes]
            scores = [score.cpu() for score in scores]
            features_pooled = [feat.cpu() for feat in features_pooled]
            if not attr_scores is None:
                attr_scores = [attr_score.cpu() for attr_score in attr_scores]
            generate_npz(1,
                         args, cfg, img_file, im, dataset_dict,
                         boxes, scores, features_pooled, attr_scores)
        # extract bbox only
        elif cfg.MODEL.BUA.EXTRACTOR.MODE == 2:
            with torch.set_grad_enabled(False):
                boxes, scores = model([dataset_dict])
            boxes = [box.cpu() for box in boxes]
            scores = [score.cpu() for score in scores]
            generate_npz(2,
                         args, cfg, img_file, im, dataset_dict,
                         boxes, scores)
        # extract roi features by bbox
        elif cfg.MODEL.BUA.EXTRACTOR.MODE == 3:
            if not os.path.exists(os.path.join(args.bbox_dir, img_file.split('.')[0] + '.npz')):
                actor.update.remote(1)
                continue
            bbox = torch.from_numpy(np.load(os.path.join(args.bbox_dir, img_file.split('.')[0] + '.npz'))['bbox']) * \
                   dataset_dict['img_scale']
            proposals = Instances(dataset_dict['image'].shape[-2:])
            proposals.proposal_boxes = BUABoxes(bbox)
            dataset_dict['proposals'] = proposals

            attr_scores = None
            with torch.set_grad_enabled(False):
                if cfg.MODEL.BUA.ATTRIBUTE_ON:
                    boxes, scores, features_pooled, attr_scores = model([dataset_dict])
                else:
                    boxes, scores, features_pooled = model([dataset_dict])
            boxes = [box.tensor.cpu() for box in boxes]
            scores = [score.cpu() for score in scores]
            features_pooled = [feat.cpu() for feat in features_pooled]
            if not attr_scores is None:
                attr_scores = [attr_score.data.cpu() for attr_score in attr_scores]
            generate_npz(3,
                         args, cfg, img_file, im, dataset_dict,
                         boxes, scores, features_pooled, attr_scores)

        actor.update.remote(1)


class CoCaInputDataflow(DataFlow):
    def __init__(self, image_dir, max_nb_images=-1):

        self.image_dir = image_dir
        self.image_ids = []
        print("Gathering image paths ...")
        s = time()
        cached_id_path = Path('blobs',f'image_ids_{FGS.max_nb_images}.p' if FGS.max_nb_images > 0 else 'image_ids_full.p')
        if Path.exists(cached_id_path):
            with open(cached_id_path,'rb') as f:
                self.image_ids = pickle.load(f)
        else:
            for i, f in enumerate(os.scandir(image_dir)):
                if (max_nb_images > 0) and (i >= max_nb_images):
                    break
                self.image_ids.append(f.name)
            with open(cached_id_path,'wb') as f:
                pickle.dump(self.image_ids, f)
        self.image_ids = self.image_ids[FGS.local_rank::WORLD_SIZE]
        print(f"Done gathering image paths after {time() - s} seconds")
        self.num_files = len(self.image_ids)
        print('Number of images: {}.'.format(self.num_files))
        print(FGS.local_rank, get_world_size())


        # region Storing / loading captions
        caption_file = 'caption_train.json'
        caption_path = Path(STORAGE_DIR, caption_file)

        if os.path.exists(caption_path):
            print(f"Not generating captions and storing them, loading from {caption_path}.")
            s = time()
            self.captions = json.load(open(caption_path, 'r'))
            print(f"Loaded captions in {time() - s}.")
        else:
            df = open_tsv(Path(PROJECT_ROOT_DIR, 'DeVLBert/tools/DownloadConcptualCaption/Train_GCC-training.tsv'), 'training')
            print("Indexing captions ...")
            ray.init()
            futures = [index_df_column.remote(df[i::FGS.num_cpus], 'caption') for i in range(FGS.num_cpus)]
            l = ray.get(futures)
            self.captions = {}
            for d in l:
                self.captions = {**self.captions, **d}
            print("Done indexing captions")
            print(f"Storing caption_train.json ...")
            json.dump(self.captions, open(caption_path, 'w'))
            print(f"Done storing captions")
        # endregion

        self.clean_count = 0
        self.dirty_count = 0
        if not FGS.from_scratch:
            if os.path.exists(input_index_file):
                print("Starting from existing index")
                with open(input_index_file, 'rb') as f:
                    self.clean_count, self.dirty_count = pickle.load(f)
            else:
                with open(input_index_file, 'wb') as f:
                    pickle.dump((self.clean_count, self.dirty_count), f)

    def total_count(self):
        return self.clean_count + self.dirty_count

    def __iter__(self):
        for image_id in self.image_ids[self.total_count():]:
            s = time()
            im = cv2.imread(os.path.join(self.image_dir, image_id)) #TODO maybe parallelize this with CPUs to not make it a bottleneck?

            if self.clean_count % CHECKPOINT_FREQUENCY == 0:
                pickle.dump((self.clean_count, self.dirty_count),
                            open(input_index_file, 'wb'))

            if im is None:
                # print(os.path.join(self.image_dir, image_id), "is illegal!")
                self.dirty_count += 1
                continue
            self.clean_count += 1
            # print(f"{type(self).__name__} iteration: {round(time() - s,3)} s\r\n")
            yield {**get_image_blob(im, FGS.pixel_mean),
                   "img_id": image_id,
                   "img_width": im.shape[0],
                   "img_height": im.shape[1],
                   "caption": self.captions[image_id]}

    def __len__(self):
        return self.num_files

class CoCaDataFlow(RNGDataFlow):
    """
    RPN input to filtered RPN output
    """

    def __init__(self, cfg, args, shuffle=False):
        self.shuffle = shuffle
        self.cfg = cfg
        self.non_batched_img_to_input_df = CoCaInputDataflow(FGS.image_dir, FGS.max_nb_images)
        self.img_to_input_df = BatchData(self.non_batched_img_to_input_df, FGS.batch_size,
                                         use_list=True)

        self.model = DefaultTrainer.build_model(cfg)
        DetectionCheckpointer(self.model, save_dir=self.cfg.OUTPUT_DIR).resume_or_load(
            Path(BUA_ROOT_DIR, cfg.MODEL.WEIGHTS).as_posix(), resume=args.resume
        )
        self.model.eval()
        # self.model = DataParallelV2(self.model)

    def __len__(self):
        return len(self.non_batched_img_to_input_df)

    def __iter__(self):
        for data_dict in self.img_to_input_df.get_data():
            s = time()
            rcnn_input = [{k: list(v)[i] for k, v in data_dict.items()} for i, _ in enumerate(
                list(data_dict.values())[0])]  # tensorpack dataflow gives dict of lists, rcnn expects list of dicts
            with torch.set_grad_enabled(False):
                model_outputs = list(self.model(rcnn_input))  # boxes, scores, features_pooled
            for i, o in enumerate(model_outputs):
                model_outputs[i] = [e.to('cpu') for e in o]
            boxes, scores, feats = model_outputs

            keep_idxs = []
            keep_feats = []
            keep_og_boxes = []
            keep_scores = []
            num_boxes = []
            for single_boxes, single_feats, single_scores, scale in zip(boxes, feats, scores, data_dict[
                'im_scale']):
                single_boxes = single_boxes.tensor  # TODO if I drop the BUABox wrapper here, and don't need it anywhere else, maybe drop it alltogether
                og_boxes = single_boxes / scale  # Nathan
                try:
                    keep_idxs_single = filter_keep_boxes(og_boxes, self.cfg, single_scores)
                except RuntimeError as e:
                    print(e)
                    print("Input at time of error", data_dict['img_id'])
                    self.img_to_input_df.ds.clean_count -= 1
                    self.img_to_input_df.ds.dirty_count += 1
                    continue
                keep_idxs.append(keep_idxs_single)

                keep_feats.append(single_feats[keep_idxs_single])
                keep_og_boxes.append(og_boxes[keep_idxs_single])
                keep_scores.append(single_scores[keep_idxs_single].numpy())
                num_boxes.append(len(keep_idxs_single))

            # print(f"{type(self).__name__} iteration: {round(time() - s, 3)} s\r\n")
            for f, s, b, nb, h, w, i, c in zip(keep_feats, keep_scores, keep_og_boxes, num_boxes,
                                               data_dict['img_height'], data_dict['img_width'], data_dict['img_id'],
                                               data_dict['caption']):
                yield [f, s, b, nb, h, w, i, c]


def main():
    args = FGS
    print(f"Setting up")
    start = time()
    cfg = setup(args)
    print(f"Setup done after {time() - start}")

    ds = CoCaDataFlow(cfg, args)

    print(f"Conceptual_Caption init done after {time() - start}")
    lmdb_path = str(Path(FGS.lmdb_folder, f'{LOCAL_LMDB_ID}.lmdb'))
    MyLMDBSerializer.save(ds, lmdb_path,
                          write_frequency=CHECKPOINT_FREQUENCY, args=FGS)
    print(f"Done after {time() - start}")


if __name__ == '__main__':
    main()
