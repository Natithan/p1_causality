# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# pylint: disable=no-member
"""
TridentNet Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import time
import json
import csv
from pathlib import Path
from tensorpack.dataflow import DataFlow, RNGDataFlow, PrefetchDataZMQ, LMDBSerializer, BatchData
from tools.DownloadConcptualCaption.download_data import _file_name
from tqdm import tqdm
FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features', 'cls_prob']
import sys
import pandas as pd

ROOT_DIR = "/cw/liir/NoCsBack/testliir/nathan/p1_causality"
csv.field_size_limit(sys.maxsize)

import argparse
import os
import sys
import torch
import cv2
import numpy as np

sys.path.append('buatest/detectron2')

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.structures import Instances

from utils.utils import mkdir, save_features
from utils.extract_utils import get_image_blob, save_bbox, save_roi_features_by_bbox, save_roi_features, \
    prep_roi_features
from utils.progress_bar import ProgressBar
from models import add_config
from models.bua.box_regression import BUABoxes
from cfg import FGS
import ray
from ray.actor import ActorHandle

BUA_ROOT_DIR = "buatest"

os.environ['CUDA_VISIBLE_DEVICES'] = FGS.gpus
import torch;torch._C._cuda_init()


# --mode
# caffe
# --num-cpus
# 32
# --gpus
# '0,1,2,3'
# --extract-mode
# roi_feats
# --min-max-boxes
# 10,100
# --config-file
# buatest/configs/bua-caffe/extract-bua-caffe-r101.yaml
# --image-dir
# /cw/liir/NoCsBack/testliir/datasets/ConceptualCaptions/training
# --mydebug
# --num_samples
# 10
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

    for im_file in (img_list):
        if os.path.exists(os.path.join(args.output_dir, im_file.split('.')[0] + '.npz')):
            actor.update.remote(1)
            continue
        im = cv2.imread(os.path.join(args.image_dir, im_file))
        if im is None:
            print(os.path.join(args.image_dir, im_file), "is illegal!")
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
                         args, cfg, im_file, im, dataset_dict,
                         boxes, scores, features_pooled, attr_scores)
        # extract bbox only
        elif cfg.MODEL.BUA.EXTRACTOR.MODE == 2:
            with torch.set_grad_enabled(False):
                boxes, scores = model([dataset_dict])
            boxes = [box.cpu() for box in boxes]
            scores = [score.cpu() for score in scores]
            generate_npz(2,
                         args, cfg, im_file, im, dataset_dict,
                         boxes, scores)
        # extract roi features by bbox
        elif cfg.MODEL.BUA.EXTRACTOR.MODE == 3:
            if not os.path.exists(os.path.join(args.bbox_dir, im_file.split('.')[0] + '.npz')):
                actor.update.remote(1)
                continue
            bbox = torch.from_numpy(np.load(os.path.join(args.bbox_dir, im_file.split('.')[0] + '.npz'))['bbox']) * \
                   dataset_dict['im_scale']
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
                         args, cfg, im_file, im, dataset_dict,
                         boxes, scores, features_pooled, attr_scores)

        actor.update.remote(1)


# Nathan: for easy debugging
def extract_feat_no_ray(split_idx, img_list, cfg, args):
    num_images = len(img_list)
    print('Number of images on split{}: {}.'.format(split_idx, num_images))

    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        Path(BUA_ROOT_DIR, cfg.MODEL.WEIGHTS).as_posix(), resume=args.resume  # TODO update the model weights location
    )
    model.eval()

    for im_file in (img_list):
        # if os.path.exists(os.path.join(args.output_dir, im_file.split('.')[0]+'.npz')):
        #     continue
        im = cv2.imread(os.path.join(args.image_dir, im_file))
        if im is None:
            print(os.path.join(args.image_dir, im_file), "is illegal!")
            continue
        return image_to_intermediate(cfg, im, im_file, model)
        # generate_npz(1,
        #     args, cfg, im_file, im, dataset_dict,
        #     boxes, scores, features_pooled, attr_scores)
        # # extract bbox only
        # elif cfg.MODEL.BUA.EXTRACTOR.MODE == 2:
        #     with torch.set_grad_enabled(False):
        #         boxes, scores = model([dataset_dict])
        #     boxes = [box.cpu() for box in boxes]
        #     scores = [score.cpu() for score in scores]
        #     generate_npz(2,
        #         args, cfg, im_file, im, dataset_dict,
        #         boxes, scores)
        # # extract roi features by bbox
        # elif cfg.MODEL.BUA.EXTRACTOR.MODE == 3:
        #     if not os.path.exists(os.path.join(args.bbox_dir, im_file.split('.')[0]+'.npz')):
        #         continue
        #     bbox = torch.from_numpy(np.load(os.path.join(args.bbox_dir, im_file.split('.')[0]+'.npz'))['bbox']) * dataset_dict['im_scale']
        #     proposals = Instances(dataset_dict['image'].shape[-2:])
        #     proposals.proposal_boxes = BUABoxes(bbox)
        #     dataset_dict['proposals'] = proposals
        #
        #     attr_scores = None
        #     with torch.set_grad_enabled(False):
        #         if cfg.MODEL.BUA.ATTRIBUTE_ON:
        #             boxes, scores, features_pooled, attr_scores = model([dataset_dict])
        #         else:
        #             boxes, scores, features_pooled = model([dataset_dict])
        #     boxes = [box.tensor.cpu() for box in boxes]
        #     scores = [score.cpu() for score in scores]
        #     features_pooled = [feat.cpu() for feat in features_pooled]
        #     if not attr_scores is None:
        #         attr_scores = [attr_score.data.cpu() for attr_score in attr_scores]
        #     generate_npz(3,
        #         args, cfg, im_file, im, dataset_dict,
        #         boxes, scores, features_pooled, attr_scores)


def image_to_intermediate(cfg, im, im_file, model):
    dataset_dict = get_image_blob(im, cfg.MODEL.PIXEL_MEAN)
    # extract roi features
    # if cfg.MODEL.BUA.EXTRACTOR.MODE == 1:
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
    image_bboxes, image_feat, info, keep_boxes = prep_roi_features(attr_scores, boxes, cfg, dataset_dict,
                                                                   features_pooled, im, im_file, scores)
    return image_bboxes, image_feat, info, keep_boxes


def open_tsv(fname, folder):
    print("Opening %s Data File..." % fname)
    df = pd.read_csv(fname, sep='\t', names=["caption", "url"], usecols=range(0, 2))
    # df = pd.read_csv(fname, sep='\t', names=["caption", "url"], usecols=range(0, 2), nrows=200)  # Nathan edited
    df['folder'] = folder
    print("Processing", len(df), " Images:")
    return df


# def _file_name(row):
#     return "%s/%s" % (row['folder'], (zlib.crc32(row['url'].encode('utf-8')) & 0xffffffff))

class CoCaInputDataflow(DataFlow):
    def __init__(self,image_dir,max_nb_images=-1):
        self.image_dir = image_dir
        self.image_names = []
        print("Gathering image paths ...")
        s = time.time()
        for i,f in enumerate(os.scandir(image_dir)):
            if (max_nb_images > 0) and (i >= max_nb_images):
                break
            self.image_names.append(f.name)
        print(f"Done gathering image paths after {time.time()-s} seconds")
        self.num_files = len(self.image_names)
        print('Number of images: {}.'.format(self.num_files))

    def __iter__(self):
        for im_name in self.image_names:
            im = cv2.imread(os.path.join(self.image_dir, im_name))
            if im is None:
                print(os.path.join(self.image_dir, im_name), "is illegal!")
                continue
            yield get_image_blob(im, FGS.pixel_mean)

    def __len__(self):
        return self.num_files

class CoCaDataFlow(RNGDataFlow):
    """
    """

    def __init__(self, cfg, args, shuffle=False):
        self.shuffle = shuffle
        self.cfg = cfg
        self.img_to_input_df = BatchData(CoCaInputDataflow(FGS.image_dir, FGS.max_nb_images), FGS.batch_size, use_list=True)
        # print("Gathering image paths ...")
        # self.image_dir = args.image_dir
        # self.image_files = list(os.scandir(image_dir))
        # self.image_names = []
        # for i,f in enumerate(os.scandir(args.image_dir)):
        #     if (args.num_samples > 0) and (i >= args.num_samples):
        #         break
        #     self.image_names.append(f.name)
        # self.num_caps = len(self.image_names)  # TODO: compute the number of PROCESSED pics
        # print('Number of images: {}.'.format(self.num_caps))
        # print("Done gathering image paths")

        self.model = DefaultTrainer.build_model(cfg)
        DetectionCheckpointer(self.model, save_dir=self.cfg.OUTPUT_DIR).resume_or_load(
            Path(BUA_ROOT_DIR, cfg.MODEL.WEIGHTS).as_posix(), resume=args.resume
        )
        self.model.eval()

        caption_path = Path(ROOT_DIR, 'DeVLBert', 'features_lmdb/CC/caption_train.json')
        if os.path.exists(caption_path):
            print(f"Not storing caption_train.json, already present at {caption_path}")
            self.captions = json.load(open(caption_path, 'r'))
        else:
            @ray.remote
            def index_captions(df):
                captions = {}
                for i, img in enumerate(df.iterrows()):
                    caption = img[1]['caption']  # .decode("utf8")
                    im_name = _file_name(img[1])
                    image_id = im_name.split('/')[1]
                    # image_id = str(i)
                    captions[image_id] = caption
                return captions

            df = open_tsv(Path(ROOT_DIR, 'DeVLBert/tools/DownloadConcptualCaption/Train_GCC-training.tsv'), 'training')
            print("Indexing captions ...")
            ray.init()
            futures = [index_captions.remote(df[i::args.num_cpus]) for i in range(args.num_cpus)]
            l = ray.get(futures)
            self.captions = {}
            for d in l:
                self.captions = {**self.captions, **d}
            print("Done")
            print(f"Storing caption_train.json ...")
            json.dump(self.captions, open(caption_path, 'w'))
            print(f"Done")


    def __len__(self):
        return len(self.img_to_input_df)

    def __iter__(self):
        for rcnn_input in self.img_to_input_df.get_data():

            rcnn_input = [{k:list(v)[i] for k,v in rcnn_input.items()} for i,_ in enumerate(list(rcnn_input.values())[0])] # tensorpack dataflow gives dict of lists, rcnn expects list of dicts
            with torch.set_grad_enabled(False):
                    model_outputs = self.model(rcnn_input) # boxes, scores, features_pooled
            for i,o in enumerate(model_outputs):
                model_outputs[i] = [e.cpu() if type(e) is torch.Tensor else e.tensor.cpu() for e in o]
            # boxes = [box.tensor.cpu() for box in boxes]
            # scores = [score.cpu() for score in scores]
            # features_pooled = [feat.cpu() for feat in features_pooled]
            # if not attr_scores is None:
            #     attr_scores = [attr_score.cpu() for attr_score in attr_scores]
            image_bboxes, image_feat, info, keep_boxes = prep_roi_features(attr_scores, boxes, cfg, dataset_dict,
                                                                           features_pooled, im, im_file, scores)
            return image_bboxes, image_feat, info, keep_boxes
            image_bboxes, image_feat, info, keep_boxes = image_to_intermediate(self.cfg, im, im_file, self.model) #TODO fix "RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method"
            image_id = info['image_id']
            image_h = info['image_h']
            image_w = info['image_w']
            num_boxes = len(keep_boxes)
            boxes = image_bboxes
            features = image_feat
            cls_prob = info['obj_cls_prob']

            caption = self.captions[image_id]

            yield [features, cls_prob, boxes, num_boxes, image_h, image_w, image_id, caption]
            # for infile in self.infiles:
            #     count = 0
            # with open(infile) as tsv_in_file:
            #     reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
            #     for item in reader:
            #         image_id = item['image_id']
            #         image_h = item['image_h']
            #         image_w = item['image_w']
            #         num_boxes = item['num_boxes']
            #         boxes = np.frombuffer(base64.b64decode(item['boxes']), dtype=np.float32).reshape(int(num_boxes), 4)
            #         features = np.frombuffer(base64.b64decode(item['features']), dtype=np.float32).reshape(int(num_boxes), 2048)
            #         cls_prob = np.frombuffer(base64.b64decode(item['cls_prob']), dtype=np.float32).reshape(int(num_boxes), 1601)
            #         caption = self.captions[image_id]
            #
            #         yield [features, cls_prob, boxes, num_boxes, image_h, image_w, image_id, caption]


def main():
    # region Parser stuff
    parser = argparse.ArgumentParser(description="PyTorch Object Detection2 Inference")
    parser.add_argument(
        "--config-file",
        default="configs/bua-caffe/extract-bua-caffe-r101.yaml",
        metavar="FILE",
        help="path to config file",
    )

    # Nathan

    parser.add_argument('--mydebug',
                        action="store_true",
                        help="if active, ray is not used, as it doesn't allow the pycharm debugger to go into the subprocess")

    parser.add_argument('--num-cpus', default=1, type=int,
                        help='number of cpus to use for ray, 0 means no limit')

    parser.add_argument('--num_samples', default=0, type=int,
                        help='number of samples to convert, 0 means all')
    parser.add_argument('--gpus', dest='gpu_id', help='GPU id(s) to use',
                        default='0', type=str)

    parser.add_argument("--mode", default="caffe", type=str, help="bua_caffe, ...")

    parser.add_argument('--extract-mode', default='roi_feats', type=str,
                        help="'roi_feats', 'bboxes' and 'bbox_feats' indicates \
                        'extract roi features directly', 'extract bboxes only' and \
                        'extract roi features with pre-computed bboxes' respectively")

    parser.add_argument('--min-max-boxes', default='min_max_default', type=str,
                        help='the number of min-max boxes of extractor')

    parser.add_argument('--out-dir', dest='output_dir',
                        help='output directory for features',
                        default="features")
    parser.add_argument('--image-dir', dest='image_dir',
                        help='directory with images',
                        default="image")
    parser.add_argument('--bbox-dir', dest='bbox_dir',
                        help='directory with bbox',
                        default="bbox")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    # endregion
    start = time.time()
    cfg = setup(args)
    print(f"Setup done after {time.time()-start}")
    num_gpus = len(args.gpu_id.split(','))

    # Extract features.
    # imglist = os.listdir(args.image_dir)
    # num_images = len(imglist)
    # print('Number of images: {}.'.format(num_images))
    # if not args.mydebug:
    #     if args.num_cpus != 0:
    #         ray.init(num_cpus=args.num_cpus)
    #     else:
    #         ray.init()
    #     pb = ProgressBar(len(imglist))
    #     actor = pb.actor
    #
    # img_lists = [imglist[i::num_gpus] for i in range(num_gpus)]

    # print('Number of GPUs: {}.'.format(num_gpus))
    # extract_feat_list = []
    # for i in range(num_gpus):
    #     if not args.mydebug:
    #         extract_feat_list.append(extract_feat.remote(i, img_lists[i], cfg, args, actor))
    #     else:
    #         extract_feat_list.append(extract_feat_no_ray(i, img_lists[i], cfg, args))
    #
    # if not args.mydebug:
    #     pb.print_until_done()
    #     ray.get(extract_feat_list)
    #     ray.get(actor.get_counter.remote())

    # corpus_path = Path(ROOT_DIR, 'buatest', 'features')
    ds = CoCaDataFlow(cfg, args)

    print(f"Conceptual_Caption init done after {time.time()-start}")
    next(ds.get_data())# TODO remove
    # TODO check the size on this
    ds1 = PrefetchDataZMQ(ds, num_proc=args.num_cpus) #TODO get LMDB saving to speed up with parallelization via PrefetchDataZMQ
    # ds1 = ds
    LMDBSerializer.save(ds1, str(Path(ROOT_DIR, 'DeVLBert', f'features_lmdb/CC/training_feat_{args.num_samples if args.num_samples > 0 else "all"}_debug_{int(time.time())}.lmdb')))
    print(f"Done after {time.time()-start}")
    # LMDBSerializer.save(ds1, '/mnt3/yangan.ya/features_lmdb/CC/training_feat_all.lmdb')


if __name__ == '__main__':
    main()
