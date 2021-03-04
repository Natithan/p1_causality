# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# pylint: disable=no-member
"""
TridentNet Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
from time import time
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
    prep_roi_features, filter_keep_boxes
from utils.progress_bar import ProgressBar
from models import add_config
from models.bua.box_regression import BUABoxes
from cfg import FGS
import ray
from ray.actor import ActorHandle

BUA_ROOT_DIR = "buatest"

os.environ['CUDA_VISIBLE_DEVICES'] = FGS.gpus
import torch;

torch._C._cuda_init()


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


# Nathan: for easy debugging
def extract_feat_no_ray(split_idx, img_list, cfg, args):
    num_images = len(img_list)
    print('Number of images on split{}: {}.'.format(split_idx, num_images))

    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        Path(BUA_ROOT_DIR, cfg.MODEL.WEIGHTS).as_posix(), resume=args.resume
    )
    model.eval()

    for img_file in (img_list):
        # if os.path.exists(os.path.join(args.output_dir, img_file.split('.')[0]+'.npz')):
        #     continue
        im = cv2.imread(os.path.join(args.image_dir, img_file))
        if im is None:
            print(os.path.join(args.image_dir, img_file), "is illegal!")
            continue
        return image_to_intermediate(cfg, im, img_file, model)
        # generate_npz(1,
        #     args, cfg, img_file, im, dataset_dict,
        #     boxes, scores, features_pooled, attr_scores)
        # # extract bbox only
        # elif cfg.MODEL.BUA.EXTRACTOR.MODE == 2:
        #     with torch.set_grad_enabled(False):
        #         boxes, scores = model([dataset_dict])
        #     boxes = [box.cpu() for box in boxes]
        #     scores = [score.cpu() for score in scores]
        #     generate_npz(2,
        #         args, cfg, img_file, im, dataset_dict,
        #         boxes, scores)
        # # extract roi features by bbox
        # elif cfg.MODEL.BUA.EXTRACTOR.MODE == 3:
        #     if not os.path.exists(os.path.join(args.bbox_dir, img_file.split('.')[0]+'.npz')):
        #         continue
        #     bbox = torch.from_numpy(np.load(os.path.join(args.bbox_dir, img_file.split('.')[0]+'.npz'))['bbox']) * dataset_dict['img_scale']
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
        #         args, cfg, img_file, im, dataset_dict,
        #         boxes, scores, features_pooled, attr_scores)


def image_to_intermediate(cfg, im, img_file, model):
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
                                                                   features_pooled, im, img_file, scores)
    return image_bboxes, image_feat, info, keep_boxes


def open_tsv(fname, folder):
    print("Opening %s Data File..." % fname)
    df = pd.read_csv(fname, sep='\t', names=["caption", "url"], usecols=range(0, 2))
    # df = pd.read_csv(fname, sep='\t', names=["caption", "url"], usecols=range(0, 2), nrows=200)  # Nathan edited
    df['folder'] = folder
    print("Processing", len(df), " Images:")
    return df


class CoCaInputDataflow(DataFlow):
    def __init__(self, image_dir, max_nb_images=-1):
        self.image_dir = image_dir
        self.image_ids = []
        print("Gathering image paths ...")
        s = time()
        for i, f in enumerate(os.scandir(image_dir)):
            if (max_nb_images > 0) and (i >= max_nb_images):
                break
            self.image_ids.append(f.name)
        print(f"Done gathering image paths after {time() - s} seconds")
        self.num_files = len(self.image_ids)
        print('Number of images: {}.'.format(self.num_files))

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
                    img_name = _file_name(img[1])
                    image_id = img_name.split('/')[1]
                    # image_id = str(i)
                    captions[image_id] = caption
                return captions

            df = open_tsv(Path(ROOT_DIR, 'DeVLBert/tools/DownloadConcptualCaption/Train_GCC-training.tsv'), 'training')
            print("Indexing captions ...")
            ray.init()
            futures = [index_captions.remote(df[i::FGS.num_cpus]) for i in range(FGS.num_cpus)]
            l = ray.get(futures)
            self.captions = {}
            for d in l:
                self.captions = {**self.captions, **d}
            print("Done")
            print(f"Storing caption_train.json ...")
            json.dump(self.captions, open(caption_path, 'w'))
            print(f"Done")

    def __iter__(self):
        for image_id in self.image_ids:
            im = cv2.imread(os.path.join(self.image_dir, image_id))
            if im is None:
                print(os.path.join(self.image_dir, image_id), "is illegal!")
                continue
            yield {**get_image_blob(im, FGS.pixel_mean),
                   "img_id": image_id,
                   "img_width": im.shape[0],
                   "img_height": im.shape[1],
                   "caption": self.captions[image_id]}

    def __len__(self):
        return self.num_files


from torch.nn.parallel._functions import Scatter, Gather
from torch.nn.parallel import DataParallel
import torch

# This code was copied from torch.nn.parallel and adapted for DataParallel to chunk lists instead of duplicating them
# (this is really all this code is here for)
#from https://discuss.pytorch.org/t/dataparallel-chunking-for-a-list-of-3d-tensors/15962/7
class DataParallelV2(DataParallel):

    def scatter_kwargs(self, inputs, kwargs, target_gpus, dim=0):
        r"""Scatter with support for kwargs dictionary"""
        inputs = self.scatter_wrapper(inputs, target_gpus, dim) if inputs else []
        kwargs = self.scatter_wrapper(kwargs, target_gpus, dim) if kwargs else []
        if len(inputs) < len(kwargs):
            inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
        elif len(kwargs) < len(inputs):
            kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
        inputs = tuple(inputs)
        kwargs = tuple(kwargs)
        return inputs, kwargs

    def scatter_wrapper(self, inputs, target_gpus, dim=0):
        r"""
        Slices tensors into approximately equal chunks and
        distributes them across given GPUs. Duplicates
        references to objects that are not tensors.
        """

        def scatter_map(obj):
            if isinstance(obj, torch.Tensor):
                return Scatter.apply(target_gpus, None, dim, obj)
            if isinstance(obj, tuple) and len(obj) > 0:
                return list(zip(*map(scatter_map, obj)))
            if isinstance(obj, list) and len(obj) > 0:
                size = len(obj) // len(target_gpus)
                return [obj[i * size:(i + 1) * size] for i in range(len(target_gpus))]
            if isinstance(obj, dict) and len(obj) > 0:
                return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
            return [obj for targets in target_gpus]

        # After scatter_map is called, a scatter_map cell will exist. This cell
        # has a reference to the actual function scatter_map, which has references
        # to a closure that has a reference to the scatter_map cell (because the
        # fn is recursive). To avoid this reference cycle, we set the function to
        # None, clearing the cell
        try:
            return scatter_map(inputs)
        finally:
            scatter_map = None

    def scatter(self, inputs, kwargs, device_ids):
        return self.scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def gather(self, outputs, output_device):
        def gather_map(outputs):
            out = outputs[0]
            if isinstance(out,BUABoxes):
                return BUABoxes(Gather.apply(output_device, self.dim, *[o.tensor for o in outputs]))
            if isinstance(out, torch.Tensor):
                return Gather.apply(output_device, self.dim, *outputs)
            if out is None:
                return None
            if isinstance(out, dict):
                if not all((len(out) == len(d) for d in outputs)):
                    raise ValueError('All dicts must have the same number of keys')
                return type(out)(((k, gather_map([d[k] for d in outputs]))
                                  for k in out))
            return type(out)(map(gather_map, zip(*outputs)))

        # Recursive function calls like this create reference cycles.
        # Setting the function to None clears the refcycle.
        try:
            res = gather_map(outputs)
        finally:
            gather_map = None
        return res

class CoCaDataFlow(RNGDataFlow):
    """
    """

    def __init__(self, cfg, args, shuffle=False):
        self.shuffle = shuffle
        self.cfg = cfg
        self.non_batched_img_to_input_df = CoCaInputDataflow(FGS.image_dir, FGS.max_nb_images)
        self.img_to_input_df = BatchData(self.non_batched_img_to_input_df, FGS.batch_size,
                                         use_list=True)
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
        self.model = DataParallelV2(self.model)


    def __len__(self):
        return int(len(self.non_batched_img_to_input_df) / len(self.model.device_ids))

    def __iter__(self):
        for data_dict in self.img_to_input_df.get_data():

            rcnn_input = [{k: list(v)[i] for k, v in data_dict.items()} for i, _ in enumerate(
                list(data_dict.values())[0])]  # tensorpack dataflow gives dict of lists, rcnn expects list of dicts
            with torch.set_grad_enabled(False):
                model_outputs = list(self.model(rcnn_input))  # boxes, scores, features_pooled
            for i, o in enumerate(model_outputs):
                model_outputs[i] = [e.cpu() if type(e) is torch.Tensor else e.tensor.cpu() for e in o]
            assert len(model_outputs) == 3, "Nathan: Make sure attribute_extraction is turned off"
            boxes, scores, feats = model_outputs

            keep_idxs = []
            keep_feats = []
            keep_og_boxes = []
            keep_scores = []
            num_boxes = []
            for single_boxes, single_feats, single_scores, scale in zip(boxes, feats, scores, data_dict[
                'im_scale']):
                og_boxes = single_boxes / scale  # Nathan
                keep_idxs_single = filter_keep_boxes(og_boxes, self.cfg, single_scores)
                keep_idxs.append(keep_idxs_single)

                keep_feats.append(single_feats[keep_idxs_single])
                keep_og_boxes.append(og_boxes[keep_idxs_single])
                keep_scores.append(single_scores[keep_idxs_single].numpy())
                num_boxes.append(len(keep_idxs_single))

            # return image_bboxes, image_feat, info, keep_boxes
            # image_bboxes, image_feat, info, keep_boxes = prep_roi_features(None, boxes, self.cfg, rcnn_input,
            #                                                                features_pooled, im, img_file, scores)
            # return image_bboxes, image_feat, info, keep_boxes
            # image_bboxes, image_feat, info, keep_boxes = image_to_intermediate(self.cfg, im, img_file, self.model) #TODO fix "RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method"
            for f, s, b, nb, h, w, i, c in zip(keep_feats, keep_scores, keep_og_boxes, num_boxes,
                                               data_dict['img_height'], data_dict['img_width'], data_dict['img_id'],
                                               data_dict['caption']):
                yield [f, s, b, nb, h, w, i, c]
            # yield [keep_feats, keep_scores, keep_og_boxes, num_boxes, data_dict['img_height'], data_dict['img_width'],
            #        data_dict['img_id'], data_dict['caption']]
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
    start = time()
    cfg = setup(args)
    print(f"Setup done after {time() - start}")
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

    print(f"Conceptual_Caption init done after {time() - start}")
    LMDBSerializer.save(ds, str(Path(ROOT_DIR, 'DeVLBert',
                                      f'features_lmdb/CC/training_feat_{args.num_samples if args.num_samples > 0 else "all"}_debug_{int(time())}.lmdb')))
    print(f"Done after {time() - start}")


if __name__ == '__main__':
    main()
