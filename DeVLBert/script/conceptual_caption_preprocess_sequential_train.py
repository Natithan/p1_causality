import os
from pathlib import Path

import numpy as np
from tensorpack.dataflow import RNGDataFlow, PrefetchDataZMQ
from tensorpack.dataflow import *
import lmdb
import json
import pdb
import csv

from tools.DownloadConcptualCaption.download_data import _file_name

FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features', 'cls_prob']
import sys
import pandas as pd
import zlib
import base64
ROOT_DIR = "/cw/liir/NoCsBack/testliir/nathan"
csv.field_size_limit(sys.maxsize)


def open_tsv(fname, folder):
    print("Opening %s Data File..." % fname)
    df = pd.read_csv(fname, sep='\t', names=["caption","url"], usecols=range(0,2),nrows=200) #Nathan edited
    df['folder'] = folder
    print("Processing", len(df), " Images:")
    return df
# def _file_name(row):
#     return "%s/%s" % (row['folder'], (zlib.crc32(row['url'].encode('utf-8')) & 0xffffffff))

class Conceptual_Caption(RNGDataFlow):
    """
    """
    def __init__(self, corpus_path, shuffle=False):
        """
        Same as in :class:`ILSVRC12`.
        """
        self.corpus_path = corpus_path
        self.image_paths = [Path(corpus_path,i) for i in os.listdir(self.corpus_path) if i.endswith(".npz")]

        self.shuffle = shuffle
        self.num_file = 16
        self.name = os.path.join(corpus_path, 'CC_resnet101_faster_rcnn_genome.tsv.%d')
        self.infiles = [self.name % i for i in range(self.num_file)]
        self.counts = []
        self.num_caps = len(self.image_paths) # TODO: compute the number of processed pics

        self.captions = {}
        df = open_tsv(Path(ROOT_DIR,'DeVLBert/tools/DownloadConcptualCaption/Train_GCC-training.tsv'), 'training')
        for i, img in enumerate(df.iterrows()):
            caption = img[1]['caption']#.decode("utf8")
            im_name = _file_name(img[1])
            image_id = im_name.split('/')[1]
            # image_id = str(i)
            self.captions[image_id] = caption

        json.dump(self.captions, open(Path(ROOT_DIR, 'DeVLBert', 'features_lmdb/CC/caption_train.json'), 'w'))

    def __len__(self):
        return self.num_caps

    def __iter__(self):
        for path in self.image_paths:
            item = np.load(path,allow_pickle=True)
            image_id = item['info'].item()['image_id']
            image_h = item['image_h'].item()
            image_w = item['image_w'].item()
            num_boxes = item['num_bbox'].item()
            boxes = item['bbox']
            features = item['x']
            cls_prob = item['info'].item()['obj_cls_prob']

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

if __name__ == '__main__':
    corpus_path = Path(ROOT_DIR,'buatest','features')
    ds = Conceptual_Caption(corpus_path)
    #TODO check the size on this
    ds1 = PrefetchDataZMQ(ds, nr_proc=1)
    LMDBSerializer.save(ds1, str(Path(ROOT_DIR,'DeVLBert','features_lmdb/CC/training_feat_all.lmdb')))
    # LMDBSerializer.save(ds1, '/mnt3/yangan.ya/features_lmdb/CC/training_feat_all.lmdb')
