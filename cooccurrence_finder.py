import ray
import time
from argparse import Namespace
import numpy as np
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
                 resume=False
                 )


def main():  # TODO make this (a lot) faster
    cfg = setup(args)
    ds = CoCaDataFlow(cfg, args)
    if args.num_cpus != 0:
        ray.init(num_cpus=args.num_cpus)
    else:
        ray.init()
    cooc_matrix_list = []
    print("Starting co-occurrence counting")
    start = time.time()
    cooc_matrix = get_cooc_matrix.remote(ds)
    print(time.time() - start)
    for i in range(num_gpus):
        cooc_matrix_list.append(get_cooc_matrix.remote(i, img_lists[i], cfg, args, actor))

    ray.get(cooc_matrix_list)
    np.save('cooc_matrix.npy', cooc_matrix)

@ray.remote(num_cpus=8, num_gpus=1)
def get_cooc_matrix(split_idx, ds):
    for e, dp in enumerate(ds.get_data()):
        if e > 30:
            break
        cls_probs = dp[1]
        cooc_matrix = np.zeros((cls_probs.shape[-1], cls_probs.shape[-1]))
        cls_idxs = np.argmax(cls_probs, 1)
        seen = []
        for n, i in enumerate(cls_idxs):
            if i in seen:  # We don't care about multiple co-occurrences: we only care about either [zero] or [one or more] co-occurrences
                continue
            seen.append(i)
            for j in cls_idxs[n:]:
                if j in seen:  # This also means we don't count self-co-occurrences, as they are always true anyway
                    continue
                if i > j:
                    cooc_matrix[j, i] += 1
                else:
                    cooc_matrix[i, j] += 1
    return cooc_matrix


if __name__ == '__main__':
    main()
