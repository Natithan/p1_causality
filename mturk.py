import random

import json

import os
import ray

from time import time

from pathlib import Path

import sys
from absl import flags

from constants import STORAGE_DIR, URL_PATH
from easyturk import EasyTurk
import interface
from raw__img_text__to__lmdb_region_feats_text import open_tsv, index_df_column
import pandas as pd

ROOT_DIR = "/cw/liir/NoCsBack/testliir/nathan/p1_causality"

# region Flags stuff
FGS = flags.FLAGS
flags.DEFINE_bool("sandbox", True, "")
flags.DEFINE_integer("max_assignments", 5, "")
FGS(sys.argv)


# endregion
def launch_causal_annotation(data, reward=1.00, tasks_per_hit=10, sandbox=False):
    et = EasyTurk(sandbox=sandbox)
    template = 'write_caption.html'
    hit_ids = []
    i = 0
    while i < len(data):
        hit = et.launch_hit(
            template, data[i:i+tasks_per_hit], reward=reward,
            title='Say whether one object in a scene "causes" another',
            description=('Say whether you think intervening on the presence of an object in a scene would have'
                         ' consequences on the probability of the presence of another object'),
            keywords='causation, image, objects',
            max_assignments=FGS.max_assignments
        )
        hit_id = hit['HIT']['HITId']
        hit_ids.append(hit_id)
        i += tasks_per_hit
    return hit_ids

def main():
    url_for_id = get_url_for_id()

    et = EasyTurk(sandbox=FGS.sandbox)
    print(et.get_account_balance())
    input_dir = Path(ROOT_DIR, 'input_mturk')
    ranking = 'absolute'
    modality = 'image'
    df = pd.read_csv(Path(input_dir, f"{ranking}_{modality}.tsv"),
                     sep='\t',
                     converters={col: lambda x: json.loads(x.replace('\'', '\"').replace('(', '[').replace(')', ']'))
                                 for col in ['words','joint_image_ids', 'marginal_img_ids']})
    data = [
        {
            'joint_url': url_for_id[random.choice(d['joint_image_ids'])] if d['joint_image_ids'] else '',
            'marginal_url_x': url_for_id[random.choice(d['marginal_img_ids'][0])],
            'marginal_url_y': url_for_id[random.choice(d['marginal_img_ids'][1])],
            'word_X':d['words'][0],
            'word_Y':d['words'][1],
         } for _, d in df.iterrows()
    ]
    launch_causal_annotation(data, reward=1, tasks_per_hit=10, sandbox=FGS.sandbox)
    hit_ids = [h['HITId'] for h in et.mtc.list_reviewable_hits()['HITs']]

    results = interface.fetch_completed_hits(hit_ids, approve=False, sandbox=FGS.sandbox)
    old_hids = [h["HITId"] for h in et.list_hits() if h["Title"] == 'Caption some pictures']
    for h in old_hids:
        et.delete_hit(h)

    for hit_id in hit_ids:
        et.approve_hit(hit_id)


def get_url_for_id():
    if os.path.exists(URL_PATH):
        print(f"Not storing {URL_PATH.name}, already present at {URL_PATH}")
        return json.load(open(URL_PATH, 'r'))
    else:
        df = open_tsv(Path(ROOT_DIR, 'DeVLBert/tools/DownloadConcptualCaption/Train_GCC-training.tsv'), 'training')
        print("Indexing urls ...")
        ray.init()
        futures = [index_df_column.remote(df[i::FGS.num_cpus], 'url') for i in range(FGS.num_cpus)]
        l = ray.get(futures)
        url_for_id = {}
        for d in l:
            url_for_id = {**url_for_id, **d}
        print("Done")
        print(f"Storing {URL_PATH} ...")
        json.dump(url_for_id, open(URL_PATH, 'w'))
        print(f"Done")
        return url_for_id


if __name__ == '__main__':
    main()
