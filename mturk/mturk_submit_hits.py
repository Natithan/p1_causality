import re

import random

import json

import os
import ray

from pathlib import Path

import sys
from absl import flags

from constants import URL_PATH, MTURK_DIR
from easyturk import EasyTurk
from util import index_df_column, open_tsv, TIME
import pandas as pd

ROOT_DIR = "/cw/liir/NoCsBack/testliir/nathan/p1_causality"
# region Flags stuff
FGS = flags.FLAGS
flags.DEFINE_bool("sandbox", True, "") #!!!! TEST IF THIS SHOULD BE FALSE, NOT SO FOR DEBUGGING !!!
flags.DEFINE_integer("max_assignments", 5, "")
flags.DEFINE_integer("nb_data", -1, "")
flags.DEFINE_float("reward", .4, "") # for tasks_per_hit labels
flags.DEFINE_integer("tasks_per_hit", 9, "") #From 10 onwards, amazon charges extra :P
flags.DEFINE_integer("test_percentage", 80, "")
flags.DEFINE_integer("min_percent_approved", 85, "")
flags.DEFINE_integer("min_hits_approved", 50, "")
FGS(sys.argv)


# endregion
def launch_causal_annotation(data, reward=1.00, tasks_per_hit=10, sandbox=False):
    et = EasyTurk(sandbox=sandbox)

    template = 'annotate_causal_pair.html'
    hit_ids = []
    i = 0
    with open(Path(MTURK_DIR, f"test_qualification_response{'_sandbox' if sandbox else ''}.json"), "r") as f:
        response = json.loads(''.join(f.readlines()))
        test_qual_id = response['QualificationType']['QualificationTypeId']
    while i < len(data):
        hit = et.launch_hit(
            template, data[i:i+tasks_per_hit], reward=reward,
            title=f'Say whether one object in a scene "causes" another{TIME if FGS.sandbox else ""}',
            description=('Say whether you think intervening on the presence of an object in a scene would have'
                         ' consequences on the probability of the presence of another object'),
            keywords='causation, image, objects',
            max_assignments=FGS.max_assignments,
            hits_approved=FGS.min_hits_approved,
            percent_approved=FGS.min_percent_approved,
            test_percentage=FGS.test_percentage,
            test_qual_id=test_qual_id
        )
        hit_id = hit['HIT']['HITId']
        hit_ids.append(hit_id)
        i += tasks_per_hit
    return hit_ids

def main():
    url_for_id = get_url_for_id()

    et = EasyTurk(sandbox=FGS.sandbox)
    print(et.get_account_balance())
    input_dir = Path(ROOT_DIR, 'mturk', 'input_mturk')

    datas = []
    for ranking in ['absolute','relative']:
        modality = 'image'

        filter = ''
        if ranking == 'relative':
            filter = '_positive'
        filepath = Path(input_dir, f"{ranking}_{modality}{filter}.tsv")
        data = csv_to_mturk_input(filepath, url_for_id)
        data = data[:FGS.nb_data if FGS.nb_data > 0 else 500]
        # Filter out pairs like ('mountain','mountains')
        data = [d for d in data if not ((d['word_X'] == d['word_Y'] + 's') or (d['word_Y'] == d['word_X'] + 's'))]
        datas += data
    random.shuffle(datas)
    df = pd.DataFrame(datas)
    df.to_csv(Path(MTURK_DIR, 'logs', f'data_for_mturk_{TIME}.tsv'), sep='\t')
    launch_causal_annotation(datas, reward=FGS.reward, tasks_per_hit=FGS.tasks_per_hit, sandbox=FGS.sandbox)


def csv_to_mturk_input(filepath, url_for_id):
    df = pd.read_csv(filepath,
                     sep='\t',
                     converters={col: lambda x: json.loads(
                         re.sub(r'(?<=\w)\'(?=\w)', '@@TMP@@',
                                x)  # to deal with e.g. '(\'outfield\', "pitcher\'s mound")'
                             .replace('\'', '\"')
                             .replace('\\\'', '\"')
                             .replace('(', '[')
                             .replace(')', ']')
                             .replace('@@TMP@@', '\''))
                                 for col in ['words', 'joint_image_ids', 'marginal_img_ids']})
    data = [
        {
            'joint_url': url_for_id[random.choice(d['joint_image_ids'])] if d['joint_image_ids'] else '',
            'marginal_url_x': url_for_id[random.choice(d['marginal_img_ids'][0])],
            'marginal_url_y': url_for_id[random.choice(d['marginal_img_ids'][1])],
            'word_X': d['words'][0],
            'word_Y': d['words'][1],
        } for _, d in df.iterrows()
    ]
    return data


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
