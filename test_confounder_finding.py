# region imports etc
import warnings

import numpy as np
import jsonargparse
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import requests
import PIL
from PIL import Image
from io import BytesIO

warnings.simplefilter(action='ignore', category=FutureWarning)
import os, glob
from constants import DEVLBERT_ROOT_DIR, LMDB_PATHS, MTURK_DIR, ARWEN_LMDB_PATHS, PROJECT_ROOT_DIR, STORAGE_DIR, URL_PATH
# export PYTHONPATH=/cw/liir/NoCsBack/testliir/nathan/p1_causality/DeVLBert:$PYTHONPATH
# os.environ['PYTHONPATH'] = f"{DEVLBERT_ROOT_DIR}:{os.environ['PYTHONPATH']}"

import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 700
import random
from pathlib import Path
import numpy as np
from pretorch_util import get_really_free_gpus
myrank = 0
os.environ['CUDA_VISIBLE_DEVICES'] = ",".join([str(i) for i in get_really_free_gpus()][myrank])
MASTER_PORT = f'{12355 + myrank}'
import torch
import torch.nn.functional as F
from pytorch_pretrained_bert import BertTokenizer
import torch.multiprocessing as mp
import torch.distributed as dist
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
from util import CLASSES, get_rank, my_maybe_print

os.chdir(DEVLBERT_ROOT_DIR)

from DeVLBert.devlbert.datasets import ConceptCapLoaderTrain
from DeVLBert.devlbert.devlbert import BertForMultiModalPreTraining, BertConfig
from scipy.stats import hypergeom
from time import time as t, sleep, strftime

# endregion

OG_DEVLBERT_PATH = Path(DEVLBERT_ROOT_DIR, "save/devlbert/pytorch_model_11.bin")
MY_DEVLBERT_PATH = Path(
    "/cw/working-gimli/nathan/devlbert_ckpts/pt_2_devlbert_base_real_try_4_again/pytorch_model_11.bin")
TMP_DEVLBERT_PATH = Path(
    "/cw/working-gimli/nathan/devlbert_checkpunten/epoch=6-step=70588-0.3.ckpt")
CFG_PATH = Path(DEVLBERT_ROOT_DIR, "config/bert_base_6layer_6conect.json")
GROUND_TRUTH_PATH = Path(MTURK_DIR, 'output_mturk', 'pair_annotations_0.8_no_cfdnce_weight.tsv')
MANUALLY_CHECKED_INCORRECT_RPN_IDS = ['591604074','204402773','2089931898','1514188503','1523504137'] # Especially 'window' is often incorrect
# STATISTIC = 'mAP'
# STATISTIC = 'avgAtt'

PRETRAINED_PATH = TMP_DEVLBERT_PATH  # MY_DEVLBERT_PATH OG_DEVLBERT_PATH TMP_DEVLBERT_PATH
QUAL_EXAMPLE_CLASSES = ['man','building','woman','tree','window','shirt','sky','wall','hair','head']
CL2ID = {cl:idx for idx,cl in enumerate(CLASSES)}
QUAL_EXAMPLE_IDS = [CL2ID[cl] for cl in QUAL_EXAMPLE_CLASSES]
# DATASET = 'coca'
# DATASET = 'flickr30k'

# OUT_DIR = Path(PROJECT_ROOT_DIR, f'{STATISTIC}_output')
# OUT_DIR = Path(PROJECT_ROOT_DIR, f'{STATISTIC}_output_debug')

BATCH_SIZE = 32
NB_CAUSES = 4

class RandomAPExact:
    def __init__(self):
        self.result_cache = {}

    def randomAPExact(self, N, R):
        if (N, R) in self.result_cache:
            return self.result_cache[(N, R)]
        else:  # From https://ufal.mff.cuni.cz/pbml/103/art-bestgen.pdf
            ap = 0
            for i in range(1, R + 1):
                for n in range(i, N - R + i + 1):
                    ap += hypergeom(N, R, n).pmf(i) * (i / n) * (i / n)
            ap /= R
            self.result_cache[(N, R)] = ap
            return ap


def add_program_argparse_args(parser):
    parser.add_argument(
        "--checkpoint",
        default=PRETRAINED_PATH,
        type=str,
        # required=True,
        help="The checkpoint for which to get the mAP score.",
    )
    parser.add_argument(
        "--run",
        default='unnamed_run',
        type=str,
        # required=True,
        help="Name of the run from which the checkpoint is taken",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        # required=True,
        help="The directory where to save output score files",
    )
    parser.add_argument(
        "--statistic",
        default='mAP',
        type=str,
        # required=True,
        help="What to calculate. Options are 'mAP' and 'avgAtt'",
    )
    parser.add_argument(
        "--dataset",
        default='coca',
        type=str,
        # required=True,
        help="Which source LMDB-stored RPN-extracted features to use. Options are 'coca' and 'flickr30k'",
    )
    parser.add_argument(
        "--max_t",
        default=-1,
        type=int,
        # required=True,
        help="If not doing a full run, max time to iterate for",
    )
    parser.add_argument(
        "--batch_size",
        default=BATCH_SIZE,
        type=int,
        # required=True,
        help="If not doing a full run, max time to iterate for",
    )
    return parser


def main_single_process(rank, world_size, run_id):
    parser = jsonargparse.ArgumentParser()
    parser = add_program_argparse_args(parser)
    args = parser.parse_args()
    AA = (args.statistic == 'avgAtt')
    setup(rank=rank, world_size=world_size)
    model = BertForMultiModalPreTraining.from_pretrained(args.checkpoint, config=BertConfig.from_json_file(CFG_PATH))
    device = torch.device(f"cuda:{rank}")
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased", do_lower_case=True
    )

    if args.dataset == 'coca':
        pass
        train_dataset = ConceptCapLoaderTrain(
            tokenizer,
            lmdb_paths=LMDB_PATHS,
            seq_len=36,
            batch_size=args.batch_size,
            predict_feature=False,
            shuffle=False,
            num_workers=0
        )
    elif args.dataset == 'flickr30k':
        from devlbert.datasets.retreival_dataset import MyRetreivalDataset
        from torch.utils.data import DataLoader
        train_dataset = \
            DataLoader(MyRetreivalDataset(
                task='RetrievalFlickr30k',
                dataroot='/cw/working-gimli/nathan/downstream_data/datasets/flickr30k',
                annotations_jsonpath='/cw/working-gimli/nathan/downstream_data/datasets/flickr30k/all_data_final_train_2014.jsonline',
                split='train',
                tokenizer=BertTokenizer,
                max_seq_length=30), batch_size=args.batch_size)
    else:
        raise ValueError
    # train_dataset = flickr30k_dataset

    mAP_dict = {
        'mAP_devlbert': [],
        'mAP_baseline': [],
        'mAP_baseline_emp': []
    }

    mAP_per_class_dict = {
        'mAP_devlbert': dict(zip(CLASSES, [[] for _ in range(len(CLASSES))])),
        'mAP_baseline': dict(zip(CLASSES, [[] for _ in range(len(CLASSES))])),
        'mAP_baseline_emp': dict(zip(CLASSES, [[] for _ in range(len(CLASSES))]))
    }

    rAPExact = RandomAPExact()
    gt_for_pair = pd.read_csv(GROUND_TRUTH_PATH, sep='\t')
    gt_for_pair['ID_X'] = gt_for_pair.apply(lambda row: word_to_id(row['word_X']), axis=1)
    gt_for_pair['ID_Y'] = gt_for_pair.apply(lambda row: word_to_id(row['word_Y']), axis=1)
    gt = GT(gt_for_pair=gt_for_pair)
    sb = t()
    START_T = t()  # TODO only consider non-masked tokens in mAP count?
    attentionAndCount_for_classid = {}
    projyAndCount_for_classid = {}

    # TEMP_LMDB_PATHS = [f"/cw/working-arwen/nathan/features_CoCa_lmdb/full_coca_36_{i}_of_4.lmdb" for i in range(4)]

    # for LMDB_PATH in TEMP_LMDB_PATHS: # Temp fix while downloading all to arwen
    #     print("TEMP GOING OVER LMDB PATHS ONE BY ONE WHILE WAITING FOR THEM TO DOWNLOAD")
    #     if DATASET == 'coca':
    #         train_dataset = ConceptCapLoaderTrain(
    #             tokenizer,
    #             lmdb_paths=[LMDB_PATH],
    #             seq_len=36,
    #             batch_size=args.batch_size,
    #             predict_feature=False,
    #             shuffle=False,
    #             num_workers=0,
    #             caption_path='/cw/working-arwen/nathan/features_CoCa_lmdb/caption_train.json'
    #         )
    LAST_SAVE_T = t()
    if args.statistic == 'qual_avgAtt':

        QUAL_IMAGES_DIR = Path(PROJECT_ROOT_DIR, 'qual_images')
        import json
        # bc2img_id = {}
        # used_img_ids = set()

        used_effect_cls = set()
        rows = pd.DataFrame(columns=['Effect variable', 'Top cause variables', 'Scores'])
        used_img_ids = set()
        id2url = json.load(open(URL_PATH, 'r'))

        if not os.path.exists(QUAL_IMAGES_DIR):
            os.makedirs(QUAL_IMAGES_DIR)
    for batch_num, batch in enumerate(tqdm(train_dataset,
                                           desc=f'Rank {rank}')):
        if args.max_t > 0:
            if t() - START_T > args.max_t:
                print("=" * 50, "Stopping early for debugging", "=" * 50)
                break
        # if t() - LAST_SAVE_T > 15000:
        #     LAST_SAVE_T = t()
        #     # save_stuff(args, attentionAndCount_for_classid, batch_num, gt, mAP_per_class_dict, rank, run_id,extra_name=os.path.basename(LMDB_PATH))
        #     save_stuff(args, attentionAndCount_for_classid, batch_num, gt, mAP_per_class_dict, rank, run_id)
        # print("\r\nbatch load time", t()-sb)
        batch = tuple(tp.to(device, non_blocking=True) for tp in batch)
        if type(train_dataset) == ConceptCapLoaderTrain:
            input_ids, input_mask, segment_ids, _, _, image_feat, image_loc, image_target, _, image_mask, \
            image_ids, _, _ = batch
        else:
            image_feat, image_loc, image_mask, input_ids, input_mask, segment_ids, image_ids, image_target = batch
        # #For debugging: checking
        # model(
        #             input_ids,
        #             image_feat,
        #             image_loc,
        #             segment_ids,
        #             input_mask,
        #             image_mask,
        #             lm_label_ids,
        #             image_label,
        #             image_target,
        #             is_next,
        #             causal_label_t,
        #             causal_label_v)

        s = t()
        # print(f'rank {rank} pid {os.getpid()},{strftime("%d %H:%M:%S")}: ' + str(image_ids))
        sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v, all_attention_mask = model.bert(
            input_ids,
            image_feat,
            image_loc,
            segment_ids,
            input_mask,
            image_mask,
            output_all_encoded_layers=False,
            output_all_attention_masks=None
        )

        y = sequence_output_v[:, 1:].to(device)

        # Reshape data for batch operation
        y = y.reshape(y.shape[0] * y.shape[1], -1)
        image_target = image_target.reshape(image_target.shape[0] * image_target.shape[1], -1)
        projected_y = model.causal_v.Wy(y)
        projected_z = model.causal_v.Wz(model.causal_v.dic_z.to(device))
        attention = torch.mm(projected_y, projected_z.t()) / (
                model.causal_v.embedding_size ** 0.5)
        attention = F.softmax(attention, 1)  # torch.Size([batch_size * nb_bbox, 1601])
        # batch_subsample
        # z_hat = attention[:2.unsqueeze(2) * model.causal_v.dic_z.unsqueeze(0)  # torch.Size([box, 1601, 2048])
        # z = torch.matmul(model.causal_v.prior.unsqueeze(0), z_hat).squeeze(1)  # torch.Size([box, 1, 2048])->torch.Size([box, 2048])
        # Most-matched element in the confounder dictionary is claimed to be most likely class to form
        # a confounder for ?r and r?
        _, max_attented_ids = attention.max(-1)
        _, max_box_ids = image_target.max(-1)
        s = t()

        sorted_cause_scores, sorted_causes_ids = attention.sort(-1, descending=True)
        # print("sorted_cause_scores, sorted_causes_ids",t() - s)
        max_attented_classes = [CLASSES[i] for i in max_attented_ids.tolist()]
        max_box_classes = [CLASSES[i] for i in max_box_ids.tolist()]
        # sorted_attended_classes = [[CLASSES[i]  for i in classes_for_box.tolist()] for classes_for_box in sorted_causes_ids]
        # confounders_for_objects_for_id = (int(image_ids[b_idx]), [(b, a) for b, a, m in
        #                                                           zip(max_box_classes, max_attented_classes,
        #                                                               image_mask[b_idx]) if m != 0])
        # print("Forward",t() - s)
        if args.statistic == 'tsne':
            # show_TSNE(model.causal_v.dic_z,CLASSES)
            projyAndCount_for_classid = {}
            for clid, vector in zip(max_box_classes, projected_y):
                if clid in projyAndCount_for_classid:
                    projyAndCount_for_classid[clid] = {'count': projyAndCount_for_classid[clid]['count'] + 1,
                                                       'sumvec': projyAndCount_for_classid[clid]['sumvec'] + vector}
                else:
                    projyAndCount_for_classid[clid] = {'count': 1, 'sumvec': vector}

        elif args.statistic == 'avgAtt':
            for i, classid in enumerate(max_box_ids):
                classid = int(classid)
                att = attention[i].detach().cpu().numpy()
                if classid in attentionAndCount_for_classid:
                    count = attentionAndCount_for_classid[classid]['count']
                    current_avg_att = attentionAndCount_for_classid[classid]['attention']
                    new_count = count + 1
                    new_avg_att = (att + current_avg_att * count) / new_count
                    attentionAndCount_for_classid[classid] = {'attention': new_avg_att, 'count': new_count}
                else:
                    print(f"Adding {CLASSES[classid]} to attentionAndCount_for_classid")
                    attentionAndCount_for_classid[classid] = {'attention': att, 'count': 1}

        elif args.statistic == 'qual_avgAtt':
            boxes_per_img = input_ids.shape[-1]
            expanded_img_ids = [str(img_id) for img_id in image_ids.tolist() for _ in range(boxes_per_img)]
            id2url = json.load(open(URL_PATH, 'r'))
            # for bc, img_id in zip(max_box_classes,expanded_img_ids):
            #     if bc in QUAL_EXAMPLE_CLASSES:
            #         if (bc not in bc2img_id) and (img_id not in used_img_ids) and (img_id in id2url):
            #             bc2img_id[bc] = img_id
            #             used_img_ids.add(img_id)
            for bc, img_id, top_cause_ids, top_cause_atts in zip(max_box_classes, expanded_img_ids,
                                                                 sorted_causes_ids[:, :NB_CAUSES].tolist(),
                                                                 sorted_cause_scores[:, :NB_CAUSES].tolist()):
                if bc in QUAL_EXAMPLE_CLASSES:
                    if (bc not in used_effect_cls) and \
                            (img_id not in used_img_ids) and \
                            (img_id in id2url) and \
                            (img_id not in MANUALLY_CHECKED_INCORRECT_RPN_IDS):
                        url = id2url[img_id]
                        try:
                            img = Image.open(BytesIO(requests.get(url).content))
                        except PIL.UnidentifiedImageError:
                            continue
                        top_cause_classes = [CLASSES[idx] for idx in top_cause_ids]
                        used_effect_cls.add(bc)
                        used_img_ids.add(img_id)
                        rows = rows.append({'Effect variable': bc, 'Top cause variables': tuple(top_cause_classes),
                                            'Scores': tuple(top_cause_atts), 'img': img}, ignore_index=True)

            if len(rows) < len(QUAL_EXAMPLE_CLASSES):
                continue
            else:
                rows['Effect variable'] = pd.Categorical(rows['Effect variable'],
                                                         QUAL_EXAMPLE_CLASSES)  # see https://stackoverflow.com/questions/13838405/custom-sorting-in-pandas-dataframe/27009771
                rows = rows.sort_values(['Effect variable'])
                for i, row in rows.iterrows():
                    object_cls = row['Effect variable']
                    img = row['img']
                    img_file_name = f'{object_cls}.png'
                    img.save(Path(QUAL_IMAGES_DIR,img_file_name))
                    rows.loc[i,'img_file'] = img_file_name
                rows.drop(['img'], axis=1).to_csv(Path(QUAL_IMAGES_DIR,'rows.csv'))
                return
                # img_array = np.array(imageio.imread(response.content).tolist())
        else:
            times = {}
            s = t()
            for box_idx, effect_object_id in enumerate(max_box_ids):
                g = t()
                effect_object_id = float(effect_object_id)
                add_or_append(times, 'float', t() - g)
                si = t()
                g = t()
                known_gts = gt.get_known_gts(effect_object_id)
                add_or_append(times, 'known_gts', t() - g)
                if len(known_gts) == 0:
                    continue
                g = t()
                current_sci = sorted_causes_ids[box_idx]
                current_scs = sorted_cause_scores[box_idx]
                add_or_append(times, 'current_sci,current_scs', t() - g)

                # print(list(zip(known_gts.cause_candidate, known_gts.cause_candidate_label,
                #                [dict(pred_causes)[w] for w in known_gts.cause_candidate])))

                # Treating confounder-attention as a ranking, where we want the ground-truth causes in our to be
                # ranked higher than the ground-truth-not-causes
                g = t()
                # TODO speed this up?
                result = sorted(
                    list(
                        zip(
                            known_gts.cause_candidate,
                            known_gts.cause_candidate_label,
                            [float(current_scs[torch.where(current_sci == i)]) for i in known_gts.cause_candidate]
                        )
                    ),
                    key=lambda r: r[-1], reverse=True)
                add_or_append(times, 'result', t() - g)

                g = t()
                nb_causes = len([r for r in result if r[1] == 'cause'])
                add_or_append(times, 'nb_causes', t() - g)
                if nb_causes == 0:
                    # print(f"No causes in database for {effect_object}")
                    continue
                else:
                    ss = t()
                    g = t()
                    expected_average_precision = rAPExact.randomAPExact(N=len(result),
                                                                        R=nb_causes)
                    add_or_append(times, 'EAP', t() - g)
                    AP_devlbert = average_precision(result)
                    AP_baseline = expected_average_precision
                    random.shuffle(result)  # Comparing to random; changes result in-place
                    AP_baseline_emp = average_precision(result)
                    mAP_dict['mAP_devlbert'].append(AP_devlbert)
                    mAP_dict['mAP_baseline'].append(AP_baseline)
                    mAP_dict['mAP_baseline_emp'].append(AP_baseline_emp)
                    mAP_per_class_dict['mAP_devlbert'][CLASSES[int(effect_object_id)]].append(AP_devlbert)
                    mAP_per_class_dict['mAP_baseline'][CLASSES[int(effect_object_id)]].append(AP_baseline)
                    mAP_per_class_dict['mAP_baseline_emp'][CLASSES[int(effect_object_id)]].append(AP_baseline_emp)
                    add_or_append(times, 'else', t() - g)

                    # print(confounders_for_objects_for_id)
                # print("for box_idx, effect_object_id",t()-s)
                #
                # for k, v in times.items():
                #     print((k, sum(v) / len(v)))
            sb = t()

    if args.statistic == 'tsne':
        py_ids,py_vecs = zip(*[(k,v['sumvec'] / v['count']) for k,v in projyAndCount_for_classid.items()])
        show_TSNE(torch.stack(py_vecs), py_ids, projected_z, CLASSES)
    save_stuff(args, attentionAndCount_for_classid, batch_num, gt, mAP_per_class_dict, rank, run_id)
    if world_size > 1:
        cleanup()


def save_stuff(args, attentionAndCount_for_classid, batch_num, gt, mAP_per_class_dict, rank, run_id, extra_name=''):
    if args.statistic == 'avgAtt':
        arr = np.zeros((1601, 1601)) - 1
        for k, v in attentionAndCount_for_classid.items():
            arr[k] = v['attention']
        avgAtt_df = pd.DataFrame(data=arr, index=CLASSES, columns=CLASSES)
        counts = [0] * len(CLASSES)
        for id in attentionAndCount_for_classid:
            counts[id] = attentionAndCount_for_classid[id]['count']
        avgAtt_df.insert(0, "Counts", counts, True)
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
        avgAtt_output_file = Path(args.out_dir, f'avgAtt_{run_id}_{batch_num}_rank{rank}.csv')
        print(f'Saving to {avgAtt_output_file}')
        avgAtt_df.to_csv(avgAtt_output_file)

        sorted_df = avgAtt_df.sort_values(by='Counts', ascending=False)
        a = {sorted_df.index[i]: sorted_df.iloc[i].iloc[1:].sort_values(ascending=False).iloc[:5] for i in range(10)}
        out_df = pd.DataFrame(
            {effect: [f'{cls}: {str(round(score, 3))}' for cls, score in zip(causes.index, causes)] for effect, causes
             in a.items()}).transpose()
        with pd.option_context("max_colwidth", 1000):
            print(out_df.to_latex(header=False, bold_rows=True))
            print(out_df.to_latex(header=False, bold_rows=True),
                  file=open(Path(args.out_dir, f'latex_top_avgAtt_{run_id}.txt'), 'w'))

    else:

        per_occured_class_dict_to_store = {
            model: {(cls, len(avpres),
                     str(gt.get_known_gts(word_to_id(cls))[['word_X', 'word_Y', 'cause_candidate_label']]).replace('\n',
                                                                                                                   '\__')): sum(
                avpres) / len(avpres)
                    for cls, avpres in
                    per_model_values.items() if len(avpres) > 0}
            for model, per_model_values in mAP_per_class_dict.items()
        }
        avg_dict_to_store = {m: [sum(d.values()) / len(d)] for m, d in
                             per_occured_class_dict_to_store.items()}

        # dict_to_store = {k: [sum(v) / len(v)] for k, v in mAP_dict.items()}

        for name, dic in zip(['avg', 'per_class'], [avg_dict_to_store, per_occured_class_dict_to_store]):
            dic['batch_num'] = batch_num
            df_to_store = pd.DataFrame(dic)
            df_to_store['excess_mAP'] = df_to_store['mAP_devlbert'] - df_to_store['mAP_baseline']
            if name == 'per_class':
                df_to_store['count'] = [i[1] for i in df_to_store.index]
                df_to_store['turker_info'] = [i[2] for i in df_to_store.index]
                df_to_store.index = [i[0] for i in df_to_store.index]
            df_to_store.sort_values(by='excess_mAP', inplace=True)
            if not os.path.exists(args.out_dir):
                os.makedirs(args.out_dir)
            # output_file = Path(args.out_dir, f'{name}_mAP_comparison_{run_id}_{batch_num}_rank{rank}_{"my_devlbert" if PRETRAINED_PATH == MY_DEVLBERT_PATH else "og_devlbert"}.csv')
            output_file = Path(args.out_dir, f'{name}_mAP_comparison_{run_id}_{batch_num}_rank{rank}{extra_name}.csv')
            pd.set_option('display.max_colwidth', None)
            print(df_to_store)
            df_to_store.to_csv(output_file, index=False if (name == 'avg') else True)


class GT:

    def __init__(self, gt_for_pair):
        self.cache = {}
        self.gt_for_pair = gt_for_pair

    def get_known_gts(self, effect_object_id):
        if effect_object_id in self.cache:
            return self.cache[effect_object_id]
        else:
            known_gts = self.gt_for_pair[
                (effect_object_id == self.gt_for_pair['ID_X']) | (effect_object_id == self.gt_for_pair['ID_Y'])]
            # g = t()
            known_gts['cause_candidate'] = known_gts['ID_X'].where(known_gts['ID_Y'] == effect_object_id,
                                                                   known_gts['ID_Y'])
            # times['known_gts["cause_candidate"]'] = (t() - g)
            # g = t()
            known_gts['cause_candidate_label'] = np.where(
                (known_gts['ID_X'] == known_gts['cause_candidate']) & (known_gts['max_resp'] == 'x-to-y'),
                'cause', np.where(known_gts['max_resp'] == 'z-to-xy', 'mere_correlate', 'effect'))
            # times['known_gts["cause_candidate_label"]'] = (t() - g)
            self.cache[effect_object_id] = known_gts
            return known_gts


def average_precision(result):
    if len([row for row in result if row[1] == 'cause']) != 0:
        return sum([precision_at_k(result, k) * rel_at_k(result, k) for k in range(len(result))]) / len(
            [row for row in result if row[1] == 'cause'])
    else:
        return None


def temp(attention, max_box_ids):
    attentionAndCount_for_classid = {}
    for i, classid in enumerate(max_box_ids):
        classid = int(classid)
        att = attention[i].detach().cpu().numpy()
        if classid in attentionAndCount_for_classid:
            count = attentionAndCount_for_classid[classid]['count']
            current_avg_att = attentionAndCount_for_classid[classid]['attention']
            new_count = count + 1
            new_avg_att = (att + current_avg_att * count) / new_count
            attentionAndCount_for_classid[classid] = {'attention': new_avg_att, 'count': new_count}
        else:
            attentionAndCount_for_classid[classid] = {'attention': att, 'count': 1}

    import numpy as np

    arr = np.zeros((1601, 1601))
    arr -= 1
    for k, v in attentionAndCount_for_classid.items():
        arr[k] = v['attention']
    return arr


def precision_at_k(result, k):
    return len([row for row in result[:k + 1] if row[1] == 'cause']) / (k + 1)


def rel_at_k(result, k):
    return int(result[k][1] == 'cause')


def word_to_id(word: str):
    return CLASSES.index(word) if word in CLASSES else None


def add_or_append(d, k, v):
    d[k] = [v] if k not in d else d[k] + [v]


def main():
    # n_procs = len(get_free_gpus())
    n_procs = 1
    run_id = int(t())
    main_single_process(rank=0, world_size=1, run_id=run_id)
    # mp.spawn(main_single_process,
    #          args=(n_procs, run_id),
    #          nprocs=n_procs,
    #          join=True)
    # if not AA:
    #     pass
    # prestring = 'avgAtt_' if AA else 'mAP_comparison_'
    # partial_files = glob.glob(f'{args.out_dir.as_posix()}/{prestring}{run_id}_*')
    # assert len(partial_files) == n_procs
    # dfs = [pd.read_csv(f) for f in partial_files]
    # avg = pd.concat(dfs)[['mAP_devlbert', 'mAP_baseline', 'mAP_baseline_emp']].mean()
    # avg.to_csv(Path(args.out_dir, f'mAP_comparison_full_{run_id}.csv'), header=False)


def setup(rank, world_size):
    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = MASTER_PORT

        # initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def reduce_dim(input_matrix, z_matrix,target_dim=2):
    total_matrix = torch.cat((input_matrix, z_matrix),dim=0)
    pca = PCA(50).fit_transform(total_matrix.cpu().detach())
    tsne_pca = TSNE(n_components=target_dim).fit_transform(pca)
    return tsne_pca[:len(input_matrix)],tsne_pca[len(input_matrix):]


def show_TSNE(input_matrix, input_annotations, z_matrix, z_annotations):

    input_reduced, z_reduced = reduce_dim(input_matrix, z_matrix,target_dim=2)
    fig, ax = plt.subplots()
    red = [255, 0, 0]
    blue = [0, 0, 255]
    green = [0, 255, 0]
    C_z = np.array([blue if txt != 'butter knife' else red for txt in z_annotations])

    C_input = np.array([green]*len(input_annotations))
    ax.scatter(input_reduced[:,0], input_reduced[:,1],c=C_input / 255,s=.5)
    ax.scatter(z_reduced[:,0], z_reduced[:,1],c=C_z / 255,s=.3)
    for annotations, reduced in zip((input_annotations,z_annotations),(input_reduced,z_reduced)):
        for i, txt in enumerate(annotations):
            ann = ax.annotate(txt, (reduced[:,0][i], reduced[:,1][i]))
            ann.set_fontsize(1)
    plt.show()
    print('plotted')

    # my_scatter_plot(a, b, annotations)
    # if exclude_background:
    #     zipped = [(aa, bb, tt) for aa, bb, tt in zip(a, b, annotations) if tt != 'background']
    #     unzipped = list(zip(*zipped))
    #     my_scatter_plot(unzipped[0], unzipped[1], unzipped[2])


def my_scatter_plot(a, b, annotations):
    fig, ax = plt.subplots()
    red = [255, 0, 0]
    blue = [0, 0, 255]
    C = np.array([blue if txt != 'butter knife' else red for txt in annotations])
    ax.scatter(a, b,c=C / 255)
    for i, txt in enumerate(annotations):
        ann = ax.annotate(txt, (a[i], b[i]))
        ann.set_fontsize(2)
    plt.show()


if __name__ == '__main__':
    main()


def tmpp():
    import lmdb
    env = lmdb.open(
        '/cw/working-gimli/nathan/downstream_data/datasets/flickr30k/flickr30k_resnet101_faster_rcnn_genome.lmdb',
        max_readers=1, readonly=True,
        lock=False, readahead=False, meminit=False)
    txn = env.begin(write=False)
    import numpy as np

    import base64

    els = []
    import pickle
    for i, (_, el) in enumerate(txn.cursor()):
        if i > 10:
            break
        item = pickle.loads(el)
        num_boxes = int(item['num_boxes'])
        item['features'] = np.frombuffer(base64.b64decode(item["features"]), dtype=np.float32).reshape(num_boxes, 2048)
        item['boxes'] = np.frombuffer(base64.b64decode(item['boxes']), dtype=np.float32).reshape(num_boxes, 4)
        item['cls_prob'] = np.frombuffer(base64.b64decode(item["cls_prob"]), dtype=np.float32).reshape(num_boxes, 1601)
        els.append(item)

